// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Test cases for IQ3_XXS loading and decompression correctness.
// Validates:
//   1. Block-level dequantization against known reference values
//   2. Multi-block (tensor-level) dequantization consistency
//   3. IQ3XXSLinear op evaluate() produces correct MatMul results
//   4. Native path == decompose path (both produce identical results)
//   5. Shape inference and validation of the IQ3XXSLinear op

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "gguf_utils/iq3_xxs_decompose.hpp"
#include "openvino/op/iq3_xxs_linear.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/graph_util.hpp"

// Lookup tables from gguf_iq3_xxs.cpp
extern const uint32_t iq3xxs_grid[256];
extern const uint8_t ksigns_iq2xs[128];
extern const uint8_t kmask_iq2xs[8];

namespace {

// -------------------------------------------------------------------
// Helper: f16 <-> f32 conversions (matches the production code exactly)
// -------------------------------------------------------------------
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(1 - exp);
        mant >>= shift;
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

// -------------------------------------------------------------------
// Helper: Reference IQ3_XXS dequantization of a single block (256 weights)
// Matches the production code in gguf_iq3_xxs.cpp exactly.
// Returns f32 values for easier comparison.
// -------------------------------------------------------------------
static std::vector<float> dequant_single_block_f32(const uint8_t* block_data) {
    constexpr int QK_K = 256;
    std::vector<float> out(QK_K);

    uint16_t d_fp16;
    memcpy(&d_fp16, block_data, 2);
    const float d = fp16_to_fp32(d_fp16);

    const uint8_t* qs = block_data + 2;
    const uint8_t* scales_and_signs = qs + QK_K / 4;  // +64 bytes

    size_t oi = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        uint32_t aux32;
        memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
            const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 0]]);
            const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 1]]);

            for (int j = 0; j < 4; ++j)
                out[oi++] = db * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
            for (int j = 0; j < 4; ++j)
                out[oi++] = db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
        }
        qs += 8;
    }
    return out;
}

// -------------------------------------------------------------------
// Helper: Build a valid IQ3_XXS block with known scale and grid indices.
// This creates deterministic test data where we can verify the output.
// -------------------------------------------------------------------
static std::vector<uint8_t> make_test_block(float scale_value, uint8_t grid_idx_fill, uint8_t sign_code_fill) {
    constexpr int BLOCK_BYTES = 98;
    std::vector<uint8_t> block(BLOCK_BYTES, 0);

    // Set super-block scale (f16)
    uint16_t d_fp16 = fp32_to_fp16(scale_value);
    memcpy(block.data(), &d_fp16, 2);

    // Fill grid indices (64 bytes): all set to the same grid index
    uint8_t* qs = block.data() + 2;
    std::fill(qs, qs + 64, grid_idx_fill);

    // Fill scales_and_signs (32 bytes = 8 x uint32_t)
    // Each uint32 encodes: bits[0..27] = 4 x 7-bit sign codes, bits[28..31] = local scale (0-15)
    uint8_t* sas = qs + 64;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        // Build uint32_t: local_scale=0 (bits 28-31), sign_codes in bits 0-27
        uint32_t val = 0;
        // Pack 4 sign codes (7 bits each) into bits 0-27
        for (int l = 0; l < 4; ++l) {
            val |= (uint32_t)(sign_code_fill & 0x7F) << (7 * l);
        }
        // Local scale 0 (bits 28-31 stay 0)
        memcpy(sas + 4 * ib32, &val, 4);
    }

    return block;
}

// -------------------------------------------------------------------
// Helper: Build a multi-row IQ3_XXS compressed blob
// -------------------------------------------------------------------
static std::vector<uint8_t> make_test_blob(int64_t N, int64_t K, std::mt19937& rng) {
    constexpr int QK_K = 256;
    constexpr int BLOCK_BYTES = 98;
    const int64_t blocks_per_row = K / QK_K;
    const int64_t total_bytes = N * blocks_per_row * BLOCK_BYTES;

    std::vector<uint8_t> blob(total_bytes);

    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    std::uniform_int_distribution<int> idx_dist(0, 255);
    std::uniform_int_distribution<int> sign_dist(0, 127);
    std::uniform_int_distribution<int> lscale_dist(0, 15);

    for (int64_t row = 0; row < N; row++) {
        for (int64_t blk = 0; blk < blocks_per_row; blk++) {
            uint8_t* block_data = blob.data() + (row * blocks_per_row + blk) * BLOCK_BYTES;

            // Random super-block scale
            float s = scale_dist(rng);
            uint16_t d_fp16 = fp32_to_fp16(s);
            memcpy(block_data, &d_fp16, 2);

            // Random grid indices
            uint8_t* qs = block_data + 2;
            for (int i = 0; i < 64; ++i) {
                qs[i] = static_cast<uint8_t>(idx_dist(rng));
            }

            // Random scales_and_signs
            uint8_t* sas = qs + 64;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                uint32_t val = 0;
                for (int l = 0; l < 4; ++l) {
                    val |= (uint32_t)(sign_dist(rng) & 0x7F) << (7 * l);
                }
                val |= (uint32_t)(lscale_dist(rng)) << 28;
                memcpy(sas + 4 * ib32, &val, 4);
            }
        }
    }
    return blob;
}

// -------------------------------------------------------------------
// Helper: Compute reference MatMul Y = X @ W^T where W is decoded f32
// X: [M, K], W: [N, K] -> Y: [M, N]
// -------------------------------------------------------------------
static std::vector<float> reference_matmul(const float* X, const float* W, size_t M, size_t N, size_t K) {
    std::vector<float> Y(M * N, 0.0f);
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += X[m * K + k] * W[n * K + k];
            }
            Y[m * N + n] = acc;
        }
    }
    return Y;
}

}  // namespace

// ===================================================================
// TEST 1: Block-level dequantization — known uniform block
// ===================================================================
TEST(IQ3XXS, BlockDequant_UniformGridIndex0) {
    // Grid index 0 -> iq3xxs_grid[0] = 0x04040404
    //   -> 4 bytes: {0x04, 0x04, 0x04, 0x04} = {4, 4, 4, 4}
    // Sign code 0 -> ksigns_iq2xs[0] = 0 -> all positive
    // Local scale = 0 (from bits 28-31 of scales_and_signs)
    //   db = d * (0.5 + 0) * 0.5 = d * 0.25
    // Expected value: d * 0.25 * 4 = d * 1.0

    const float scale = 0.5f;
    auto block = make_test_block(scale, /*grid_idx=*/0, /*sign_code=*/0);
    auto values = dequant_single_block_f32(block.data());

    ASSERT_EQ(values.size(), 256u);

    // All values should be: scale * 0.25 * 4 = scale * 1.0 = 0.5
    const float expected = scale * (0.5f + 0.0f) * 0.5f * 4.0f;
    for (size_t i = 0; i < values.size(); i++) {
        EXPECT_NEAR(values[i], expected, 1e-4f)
            << "Mismatch at index " << i;
    }
}

TEST(IQ3XXS, BlockDequant_WithSignCode1) {
    // Grid index 0 -> all grid bytes = 4
    // Sign code 1 -> ksigns_iq2xs[1] = 129 = 0b10000001
    //   -> bit 0 set (sign for weight 0 of each group of 4 from grid1) -> negative
    //   -> bit 7 set (parity bit)
    // Local scale = 0: db = d * 0.25
    // For grid1: weight[0] -> negative, weight[1,2,3] -> positive
    // For grid2: weight[4] is bit 4=0 -> positive (wait, apply kmask_iq2xs[j+4]):
    //   grid2 signs use bits 4,5,6,7 of the sign byte
    //   ksigns_iq2xs[1]=129=0b10000001: bit7=1 -> grid2[j=3] is negative

    const float scale = 1.0f;
    auto block = make_test_block(scale, /*grid_idx=*/0, /*sign_code=*/1);
    auto values = dequant_single_block_f32(block.data());

    ASSERT_EQ(values.size(), 256u);

    const float db = scale * 0.25f;  // d * (0.5+0) * 0.5
    const float grid_val = 4.0f;     // grid byte = 0x04

    // Signs for sign_code=1: ksigns_iq2xs[1] = 129 = 0b10000001
    const uint8_t sign_byte = ksigns_iq2xs[1];  // 129
    EXPECT_EQ(sign_byte, 129u);

    // Check first group of 8 values (from the first sub-block's first l=0 group)
    // grid1 (4 values): signs applied via kmask_iq2xs[0..3] = {1,2,4,8}
    // grid2 (4 values): signs applied via kmask_iq2xs[4..7] = {16,32,64,128}
    float expected_signs[8];
    for (int j = 0; j < 4; j++)
        expected_signs[j] = (sign_byte & kmask_iq2xs[j]) ? -1.f : 1.f;
    for (int j = 0; j < 4; j++)
        expected_signs[4 + j] = (sign_byte & kmask_iq2xs[j + 4]) ? -1.f : 1.f;

    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(values[i], db * grid_val * expected_signs[i], 1e-5f)
            << "Mismatch at index " << i
            << " expected sign=" << expected_signs[i];
    }
}

TEST(IQ3XXS, BlockDequant_LocalScaleVariation) {
    // Test that local_scale (bits 28-31 of scales_and_signs) affects the output.
    // With local_scale = L: db = d * (0.5 + L) * 0.5

    const float d = 1.0f;
    constexpr int BLOCK_BYTES = 98;
    std::vector<uint8_t> block(BLOCK_BYTES, 0);

    uint16_t d_fp16 = fp32_to_fp16(d);
    memcpy(block.data(), &d_fp16, 2);

    // Grid index 0 (grid byte = 4), no signs
    uint8_t* qs = block.data() + 2;
    std::fill(qs, qs + 64, 0);  // grid index 0

    // Set different local scales for each sub-block
    uint8_t* sas = qs + 64;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t val = (uint32_t)ib32 << 28;  // local_scale = ib32 (0..7)
        memcpy(sas + 4 * ib32, &val, 4);
    }

    auto values = dequant_single_block_f32(block.data());

    // Check that each sub-block's values reflect the local scale
    for (int ib32 = 0; ib32 < 8; ib32++) {
        float expected_db = d * (0.5f + (float)ib32) * 0.5f;
        float expected_val = expected_db * 4.0f;  // grid byte = 4, sign positive
        // First value in each 32-weight sub-block
        EXPECT_NEAR(values[ib32 * 32], expected_val, 1e-5f)
            << "Sub-block " << ib32 << " local_scale=" << ib32;
    }
}

// ===================================================================
// TEST 2: dequantize_iq3_xxs_blob consistency
// ===================================================================
TEST(IQ3XXS, BlobDequant_SingleRow) {
    // Create a single-row tensor (N=1, K=256)
    std::mt19937 rng(42);
    const int64_t N = 1, K = 256;
    auto blob = make_test_blob(N, K, rng);

    // Use the production dequant function
    auto result = ov::genai::dequantize_iq3_xxs_blob(blob.data(), N, K);
    ASSERT_EQ(result.size(), static_cast<size_t>(N * K));

    // Cross-check against our local reference
    auto ref = dequant_single_block_f32(blob.data());
    ASSERT_EQ(ref.size(), 256u);

    for (size_t i = 0; i < 256; i++) {
        EXPECT_FLOAT_EQ(result[i], ref[i])
            << "Mismatch at index " << i;
    }
}

TEST(IQ3XXS, BlobDequant_MultiRow) {
    // Multi-row tensor: N=4, K=512 (2 blocks per row)
    std::mt19937 rng(123);
    const int64_t N = 4, K = 512;
    auto blob = make_test_blob(N, K, rng);

    auto result = ov::genai::dequantize_iq3_xxs_blob(blob.data(), N, K);
    ASSERT_EQ(result.size(), static_cast<size_t>(N * K));

    // Verify each row/block independently
    constexpr int BLOCK_BYTES = 98;
    const int64_t blocks_per_row = K / 256;

    for (int64_t row = 0; row < N; row++) {
        for (int64_t blk = 0; blk < blocks_per_row; blk++) {
            const uint8_t* block_data = blob.data() + (row * blocks_per_row + blk) * BLOCK_BYTES;
            auto ref = dequant_single_block_f32(block_data);

            for (size_t i = 0; i < 256; i++) {
                size_t flat_idx = row * K + blk * 256 + i;
                EXPECT_FLOAT_EQ(result[flat_idx], ref[i])
                    << "Row=" << row << " Block=" << blk << " idx=" << i;
            }
        }
    }
}

TEST(IQ3XXS, BlobDequant_LargerTensor) {
    // Realistic-ish size: N=32, K=1024
    std::mt19937 rng(999);
    const int64_t N = 32, K = 1024;
    auto blob = make_test_blob(N, K, rng);

    auto result = ov::genai::dequantize_iq3_xxs_blob(blob.data(), N, K);
    ASSERT_EQ(result.size(), static_cast<size_t>(N * K));

    // Spot-check first and last row
    constexpr int BLOCK_BYTES = 98;
    const int64_t blocks_per_row = K / 256;

    // First row, first block
    {
        auto ref = dequant_single_block_f32(blob.data());
        for (size_t i = 0; i < 256; i++) {
            EXPECT_FLOAT_EQ(result[i], ref[i]) << "First row, first block, idx=" << i;
        }
    }

    // Last row, last block
    {
        const uint8_t* last_block = blob.data() + ((N - 1) * blocks_per_row + (blocks_per_row - 1)) * BLOCK_BYTES;
        auto ref = dequant_single_block_f32(last_block);
        for (size_t i = 0; i < 256; i++) {
            size_t flat_idx = (N - 1) * K + (blocks_per_row - 1) * 256 + i;
            EXPECT_FLOAT_EQ(result[flat_idx], ref[i]) << "Last row, last block, idx=" << i;
        }
    }
}

// ===================================================================
// TEST 3: IQ3XXSLinear op shape inference
// ===================================================================
TEST(IQ3XXS, OpShapeInference_Basic) {
    const int64_t M = 4, K = 256, N = 8;
    const int64_t blocks_per_row = K / 256;
    const int64_t total_bytes = N * blocks_per_row * 98;

    // Create activation parameter [M, K]
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});

    // Create compressed weight constant [total_bytes]
    std::vector<uint8_t> dummy_blob(total_bytes, 0);
    auto compressed = std::make_shared<ov::op::v0::Constant>(
        ov::element::u8, ov::Shape{(size_t)total_bytes}, dummy_blob.data());

    // Create IQ3XXSLinear op
    auto iq3_linear = std::make_shared<ov::op::internal::IQ3XXSLinear>(
        activation, compressed, ov::Shape{(size_t)N, (size_t)K}, 256, 98);

    // Check output shape: [M, N]
    auto out_shape = iq3_linear->get_output_shape(0);
    EXPECT_EQ(out_shape.size(), 2u);
    EXPECT_EQ(out_shape[0], (size_t)M);
    EXPECT_EQ(out_shape[1], (size_t)N);

    // Check output type matches activation
    EXPECT_EQ(iq3_linear->get_output_element_type(0), ov::element::f32);
}

TEST(IQ3XXS, OpShapeInference_BatchDims) {
    // Batched activation: [batch, M, K] -> [batch, M, N]
    const size_t batch = 2, M = 8, K = 512, N = 16;
    const int64_t blocks_per_row = K / 256;
    const int64_t total_bytes = N * blocks_per_row * 98;

    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{batch, M, K});
    std::vector<uint8_t> dummy_blob(total_bytes, 0);
    auto compressed = std::make_shared<ov::op::v0::Constant>(
        ov::element::u8, ov::Shape{(size_t)total_bytes}, dummy_blob.data());

    auto iq3_linear = std::make_shared<ov::op::internal::IQ3XXSLinear>(
        activation, compressed, ov::Shape{N, K}, 256, 98);

    auto out_shape = iq3_linear->get_output_shape(0);
    EXPECT_EQ(out_shape.size(), 3u);
    EXPECT_EQ(out_shape[0], batch);
    EXPECT_EQ(out_shape[1], M);
    EXPECT_EQ(out_shape[2], N);
}

// ===================================================================
// TEST 4: IQ3XXSLinear evaluate — correctness vs reference MatMul
// ===================================================================
TEST(IQ3XXS, OpEvaluate_M1) {
    // M=1 (decode path): single activation row
    std::mt19937 rng(7);
    const int64_t M = 1, K = 256, N = 4;

    // Create random compressed weights
    auto blob = make_test_blob(N, K, rng);
    const int64_t total_bytes = N * (K / 256) * 98;

    // Dequantize to f32 for reference
    auto W_f32 = ov::genai::dequantize_iq3_xxs_blob(blob.data(), N, K);

    // Create random activation
    std::vector<float> act(M * K);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : act) v = dist(rng);

    // Reference MatMul: Y = X @ W^T, [1,K] x [K,N] -> [1,N]
    auto ref_Y = reference_matmul(act.data(), W_f32.data(), M, N, K);

    // Create IQ3XXSLinear op and evaluate
    auto activation_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    auto compressed_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::u8, ov::Shape{(size_t)total_bytes}, blob.data());

    auto iq3_linear = std::make_shared<ov::op::internal::IQ3XXSLinear>(
        activation_param, compressed_const, ov::Shape{(size_t)N, (size_t)K}, 256, 98);

    // Prepare tensors for evaluate
    ov::Tensor act_tensor(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    memcpy(act_tensor.data(), act.data(), M * K * sizeof(float));

    ov::Tensor weight_tensor(ov::element::u8, ov::Shape{(size_t)total_bytes});
    memcpy(weight_tensor.data(), blob.data(), total_bytes);

    ov::Tensor out_tensor(ov::element::f32, ov::Shape{(size_t)M, (size_t)N});

    ov::TensorVector inputs = {act_tensor, weight_tensor};
    ov::TensorVector outputs = {out_tensor};

    bool ok = iq3_linear->evaluate(outputs, inputs);
    ASSERT_TRUE(ok);

    const float* out_data = outputs[0].data<float>();
    for (int64_t n = 0; n < N; n++) {
        EXPECT_NEAR(out_data[n], ref_Y[n], std::abs(ref_Y[n]) * 1e-5f + 1e-6f)
            << "Output mismatch at n=" << n;
    }
}

TEST(IQ3XXS, OpEvaluate_M16) {
    // M=16 (prefill path): multiple activation rows
    std::mt19937 rng(77);
    const int64_t M = 16, K = 512, N = 8;
    const int64_t blocks_per_row = K / 256;
    const int64_t total_bytes = N * blocks_per_row * 98;

    auto blob = make_test_blob(N, K, rng);
    auto W_f32 = ov::genai::dequantize_iq3_xxs_blob(blob.data(), N, K);

    std::vector<float> act(M * K);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : act) v = dist(rng);

    auto ref_Y = reference_matmul(act.data(), W_f32.data(), M, N, K);

    // Evaluate via the op
    auto activation_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    auto compressed_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::u8, ov::Shape{(size_t)total_bytes}, blob.data());
    auto iq3_linear = std::make_shared<ov::op::internal::IQ3XXSLinear>(
        activation_param, compressed_const, ov::Shape{(size_t)N, (size_t)K}, 256, 98);

    ov::Tensor act_tensor(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    memcpy(act_tensor.data(), act.data(), M * K * sizeof(float));

    ov::Tensor weight_tensor(ov::element::u8, ov::Shape{(size_t)total_bytes});
    memcpy(weight_tensor.data(), blob.data(), total_bytes);

    ov::Tensor out_tensor(ov::element::f32, ov::Shape{(size_t)M, (size_t)N});

    ov::TensorVector inputs = {act_tensor, weight_tensor};
    ov::TensorVector outputs = {out_tensor};

    bool ok = iq3_linear->evaluate(outputs, inputs);
    ASSERT_TRUE(ok);

    const float* out_data = outputs[0].data<float>();
    for (int64_t i = 0; i < M * N; i++) {
        EXPECT_NEAR(out_data[i], ref_Y[i], std::abs(ref_Y[i]) * 1e-5f + 1e-6f)
            << "Output mismatch at flat index " << i
            << " (m=" << i / N << ", n=" << i % N << ")";
    }
}

// ===================================================================
// TEST 5: Native path == Decompose path
// ===================================================================
TEST(IQ3XXS, NativeEqualsDecompose) {
    // Verify that IQ3XXSLinear::evaluate() gives the same result as
    // decomposing to dequantized constant + MatMul.
    std::mt19937 rng(2026);
    const int64_t M = 4, K = 256, N = 8;
    const int64_t blocks_per_row = K / 256;
    const int64_t total_bytes = N * blocks_per_row * 98;

    auto blob = make_test_blob(N, K, rng);

    // Random activation
    std::vector<float> act(M * K);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : act) v = dist(rng);

    // --- Path A: Native IQ3XXSLinear evaluate ---
    auto activation_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    auto compressed_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::u8, ov::Shape{(size_t)total_bytes}, blob.data());
    auto iq3_linear = std::make_shared<ov::op::internal::IQ3XXSLinear>(
        activation_param, compressed_const, ov::Shape{(size_t)N, (size_t)K}, 256, 98);

    ov::Tensor act_tensor(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    memcpy(act_tensor.data(), act.data(), M * K * sizeof(float));
    ov::Tensor weight_tensor(ov::element::u8, ov::Shape{(size_t)total_bytes});
    memcpy(weight_tensor.data(), blob.data(), total_bytes);
    ov::Tensor native_out(ov::element::f32, ov::Shape{(size_t)M, (size_t)N});

    ov::TensorVector native_inputs = {act_tensor, weight_tensor};
    ov::TensorVector native_outputs = {native_out};
    ASSERT_TRUE(iq3_linear->evaluate(native_outputs, native_inputs));

    // --- Path B: Decompose (dequant + MatMul) ---
    auto W_f32 = ov::genai::dequantize_iq3_xxs_blob(blob.data(), N, K);
    auto decompose_Y = reference_matmul(act.data(), W_f32.data(), M, N, K);

    // --- Compare ---
    const float* native_data = native_outputs[0].data<float>();
    for (int64_t i = 0; i < M * N; i++) {
        EXPECT_FLOAT_EQ(native_data[i], decompose_Y[i])
            << "Native != Decompose at index " << i;
    }
}

// ===================================================================
// TEST 6: Decompose pass actually replaces IQ3XXSLinear nodes
// ===================================================================
TEST(IQ3XXS, DecomposePass_ReplacesNodes) {
    const int64_t M = 2, K = 256, N = 4;
    const int64_t total_bytes = N * (K / 256) * 98;

    // Build a model with IQ3XXSLinear
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    std::vector<uint8_t> dummy_blob(total_bytes, 0);
    auto compressed = std::make_shared<ov::op::v0::Constant>(
        ov::element::u8, ov::Shape{(size_t)total_bytes}, dummy_blob.data());
    auto iq3_linear = std::make_shared<ov::op::internal::IQ3XXSLinear>(
        activation, compressed, ov::Shape{(size_t)N, (size_t)K}, 256, 98);

    auto result = std::make_shared<ov::op::v0::Result>(iq3_linear);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{activation});

    // Verify IQ3XXSLinear exists in graph
    bool found_iq3 = false;
    for (auto& node : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::internal::IQ3XXSLinear>(node)) {
            found_iq3 = true;
            break;
        }
    }
    ASSERT_TRUE(found_iq3) << "IQ3XXSLinear not found in model before decompose";

    // Run decompose
    bool modified = ov::genai::decompose_iq3_xxs_linear(model);
    EXPECT_TRUE(modified);

    // Verify IQ3XXSLinear is gone, replaced by MatMul
    bool found_iq3_after = false;
    bool found_matmul = false;
    for (auto& node : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::internal::IQ3XXSLinear>(node))
            found_iq3_after = true;
        if (std::dynamic_pointer_cast<ov::op::v0::MatMul>(node))
            found_matmul = true;
    }
    EXPECT_FALSE(found_iq3_after) << "IQ3XXSLinear should be gone after decompose";
    EXPECT_TRUE(found_matmul) << "MatMul should exist after decompose";

    // Verify output shape is preserved
    model->validate_nodes_and_infer_types();
    auto out_shape = model->get_results()[0]->get_output_shape(0);
    EXPECT_EQ(out_shape, (ov::Shape{(size_t)M, (size_t)N}));
}

// ===================================================================
// TEST 7: All-zero scale produces all-zero output
// ===================================================================
TEST(IQ3XXS, BlockDequant_ZeroScale) {
    // If d = 0.0, all weights should be 0 regardless of grid/sign
    auto block = make_test_block(0.0f, /*grid_idx=*/42, /*sign_code=*/99);
    auto values = dequant_single_block_f32(block.data());

    for (size_t i = 0; i < values.size(); i++) {
        EXPECT_EQ(values[i], 0.0f) << "Non-zero at index " << i << " with zero scale";
    }
}

// ===================================================================
// TEST 8: Compressed blob size is correct
// ===================================================================
TEST(IQ3XXS, CompressedSize_Validation) {
    // IQ3_XXS: 98 bytes per 256 weights = 3.0625 bits/weight
    const int64_t N = 64, K = 4096;
    const int64_t blocks_per_row = K / 256;
    const int64_t expected_bytes = N * blocks_per_row * 98;

    std::mt19937 rng(12345);
    auto blob = make_test_blob(N, K, rng);
    EXPECT_EQ((int64_t)blob.size(), expected_bytes);

    // Compare against what FP16 would need
    const int64_t fp16_bytes = N * K * 2;
    // Compression ratio should be ~5.22x
    float ratio = (float)fp16_bytes / (float)expected_bytes;
    EXPECT_GT(ratio, 5.0f);
    EXPECT_LT(ratio, 5.5f);
}

// ===================================================================
// TEST 9: Grid table correctness — spot checks from ggml-common.h
// ===================================================================
TEST(IQ3XXS, GridTable_SpotCheck) {
    // iq3xxs_grid[0] = 0x04040404 -> bytes: {4, 4, 4, 4}
    EXPECT_EQ(iq3xxs_grid[0], 0x04040404u);

    // Each byte in the grid represents 2*v+1 where v is a 3-bit trit
    // So valid byte values are: 1, 3, 5, 7, 9, 11, 13 ... or even-valued for hex 04=4
    // Actually the grid values are raw uint8 from the codebook.
    // Verify all grid entries have 4 bytes that are valid (non-zero for most)
    const uint8_t* grid_bytes = reinterpret_cast<const uint8_t*>(iq3xxs_grid);
    for (int i = 0; i < 256; i++) {
        // Grid values should be reasonable (typically small: 0x04, 0x0c, 0x14, etc.)
        const uint8_t* entry = grid_bytes + i * 4;
        for (int j = 0; j < 4; j++) {
            EXPECT_LE(entry[j], 0x3Eu)  // max grid value observed in the table
                << "Grid[" << i << "] byte[" << j << "] = " << (int)entry[j] << " unexpectedly large";
        }
    }
}

// ===================================================================
// TEST 10: Sign table correctness — even parity property
// ===================================================================
TEST(IQ3XXS, SignTable_EvenParity) {
    // ksigns_iq2xs has 128 entries; each entry has even parity (even number of 1-bits)
    for (int i = 0; i < 128; i++) {
        uint8_t val = ksigns_iq2xs[i];
        int popcount = 0;
        for (int b = 0; b < 8; b++) {
            if (val & (1 << b)) popcount++;
        }
        EXPECT_EQ(popcount % 2, 0)
            << "ksigns_iq2xs[" << i << "] = " << (int)val << " has odd parity";
    }
}

// ===================================================================
// TEST 11: Model construction — no FP16 weight materialization in native mode
// ===================================================================
TEST(IQ3XXS, NativeMode_NoFP16WeightConstant) {
    // Build a simple model with IQ3XXSLinear and verify that no large FP16
    // constant exists (the weight stays as compressed u8 blob).
    const int64_t M = 4, K = 512, N = 16;
    const int64_t blocks_per_row = K / 256;
    const int64_t total_bytes = N * blocks_per_row * 98;

    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    std::vector<uint8_t> dummy_blob(total_bytes, 0);
    auto compressed = std::make_shared<ov::op::v0::Constant>(
        ov::element::u8, ov::Shape{(size_t)total_bytes}, dummy_blob.data());
    auto iq3_linear = std::make_shared<ov::op::internal::IQ3XXSLinear>(
        activation, compressed, ov::Shape{(size_t)N, (size_t)K}, 256, 98);
    auto result = std::make_shared<ov::op::v0::Result>(iq3_linear);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{activation});

    // Scan all constants in the model
    for (auto& node : model->get_ordered_ops()) {
        auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
        if (!constant) continue;

        // There should be no f16 or f32 constant with shape [N, K] (the weight shape)
        if (constant->get_element_type() == ov::element::f16 ||
            constant->get_element_type() == ov::element::f32) {
            auto shape = constant->get_shape();
            if (shape.size() == 2 && shape[0] == (size_t)N && shape[1] == (size_t)K) {
                FAIL() << "Found materialized FP weight constant with shape [" << N << ", " << K << "]";
            }
        }
    }

    // The compressed constant should be u8 and much smaller than N*K*2 (fp16 size)
    size_t total_const_bytes = 0;
    for (auto& node : model->get_ordered_ops()) {
        auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
        if (constant) {
            total_const_bytes += constant->get_byte_size();
        }
    }
    const size_t fp16_size = N * K * 2;
    EXPECT_LT(total_const_bytes, fp16_size)
        << "Total constant size (" << total_const_bytes
        << ") should be less than FP16 weight size (" << fp16_size << ")";
}

