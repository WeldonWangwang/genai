// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/iq3_xxs_linear.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "openvino/core/type/float16.hpp"

// Lookup tables from gguf_iq3_xxs.cpp
extern const uint32_t iq3xxs_grid[256];
extern const uint8_t ksigns_iq2xs[128];
extern const uint8_t kmask_iq2xs[8];

namespace ov {
namespace op {
namespace internal {

IQ3XXSLinear::IQ3XXSLinear(const Output<Node>& activation,
                           const Output<Node>& compressed_weights,
                           const ov::Shape& weight_shape,
                           int64_t block_size,
                           int64_t bytes_per_block)
    : Op({activation, compressed_weights}),
      m_weight_shape(weight_shape),
      m_block_size(block_size),
      m_bytes_per_block(bytes_per_block) {
    constructor_validate_and_infer_types();
}

bool IQ3XXSLinear::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("weight_shape", m_weight_shape);
    visitor.on_attribute("block_size", m_block_size);
    visitor.on_attribute("bytes_per_block", m_bytes_per_block);
    return true;
}

void IQ3XXSLinear::validate_and_infer_types() {
    // Input 0: activation [M, K] or [batch..., M, K]
    const auto& activation_type = get_input_element_type(0);
    const auto& activation_pshape = get_input_partial_shape(0);

    // Input 1: compressed weights blob [total_bytes] - must be u8
    const auto& weights_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          weights_type == element::u8,
                          "Compressed weights must be u8 type, got: ",
                          weights_type);

    // Validate weight_shape: [N, K]
    NODE_VALIDATION_CHECK(this,
                          m_weight_shape.size() == 2,
                          "weight_shape must be 2D [N, K], got rank: ",
                          m_weight_shape.size());

    const int64_t N = static_cast<int64_t>(m_weight_shape[0]);
    const int64_t K = static_cast<int64_t>(m_weight_shape[1]);

    // Validate K is compatible with block_size
    NODE_VALIDATION_CHECK(this,
                          K % m_block_size == 0,
                          "K (", K, ") must be divisible by block_size (", m_block_size, ")");

    // Validate compressed data size
    const int64_t blocks_per_row = K / m_block_size;
    const int64_t expected_bytes = N * blocks_per_row * m_bytes_per_block;
    if (get_input_partial_shape(1).is_static()) {
        const auto& weights_shape = get_input_partial_shape(1).to_shape();
        NODE_VALIDATION_CHECK(this,
                              weights_shape.size() == 1,
                              "Compressed weights must be 1D blob");
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(weights_shape[0]) == expected_bytes,
                              "Compressed weights size mismatch: expected ",
                              expected_bytes, " bytes, got ", weights_shape[0]);
    }

    // Output shape: activation leading dims + N
    // activation: [..., M, K] -> output: [..., M, N]
    if (activation_pshape.rank().is_dynamic()) {
        set_output_type(0, activation_type, ov::PartialShape::dynamic());
    } else {
        auto output_pshape = activation_pshape;
        // Last dim of activation (K) replaced by N (from weight_shape[0])
        output_pshape[output_pshape.rank().get_length() - 1] = N;
        set_output_type(0, activation_type, output_pshape);
    }
}

std::shared_ptr<Node> IQ3XXSLinear::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<IQ3XXSLinear>(new_args[0],
                                          new_args[1],
                                          m_weight_shape,
                                          m_block_size,
                                          m_bytes_per_block);
}

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; f = sign | ((exp + 112) << 23) | (mant << 13); }
    } else if (exp == 31) { f = sign | 0x7F800000 | (mant << 13); }
    else { f = sign | ((exp + 112) << 23) | (mant << 13); }
    float result; memcpy(&result, &f, 4); return result;
}

// Decode a single IQ3_XXS weight row (K values) from the compressed blob into
// a contiguous f32 buffer. `w_row` points at the start of this row's bytes.
// This isolates the (relatively expensive) codebook lookup so it can be done
// once per output channel and reused across all activation rows.
static inline void decode_iq3_xxs_row(const uint8_t* w_row, float* out, size_t blocks_per_row) {
    constexpr size_t QK_K = 256;
    for (size_t blk = 0; blk < blocks_per_row; blk++) {
        const uint8_t* block_data = w_row + blk * 98;

        uint16_t d_fp16;
        memcpy(&d_fp16, block_data, 2);
        const float d = fp16_to_f32(d_fp16);

        const uint8_t* qs = block_data + 2;
        const uint8_t* scales_and_signs = qs + QK_K / 4;  // +64

        float* o = out + blk * QK_K;
        size_t oi = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 0]]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 1]]);

                for (int j = 0; j < 4; ++j) {
                    o[oi++] = db * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                }
                for (int j = 0; j < 4; ++j) {
                    o[oi++] = db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                }
            }
            qs += 8;
        }
    }
}

bool IQ3XXSLinear::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // activation: [..., M, K], compressed_weights: [total_bytes]
    const auto& act_shape = inputs[0].get_shape();
    const size_t rank = act_shape.size();
    const size_t M = (rank >= 2) ? act_shape[rank - 2] : 1;
    const size_t K = act_shape[rank - 1];
    const size_t N = m_weight_shape[0];

    // Compute batch dimensions
    size_t batch = 1;
    for (size_t i = 0; i + 2 < rank; i++) batch *= act_shape[i];

    // Setup output shape
    ov::Shape out_shape = act_shape;
    out_shape[rank - 1] = N;
    outputs[0].set_shape(out_shape);

    const uint8_t* compressed = inputs[1].data<uint8_t>();

    constexpr size_t QK_K = 256;
    const size_t blocks_per_row = K / QK_K;
    const size_t bytes_per_row = blocks_per_row * 98;
    const size_t rows = batch * M;  // total activation rows

    // Provide an f32 view of the activations. The CPU plugin may execute this op
    // with f16 activations (e.g. ov::hint::inference_precision=f16, which is
    // needed to fit large models in limited RAM). Decode/compute is done in f32
    // for accuracy; only the (small) activation and output tensors are converted.
    // The large compressed weights are never upcast.
    const auto act_type = inputs[0].get_element_type();
    std::vector<float> act_f32_storage;
    const float* act_data = nullptr;
    if (act_type == ov::element::f32) {
        act_data = inputs[0].data<float>();
    } else if (act_type == ov::element::f16) {
        const ov::float16* a16 = inputs[0].data<ov::float16>();
        act_f32_storage.resize(rows * K);
        for (size_t i = 0; i < rows * K; ++i) {
            act_f32_storage[i] = static_cast<float>(a16[i]);
        }
        act_data = act_f32_storage.data();
    } else {
        return false;
    }

    const auto out_type = outputs[0].get_element_type();
    float* out_f32 = (out_type == ov::element::f32) ? outputs[0].data<float>() : nullptr;
    ov::float16* out_f16 = (out_type == ov::element::f16) ? outputs[0].data<ov::float16>() : nullptr;
    if (!out_f32 && !out_f16) {
        return false;
    }

    // Worker: process output channels [n_begin, n_end). For each channel, decode
    // the weight row ONCE into a local buffer, then accumulate the dot product
    // against every activation row. This amortizes the codebook decode over all
    // M rows (critical for prefill) instead of re-decoding per (row, channel).
    auto worker = [&](size_t n_begin, size_t n_end) {
        std::vector<float> wbuf(K);
        for (size_t n = n_begin; n < n_end; n++) {
            decode_iq3_xxs_row(compressed + n * bytes_per_row, wbuf.data(), blocks_per_row);
            const float* w = wbuf.data();
            for (size_t r = 0; r < rows; r++) {
                const float* a = act_data + r * K;
                float acc = 0.0f;
                // Contiguous f32 dot product; auto-vectorizes under /O2.
                for (size_t k = 0; k < K; k++) {
                    acc += a[k] * w[k];
                }
                if (out_f32) {
                    out_f32[r * N + n] = acc;
                } else {
                    out_f16[r * N + n] = ov::float16(acc);
                }
            }
        }
    };

    // Parallelize over output channels with std::thread. Use threads only when
    // the work is large enough to amortize spawn overhead; otherwise run serially.
    const size_t total_work = N * rows * K;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    size_t nthreads = static_cast<size_t>(hw);
    constexpr size_t PARALLEL_THRESHOLD = 1ull << 20;  // ~1M MACs
    if (total_work < PARALLEL_THRESHOLD || N < nthreads || nthreads <= 1) {
        worker(0, N);
        return true;
    }

    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    const size_t chunk = (N + nthreads - 1) / nthreads;
    for (size_t t = 1; t < nthreads; t++) {
        const size_t b = t * chunk;
        if (b >= N) break;
        const size_t e = std::min(N, b + chunk);
        pool.emplace_back(worker, b, e);
    }
    // Current thread handles the first chunk.
    worker(0, std::min(N, chunk));
    for (auto& th : pool) th.join();
    return true;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
