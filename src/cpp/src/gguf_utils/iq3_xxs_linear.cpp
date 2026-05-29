// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/iq3_xxs_linear.hpp"

#include <cstring>
#include <vector>

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

    const float* act_data = inputs[0].data<float>();
    const uint8_t* compressed = inputs[1].data<uint8_t>();
    float* out_data = outputs[0].data<float>();

    constexpr size_t QK_K = 256;
    const size_t blocks_per_row = K / QK_K;
    const size_t bytes_per_row = blocks_per_row * 98;

    // For each batch×M row, compute dot product with each of N weight rows
    for (size_t b = 0; b < batch * M; b++) {
        const float* a_row = act_data + b * K;
        float* o_row = out_data + b * N;

        for (size_t n = 0; n < N; n++) {
            float acc = 0.0f;
            const uint8_t* w_row = compressed + n * bytes_per_row;

            for (size_t blk = 0; blk < blocks_per_row; blk++) {
                const uint8_t* block_data = w_row + blk * 98;

                // Decode super-block scale
                uint16_t d_fp16;
                memcpy(&d_fp16, block_data, 2);
                const float d = fp16_to_f32(d_fp16);

                const uint8_t* qs = block_data + 2;
                const uint8_t* scales_and_signs = qs + QK_K / 4;  // +64

                size_t w_offset = blk * QK_K;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    uint32_t aux32;
                    memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
                    const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

                    for (int l = 0; l < 4; ++l) {
                        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                        const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 0]]);
                        const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 1]]);

                        for (int j = 0; j < 4; ++j) {
                            float w = db * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                            acc += a_row[w_offset++] * w;
                        }
                        for (int j = 0; j < 4; ++j) {
                            float w = db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                            acc += a_row[w_offset++] * w;
                        }
                    }
                    qs += 8;
                }
            }
            o_row[n] = acc;
        }
    }
    return true;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
