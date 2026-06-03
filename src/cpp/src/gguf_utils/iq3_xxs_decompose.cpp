// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Pre-inference decomposition pass: replaces IQ3XXSLinear ops with
// dequantized_weight_constant + MatMul for plugins that don't natively support it.

#include "gguf_utils/iq3_xxs_decompose.hpp"

#include <cstring>
#include <vector>

#include "openvino/op/iq3_xxs_linear.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/core/graph_util.hpp"

// These are defined in gguf_iq3_xxs.cpp - declared extern here
extern const uint32_t iq3xxs_grid[256];
extern const uint8_t ksigns_iq2xs[128];
extern const uint8_t kmask_iq2xs[8];

namespace ov {
namespace genai {

static float fp16_to_fp32_local(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
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

std::vector<float> dequantize_iq3_xxs_blob(const uint8_t* data, int64_t N, int64_t K) {
    constexpr int QK_K = 256;
    constexpr int BLOCK_BYTES = 98;
    const int64_t num_blocks_per_row = K / QK_K;

    std::vector<float> result(N * K);

    for (int64_t row = 0; row < N; row++) {
        for (int64_t blk = 0; blk < num_blocks_per_row; blk++) {
            const uint8_t* block_data = data + (row * num_blocks_per_row + blk) * BLOCK_BYTES;
            float* out = result.data() + row * K + blk * QK_K;

            uint16_t d_fp16;
            memcpy(&d_fp16, block_data, 2);
            const float d = fp16_to_fp32_local(d_fp16);

            const uint8_t* qs = block_data + 2;
            const uint8_t* scales_and_signs = qs + QK_K / 4;

            int out_idx = 0;
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                uint32_t aux32;
                memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));

                const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                    const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 0]]);
                    const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 1]]);

                    for (int j = 0; j < 4; ++j) {
                        float val = db * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                        out[out_idx++] = val;
                    }
                    for (int j = 0; j < 4; ++j) {
                        float val = db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                        out[out_idx++] = val;
                    }
                }
                qs += 8;
            }
        }
    }
    return result;
}

bool decompose_iq3_xxs_linear(const std::shared_ptr<ov::Model>& model) {
    bool modified = false;

    for (auto& node : model->get_ordered_ops()) {
        auto iq3_node = std::dynamic_pointer_cast<ov::op::internal::IQ3XXSLinear>(node);
        if (!iq3_node)
            continue;

        auto weight_shape = iq3_node->get_weight_shape();
        int64_t N = static_cast<int64_t>(weight_shape[0]);
        int64_t K = static_cast<int64_t>(weight_shape[1]);

        auto compressed_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            iq3_node->input_value(1).get_node_shared_ptr());
        if (!compressed_const)
            continue;

        const uint8_t* compressed_data = compressed_const->get_data_ptr<uint8_t>();

        auto decompressed = dequantize_iq3_xxs_blob(compressed_data, N, K);

        auto weight_const = std::make_shared<ov::op::v0::Constant>(
            ov::element::f32,
            ov::Shape{static_cast<size_t>(N), static_cast<size_t>(K)},
            decompressed.data());
        weight_const->set_friendly_name(iq3_node->get_friendly_name() + ".weight_f32");

        auto matmul = std::make_shared<ov::op::v0::MatMul>(
            iq3_node->input_value(0), weight_const, false, true);
        matmul->set_friendly_name(iq3_node->get_friendly_name());

        ov::replace_node(iq3_node, matmul);
        modified = true;
    }

    return modified;
}

}  // namespace genai
}  // namespace ov
