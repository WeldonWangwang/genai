// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//
// Reference dequantizer for IQ4_XS (4.25 bpw) GGUF tensor type.
// Format: 256-weight super-blocks, each 136 bytes:
//   - 2 bytes: f16 super-block scale 'd'
//   - 2 bytes: uint16_t scales_h (high 2 bits of 8 sub-block scales)
//   - 4 bytes: uint8_t scales_l[4] (low 4 bits of 8 sub-block scales)
//   - 128 bytes: uint8_t qs[128] (4-bit quantized values, 2 per byte)
//

#include <cstdint>
#include <cstring>

#include "gguf_utils/gguf.hpp"

// Non-linear quantization values for IQ4_NL / IQ4_XS
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// QK_K = 256 weights per super-block
static constexpr int QK_K = 256;
// IQ4_XS block size: 2 + 2 + 4 + 128 = 136 bytes
static constexpr int IQ4_XS_BLOCK_BYTES = 136;

static inline float fp16_to_fp32_iq4xs(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign; float f; memcpy(&f, &r, 4); return f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) { exp = 255; }
    else { exp += 112; }
    uint32_t result = sign | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &result, 4); return f;
}

static inline uint16_t fp32_to_fp16_iq4xs(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(1 - exp);
        mant >>= shift;
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) { return (uint16_t)(sign | 0x7C00); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

ov::Tensor dequantize_iq4_xs(gguf_tensor* tensor) {
    OPENVINO_ASSERT(tensor->type == GGUF_TYPE_IQ4_XS,
                    "[load_gguf] dequantize_iq4_xs called with wrong tensor type: ", tensor->type);

    const uint64_t num_weights = tensor->num_weights;
    OPENVINO_ASSERT(num_weights % QK_K == 0,
                    "[load_gguf] IQ4_XS tensor num_weights not divisible by 256: ", num_weights);

    const uint64_t num_blocks = num_weights / QK_K;
    OPENVINO_ASSERT(tensor->bsize == num_blocks * IQ4_XS_BLOCK_BYTES,
                    "[load_gguf] IQ4_XS tensor size mismatch: expected ",
                    num_blocks * IQ4_XS_BLOCK_BYTES, " got ", tensor->bsize);

    ov::Shape shape;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        shape.push_back(tensor->dim[i]);
    }

    ov::Tensor result(ov::element::f16, shape);
    uint16_t* out = reinterpret_cast<uint16_t*>(result.data());
    const uint8_t* raw = tensor->weights_data;

    for (uint64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint8_t* block_data = raw + block_idx * IQ4_XS_BLOCK_BYTES;

        // Read f16 super-block scale
        uint16_t d_fp16;
        memcpy(&d_fp16, block_data, 2);
        const float d = fp16_to_fp32_iq4xs(d_fp16);

        // Read scales_h (2 bytes, offset 2)
        uint16_t scales_h;
        memcpy(&scales_h, block_data + 2, 2);

        // scales_l at offset 4, length 4 bytes
        const uint8_t* scales_l = block_data + 4;

        // qs at offset 8, length 128 bytes
        const uint8_t* qs = block_data + 8;

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            // Reconstruct 6-bit sub-block scale
            const int ls_low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
            const int ls_high = ((scales_h >> (2 * ib)) & 3) << 4;
            const int ls = ls_low | ls_high;
            const float dl = d * (ls - 32);

            // 16 bytes encode 32 weights (4 bits each)
            const uint8_t* qb = qs + ib * 16;
            for (int j = 0; j < 16; ++j) {
                out[j]      = fp32_to_fp16_iq4xs(dl * kvalues_iq4nl[qb[j] & 0xf]);
                out[j + 16] = fp32_to_fp16_iq4xs(dl * kvalues_iq4nl[qb[j] >> 4]);
            }
            out += 32;
        }
    }
    return result;
}
