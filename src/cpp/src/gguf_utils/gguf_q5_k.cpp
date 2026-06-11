// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//
// Reference dequantizer for Q5_K (5.5 bpw) GGUF tensor type.
// Block = 176 bytes for 256 weights:
//   d (f16), dmin (f16), scales[12], qh[32], qs[128]
//

#include <cstdint>
#include <cstring>

#include "gguf_utils/gguf.hpp"

static constexpr int QK_K = 256;
static constexpr int Q5_K_BLOCK_BYTES = 176;

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static inline float fp16_to_fp32_k(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t result;
    if (exp == 0) {
        if (mant == 0) {
            result = sign;  // +/- zero
        } else {
            // subnormal: normalize using a signed shift count. The previous
            // unsigned `exp--` underflowed to 0xFFFFFFFF for tiny f16 scales,
            // turning them into Inf/NaN and corrupting whole quant blocks.
            int e = -1;
            do { mant <<= 1; ++e; } while (!(mant & 0x400));
            mant &= 0x3FF;
            result = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        result = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        result = sign | ((exp + 112) << 23) | (mant << 13);  // normal
    }
    float f; memcpy(&f, &result, 4); return f;
}

static inline uint16_t fp32_to_fp16_k(float f) {
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

ov::Tensor dequantize_q5_k(gguf_tensor* tensor) {
    OPENVINO_ASSERT(tensor->type == GGUF_TYPE_Q5_K,
                    "[load_gguf] dequantize_q5_k wrong tensor type: ", tensor->type);
    const uint64_t num_weights = tensor->num_weights;
    OPENVINO_ASSERT(num_weights % QK_K == 0,
                    "[load_gguf] Q5_K num_weights not divisible by 256: ", num_weights);
    const uint64_t num_blocks = num_weights / QK_K;

    ov::Shape shape;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        shape.push_back(tensor->dim[i]);
    }
    ov::Tensor result(ov::element::f16, shape);
    uint16_t* out = reinterpret_cast<uint16_t*>(result.data());
    const uint8_t* raw = tensor->weights_data;

    for (uint64_t b = 0; b < num_blocks; ++b) {
        const uint8_t* bd = raw + b * Q5_K_BLOCK_BYTES;
        uint16_t d_fp16, dmin_fp16;
        memcpy(&d_fp16, bd, 2);
        memcpy(&dmin_fp16, bd + 2, 2);
        const float d = fp16_to_fp32_k(d_fp16);
        const float dmin = fp16_to_fp32_k(dmin_fp16);
        const uint8_t* scbytes = bd + 4;    // 12 bytes
        const uint8_t* qh = bd + 16;        // 32 bytes
        const uint8_t* ql = bd + 48;        // 128 bytes

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, mm;
            get_scale_min_k4(is + 0, scbytes, &sc, &mm);
            const float d1 = d * sc, m1 = dmin * mm;
            get_scale_min_k4(is + 1, scbytes, &sc, &mm);
            const float d2 = d * sc, m2 = dmin * mm;
            for (int l = 0; l < 32; ++l) {
                float v = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                out[l] = fp32_to_fp16_k(v);
            }
            for (int l = 0; l < 32; ++l) {
                float v = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                out[l + 32] = fp32_to_fp16_k(v);
            }
            out += 64;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return result;
}
