// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//
// Reference dequantizer for Q3_K (3.4375 bpw) GGUF tensor type.
// Block = 110 bytes for 256 weights:
//   hmask[32], qs[64], scales[12], d (f16)
//

#include <cstdint>
#include <cstring>

#include "gguf_utils/gguf.hpp"

static constexpr int QK_K = 256;
static constexpr int Q3_K_BLOCK_BYTES = 110;

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

ov::Tensor dequantize_q3_k(gguf_tensor* tensor) {
    OPENVINO_ASSERT(tensor->type == GGUF_TYPE_Q3_K,
                    "[load_gguf] dequantize_q3_k wrong tensor type: ", tensor->type);
    const uint64_t num_weights = tensor->num_weights;
    OPENVINO_ASSERT(num_weights % QK_K == 0,
                    "[load_gguf] Q3_K num_weights not divisible by 256: ", num_weights);
    const uint64_t num_blocks = num_weights / QK_K;

    ov::Shape shape;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        shape.push_back(tensor->dim[i]);
    }
    ov::Tensor result(ov::element::f16, shape);
    uint16_t* out = reinterpret_cast<uint16_t*>(result.data());
    const uint8_t* raw = tensor->weights_data;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    for (uint64_t b = 0; b < num_blocks; ++b) {
        const uint8_t* bd = raw + b * Q3_K_BLOCK_BYTES;
        const uint8_t* hmask = bd;              // 32 bytes
        const uint8_t* qbase = bd + 32;         // 64 bytes
        const uint8_t* scbytes = bd + 96;       // 12 bytes
        uint16_t d_fp16; memcpy(&d_fp16, bd + 108, 2);
        const float d_all = fp16_to_fp32_k(d_fp16);

        memcpy(aux, scbytes, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        const uint8_t* q = qbase;
        uint8_t m = 1;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int v = ((int)((q[l] >> shift) & 3)) - ((hmask[l] & m) ? 0 : 4);
                    *out++ = fp32_to_fp16_k(dl * v);
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int v = ((int)((q[l + 16] >> shift) & 3)) - ((hmask[l + 16] & m) ? 0 : 4);
                    *out++ = fp32_to_fp16_k(dl * v);
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
    return result;
}
