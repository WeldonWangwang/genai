// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//
// Reference dequantizer for IQ3_XXS (3.0625 bpw) GGUF tensor type.
// Format: 256-weight super-blocks, each 98 bytes:
//   - 2 bytes: f16 super-block scale 'd'
//   - 32 bytes (QK_K/4): 8 grid indices per 32-weight sub-block (one byte each)
//   - 32 bytes (QK_K/4): scales_and_signs (uint32 per sub-block, packing 4x7-bit sign indices + 4-bit local scale)
// Decode per sub-block (32 weights, 4 groups of 8):
//   local_scale = d * 0.5f * (0.5f + (scales_and_signs >> 28))
//               = d * 0.25f * (1 + 2 * l)  where l = scales_and_signs >> 28
//   For each of the 4 groups (k=0..3):
//     grid_index = qs[2*k+0], qs[2*k+1] -> two 4-weight lookups from iq3xxs_grid
//     signs = ksigns_iq2xs[(scales_and_signs >> 7*k) & 127]
//     weight[j] = local_scale * grid_value[j] * (sign_bit ? -1 : 1)
//

#include <cstdint>
#include <cstring>

#include "gguf_utils/gguf.hpp"

// The IQ3_XXS grid: 256 entries from ggml-common.h
// Each uint32_t encodes 4 values (each byte is 2*v+1 where v is a 3-bit trit)
static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e,
    0x04041404, 0x04041414, 0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c,
    0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14, 0x040c140c, 0x040c142c,
    0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c,
    0x04141c1c, 0x04141c3e, 0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c,
    0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c,
    0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04,
    0x043e0c24, 0x043e0c34, 0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c,
    0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c, 0x0c041c04, 0x0c041c14,
    0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14,
    0x0c14140c, 0x0c141c04, 0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404,
    0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04,
    0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404,
    0x14041414, 0x14041434, 0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c,
    0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c, 0x140c1c04, 0x140c341c,
    0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c,
    0x141c0c04, 0x141c0c24, 0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c,
    0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24, 0x143e040c, 0x143e041c,
    0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414,
    0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c,
    0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14, 0x1c1c0c0c, 0x1c1c1c1c,
    0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404,
    0x24040424, 0x24040c3e, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e,
    0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404, 0x24143404, 0x24143434,
    0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04,
    0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434,
    0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14, 0x2c1c0414, 0x2c1c2c1c,
    0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434,
    0x34043424, 0x340c140c, 0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04,
    0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14, 0x34341c1c, 0x343e041c,
    0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14,
    0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c,
    0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// Sign lookup table from ggml-common.h: 7-bit index -> 8-bit sign pattern (with even parity)
// ksigns_iq2xs[i] gives the 8-bit sign pattern where bit 7 ensures even parity
static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// Mask for applying signs to each of the 8 weights in a group
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// QK_K = 256 weights per super-block
static constexpr int QK_K = 256;
// IQ3_XXS block size: 2 (d) + QK_K/4 (qs) + QK_K/4 (scales_and_signs) = 2 + 64 + 64... 
// Actually: sizeof(block_iq3_xxs) = 2 + 3*(QK_K/8) = 2 + 96 = 98 bytes
// Layout: f16 d (2 bytes) + qs[QK_K/4] (64 bytes) + scales_and_signs[QK_K/8] (32 bytes) = 98 bytes
// Wait, that's 2+64+32=98. Let me recalculate:
// qs: QK_K/4 = 64 bytes (grid indices, 8 per sub-block * 8 sub-blocks)  
// scales_and_signs: 4 bytes * (QK_K/32) = 4*8 = 32 bytes
// Total: 2 + 64 + 32 = 98 ✓

struct block_iq3_xxs {
    uint16_t d;           // f16 super-block scale
    uint8_t qs[QK_K/4];  // grid indices (8 per 32-weight sub-block)
    // followed by: uint8_t scales_and_signs[QK_K/4] but as part of qs in memory layout
    // Actually the layout from llama.cpp: qs has QK_K/4 bytes for grid indices,
    // then scales_and_signs is at qs + QK_K/4, but total .qs size = 3*(QK_K/8) = 96 bytes
};

// The actual block size from the GGUF features table: 256 weights, 98 bytes
static constexpr int IQ3_XXS_BLOCK_BYTES = 98;

static inline float fp16_to_fp32(uint16_t h) {
    // Simple bit manipulation for f16 -> f32
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        // Denormalized
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp += 112;  // 127 - 15
    }

    uint32_t result = sign | (exp << 23) | (mant << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

static inline uint16_t fp32_to_fp16(float f) {
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
        return (uint16_t)(sign | 0x7C00);  // Inf
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

ov::Tensor dequantize_iq3_xxs(gguf_tensor* tensor) {
    OPENVINO_ASSERT(tensor->type == GGUF_TYPE_IQ3_XXS,
                    "[load_gguf] dequantize_iq3_xxs called with wrong tensor type: ", tensor->type);

    const uint64_t num_weights = tensor->num_weights;
    OPENVINO_ASSERT(num_weights % QK_K == 0,
                    "[load_gguf] IQ3_XXS tensor num_weights not divisible by 256: ", num_weights);

    const uint64_t num_blocks = num_weights / QK_K;

    // Verify byte size
    OPENVINO_ASSERT(tensor->bsize == num_blocks * IQ3_XXS_BLOCK_BYTES,
                    "[load_gguf] IQ3_XXS tensor size mismatch: expected ",
                    num_blocks * IQ3_XXS_BLOCK_BYTES, " got ", tensor->bsize);

    // Get output shape (same as original tensor shape)
    ov::Shape shape;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        shape.push_back(tensor->dim[i]);
    }

    ov::Tensor result(ov::element::f16, shape);
    uint16_t* out = reinterpret_cast<uint16_t*>(result.data());

    const uint8_t* raw = tensor->weights_data;

    for (uint64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint8_t* block_data = raw + block_idx * IQ3_XXS_BLOCK_BYTES;

        // Read f16 super-block scale
        uint16_t d_fp16;
        memcpy(&d_fp16, block_data, 2);
        const float d = fp16_to_fp32(d_fp16);

        const uint8_t* qs = block_data + 2;                    // grid indices: QK_K/4 = 64 bytes
        const uint8_t* scales_and_signs = qs + QK_K / 4;       // 32 bytes (8 uint32_t)

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            // Read the 4-byte scales_and_signs for this sub-block
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));

            // Local scale: d * 0.5f * (0.5f + (aux32 >> 28))
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

            for (int l = 0; l < 4; ++l) {
                // Get sign pattern for this group of 8 weights
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];

                // Two grid lookups per group of 8: first 4 from grid1, next 4 from grid2
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 0]]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&iq3xxs_grid[qs[2 * l + 1]]);

                for (int j = 0; j < 4; ++j) {
                    float val = db * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                    *out++ = fp32_to_fp16(val);
                }
                for (int j = 0; j < 4; ++j) {
                    float val = db * grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                    *out++ = fp32_to_fp16(val);
                }
            }
            qs += 8;  // 8 grid index bytes per 32-weight sub-block
        }
    }

    return result;
}
