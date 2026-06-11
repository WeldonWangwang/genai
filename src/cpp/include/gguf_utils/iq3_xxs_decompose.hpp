// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "openvino/core/model.hpp"

namespace ov {
namespace genai {

/// Dequantize a raw IQ3_XXS weight blob into a row-major f32 [N, K] buffer.
/// Used both by the decomposition pass and by the embedding (Gather) path.
std::vector<float> dequantize_iq3_xxs_blob(const uint8_t* data, int64_t N, int64_t K);

/// Decompose all IQ3XXSLinear ops in the model into dequantized MatMul subgraphs.
/// Returns true if any nodes were replaced.
bool decompose_iq3_xxs_linear(const std::shared_ptr<ov::Model>& model);

}  // namespace genai
}  // namespace ov
