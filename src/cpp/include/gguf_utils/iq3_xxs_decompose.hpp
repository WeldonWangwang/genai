// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "openvino/core/model.hpp"

namespace ov {
namespace genai {

/// Decompose all IQ3XXSLinear ops in the model into dequantized MatMul subgraphs.
/// Returns true if any nodes were replaced.
bool decompose_iq3_xxs_linear(const std::shared_ptr<ov::Model>& model);

}  // namespace genai
}  // namespace ov
