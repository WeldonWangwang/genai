// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

/// \brief Plugin-prep transformation for the CompressedConstant path.
///
/// Rewrites every subgraph of the form
///
///     MatMul(input, CompressedConstant<IQ3_XXS>, transpose_b=true)
///
/// into the legacy native-kernel form
///
///     IQ3XXSLinear(input_f32, raw_u8_constant, weight_shape=logical, 256, 98)
///
/// This is the bridge between the new unified CompressedConstant frontend (which
/// all standard OpenVINO passes can see and reason about as a normal MatMul on a
/// Constant) and the existing CPU plugin native kernel (IQ3XXSLinear Node), which
/// expects a raw u8 [N_bytes] Constant on its weight port.
///
/// Run this pass once, after the GenAI model is constructed and *before* calling
/// `ov::Core::compile_model`. It is a no-op on graphs that do not contain
/// `CompressedConstant` nodes, so it can be unconditionally invoked.
///
/// Returns true if any node was replaced.
///
/// \note Only IQ3_XXS is wired up today. Other QuantType values (IQ2_S, IQ4_XS,
/// Q3_K, ...) are silently left in the graph as CompressedConstant; add the
/// corresponding rewrite branch here when the matching plugin kernel exists.
OPENVINO_GENAI_EXPORTS bool rewrite_compressed_matmul_to_iq3_xxs_linear(const std::shared_ptr<ov::Model>& model);

}  // namespace genai
}  // namespace ov
