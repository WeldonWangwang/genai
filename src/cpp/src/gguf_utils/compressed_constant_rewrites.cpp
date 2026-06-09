// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/compressed_constant_rewrites.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/iq3_xxs_linear.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/util/compressed_constant.hpp"

namespace ov {
namespace genai {

namespace {

using CC = ov::op::util::CompressedConstant;

// Promote `input` to f32 if it is not already, returning the (possibly newly
// inserted) Output that an IQ3XXSLinear can consume. Mirrors the behaviour of
// the original direct-construction path in building_blocks.cpp.
ov::Output<ov::Node> ensure_f32(const ov::Output<ov::Node>& input) {
    if (input.get_element_type() == ov::element::f32) {
        return input;
    }
    return std::make_shared<ov::op::v0::Convert>(input, ov::element::f32)->output(0);
}

// Build a fresh Constant that owns a copy of the CompressedConstant's raw u8 blob,
// preserving its byte size. IQ3XXSLinear's plugin kernel expects exactly this shape
// (u8 [N_bytes]).
std::shared_ptr<ov::op::v0::Constant> clone_raw_blob_as_u8_constant(const std::shared_ptr<CC>& cc) {
    const auto bytes = cc->get_compressed_byte_size();
    return std::make_shared<ov::op::v0::Constant>(ov::element::u8,
                                                   ov::Shape{bytes},
                                                   cc->get_compressed_data_ptr());
}

// Rewrite one MatMul that consumes a CompressedConstant<IQ3_XXS> on input(1).
// `cc` is the matched constant, `matmul` is its consumer. Returns true on success.
bool rewrite_one_iq3_xxs_matmul(const std::shared_ptr<ov::op::v0::MatMul>& matmul,
                                const std::shared_ptr<CC>& cc) {
    // Only IQ3_XXS is wired in this pass — silently skip anything else so that
    // future quantization formats can be added incrementally.
    if (cc->get_quant_type() != CC::QuantType::IQ3_XXS) {
        return false;
    }

    // IQ3XXSLinear's kernel always computes Y = X @ W^T (transpose_b semantics).
    // If the consumer MatMul disagrees, do not rewrite — leave it for a generic
    // dequant fallback to handle correctly.
    if (matmul->get_transpose_a() || !matmul->get_transpose_b()) {
        return false;
    }

    auto input_f32 = ensure_f32(matmul->input_value(0));
    auto raw_u8 = clone_raw_blob_as_u8_constant(cc);
    raw_u8->set_friendly_name(cc->get_friendly_name() + ".u8_raw");

    auto iq3_linear = std::make_shared<ov::op::internal::IQ3XXSLinear>(
        input_f32,
        raw_u8,
        cc->get_logical_shape(),
        /*block_size=*/256,
        /*bytes_per_block=*/98);
    iq3_linear->set_friendly_name(matmul->get_friendly_name());

    ov::replace_node(matmul, iq3_linear);
    return true;
}

}  // namespace

bool rewrite_compressed_matmul_to_iq3_xxs_linear(const std::shared_ptr<ov::Model>& model) {
    bool modified = false;

    // We snapshot the node list because we rewrite the graph during iteration.
    for (auto& node : model->get_ordered_ops()) {
        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        if (!matmul) {
            continue;
        }
        auto weight_node = matmul->input_value(1).get_node_shared_ptr();
        auto cc = std::dynamic_pointer_cast<CC>(weight_node);
        if (!cc) {
            continue;
        }
        if (rewrite_one_iq3_xxs_matmul(matmul, cc)) {
            modified = true;
        }
    }

    return modified;
}

}  // namespace genai
}  // namespace ov
