// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <string>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/runtime/properties.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "CPU";  // GPU can be used as well
    if (argc > 3)
        device = argv[3];

    // Optional: reduce peak memory / weight precision during compile by hinting
    // f16 inference precision. Useful for large GGUF models on memory-constrained
    // hosts. Enable with OV_GENAI_INFERENCE_PRECISION=f16 (or bf16/f32).
    ov::AnyMap properties;
    if (const char* prec = std::getenv("OV_GENAI_INFERENCE_PRECISION")) {
        std::string p(prec);
        if (p == "f16")
            properties[ov::hint::inference_precision.name()] = ov::element::f16;
        else if (p == "bf16")
            properties[ov::hint::inference_precision.name()] = ov::element::bf16;
        else if (p == "f32")
            properties[ov::hint::inference_precision.name()] = ov::element::f32;
    }

    ov::genai::LLMPipeline pipe(models_path, device, properties);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
