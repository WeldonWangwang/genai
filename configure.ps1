#!/usr/bin/env powershell
param(
    [string]$BuildDir = "C:\Users\gta\www\openvino\build",
    [string]$SourceDir = "C:\Users\gta\www\openvino.genai",
    [string]$BinaryDir = "$SourceDir\build"
)

Write-Host "Configuring openvino.genai with OpenVINO from: $BuildDir"

# Check if OpenVINO cmake exists
$openvinoConfig = "$BuildDir\cmake\OpenVINOConfig.cmake"
$openvinoDeveloperConfig = "$BuildDir\cmake\OpenVINODeveloperPackageConfig.cmake"

if (Test-Path $openvinoConfig) {
    Write-Host "✓ Found OpenVINOConfig.cmake"
} else {
    Write-Host "⚠ OpenVINOConfig.cmake not found at $openvinoConfig"
}

if (Test-Path $openvinoDeveloperConfig) {
    Write-Host "✓ Found OpenVINODeveloperPackageConfig.cmake"
} else {
    Write-Host "⚠ OpenVINODeveloperPackageConfig.cmake not found"
}

# Set environment
$env:OpenVINO_DIR = $BuildDir
$env:CMAKE_PREFIX_PATH = "$BuildDir\cmake;$env:CMAKE_PREFIX_PATH"

Set-Location $SourceDir

# Run CMake
Write-Host "Running CMake..."
cmake -DCMAKE_BUILD_TYPE=Release `
    -DOpenVINO_DIR="$BuildDir" `
    -DCMAKE_PREFIX_PATH="$BuildDir\cmake" `
    -S ./ `
    -B ./build/ `
    @args

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ CMake configuration successful"
} else {
    Write-Host "✗ CMake configuration failed with exit code $LASTEXITCODE"
}
