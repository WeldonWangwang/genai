param([string]$OpenvPath = "C:\Users\gta\www\openvino\build\install")

$env:INTEL_OPENVINO_DIR = $OpenvPath
$env:OpenVINO_DIR = $OpenvPath
$env:Path = "$OpenvPath\runtime\bin\intel64\Release;$OpenvPath\bin\intel64\Release;$env:Path"

Write-Host "OpenVINO environment ready"
Write-Host "INTEL_OPENVINO_DIR: $env:INTEL_OPENVINO_DIR"
