# Download the trained ONNX models into .\models (Windows PowerShell).
#   $env:MODELS_URL = "<release asset url>"; .\scripts\download_models.ps1
$ErrorActionPreference = "Stop"
$url = $env:MODELS_URL
if (-not $url) { $url = "https://github.com/Ali-prog-spec/Restoration-Sketch-studio/releases/download/v1.0/models.zip" }
$root = Split-Path -Parent $PSScriptRoot
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force $models | Out-Null
$zip = Join-Path $models "models.zip"
Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zip
Expand-Archive -Path $zip -DestinationPath $models -Force
Remove-Item $zip
Get-ChildItem $models
