#!/usr/bin/env sh
# Download the trained ONNX models into ./models (Linux / macOS / Git Bash).
#   MODELS_URL=<release asset url> sh scripts/download_models.sh
set -e
MODELS_URL="${MODELS_URL:-https://github.com/Ali-prog-spec/Restoration-Sketch-studio/releases/download/v1.0/models.zip}"
cd "$(dirname "$0")/.."
mkdir -p models
echo "Downloading $MODELS_URL"
curl -L --fail -o models/models.zip "$MODELS_URL"
if [ -f models/models.sha256 ]; then
  (cd models && sha256sum -c models.sha256)
fi
unzip -o models/models.zip -d models
rm models/models.zip
ls -l models
