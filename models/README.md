# Trained models

The application needs these ONNX files in this folder (they are **not** committed to Git):

| File | Task | Inputs → outputs |
|---|---|---|
| `task1_universal_dae.onnx` | 1 | `input` float32 [N,3,128,128] in [0,1] → `restored` |
| `task2_classifier.onnx` | 2 | `input` → `logits` [N,4] (clean, salt_pepper, blur, occlusion) |
| `task2_salt_expert.onnx` | 2 | `input` → `restored` |
| `task2_blur_expert.onnx` | 2 | `input` → `restored` |
| `task2_occlusion_expert.onnx` | 2 | `input` → `restored` |
| `task3_soft_moe.onnx` | 3 | `input` → `restored`, `weights` [N,4] (complete MoE pipeline, τ baked in) |
| `task4_generator.onnx` | 4 | `photo` float32 [N,3,128,128] in [-1,1], `style` int64 [N] in {0,1,2} → `sketch` [N,1,128,128] in [-1,1] |
| `model_card.json` | – | SHA-256 of every file, input specs, τ, class/style names |

## Option A — download (evaluators)

The files are published as `models.zip` on the repository's **GitHub Releases** page
(chosen over Git LFS because LFS bandwidth quotas are small and a release asset needs
no extra tooling).

```bash
# Linux / macOS / Git Bash
MODELS_URL=https://github.com/Ali-prog-spec/Restoration-Sketch-studio/releases/download/v1.0/models.zip sh scripts/download_models.sh
```
```powershell
# Windows PowerShell
$env:MODELS_URL="https://github.com/Ali-prog-spec/Restoration-Sketch-studio/releases/download/v1.0/models.zip"; .\scripts\download_models.ps1
```
Or download `models.zip` from the release page manually and unzip it here.

Release page: https://github.com/Ali-prog-spec/Restoration-Sketch-studio/releases/tag/v1.0
(`models.zip` is built with `python scripts/package_models.py` → `dist/models.zip`).

## Option B — reproduce

```bash
python scripts/run_pipeline.py --mode final      # trains everything and runs scripts/export_onnx.py
```
Every export is validated against PyTorch; see `outputs/onnx/validation_report.md`.

PyTorch checkpoints (`checkpoints/**/*.pt`) can be attached to the same release for completeness.
