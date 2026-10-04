# Generative AI Assignment 1 — Restoration & Sketch Studio

Four generative systems, trained, evaluated and deployed as one browser application:

| Task | System | Data |
|---|---|---|
| 1 | **Universal Restoration** — one convolutional denoising autoencoder with a genuine bottleneck (no skip connections) | Oxford-IIIT Pet |
| 2 | **Hard-Routed Restoration** — CNN corruption classifier → one of three specialist autoencoders (identity bypass for clean) | Oxford-IIIT Pet |
| 3 | **Soft Mixture-of-Experts Restoration** — gate (init. from Task 2 classifier) mixes identity + three experts; warm-up then joint fine-tuning | Oxford-IIIT Pet |
| 4 | **Face-to-Sketch Generator** — pix2pix U-Net + PatchGAN, learned style embedding in *both* networks | FS2K |

Corruptions (clean, salt-and-pepper, Gaussian blur, rectangular occlusion) are generated **at runtime**
for training and from **deterministic manifests** for validation/test. Optuna tunes every task, MLflow
records every run, every deployed model is exported to ONNX and checked against PyTorch, and a
FastAPI + React/Tailwind app runs in Docker Compose.

- Requirements checklist: [REQUIREMENTS.md](REQUIREMENTS.md) · progress/status: [PROGRESS.md](PROGRESS.md)
- Design decisions and evidence: [docs/research_notes.md](docs/research_notes.md) · references: [docs/references.bib](docs/references.bib)
- Google Stitch instructions (manual step): [docs/STITCH_GUIDE.md](docs/STITCH_GUIDE.md) · AI use: [docs/AI_USE.md](docs/AI_USE.md)
- Final PDF compliance audit: [docs/final_compliance_checklist.md](docs/final_compliance_checklist.md)

---

## Quick start for evaluators (application only)

Prerequisites: Git and Docker Desktop (with Docker Compose v2). No Python, Node or VS Code needed.

```bash
git clone https://github.com/Ali-prog-spec/Restoration-Sketch-studio.git
cd Restoration-Sketch-studio
# 1) obtain the trained ONNX models (GitHub Release v1.0, models.zip 171 MB)
sh scripts/download_models.sh                # Windows: powershell -File scripts\download_models.ps1
# 2) start everything
docker compose up --build
# 3) open http://localhost:8080
```

`http://localhost:8080/#status` shows whether all seven models were found. The API is also exposed
at `http://localhost:8000` (interactive docs at `/docs`). Optional environment variables:
`MODEL_PATH` (host folder with the models, default `./models`), `FRONTEND_PORT` (8080),
`BACKEND_PORT` (8000), `MAX_UPLOAD_MB` (10), `ORT_THREADS`.

---

## Architecture

```
Browser ──► frontend container (nginx + React/Tailwind build) ──/api──► backend container (FastAPI + ONNX Runtime)
                                                                            │  validate upload · RGB + 128×128
                                                                            │  runtime corruption (src/corruptions)
                                                                            ▼  inference · base64 PNG + JSON
                                                                     ./models/*.onnx (mounted read-only)
```

Backend endpoints: `GET /health`, `GET /api/info`, `GET /api/samples`, `POST /api/universal-restoration`,
`POST /api/hard-routing`, `POST /api/soft-mixture`, `POST /api/face-to-sketch` (details in
[backend/app/main.py](backend/app/main.py)).

## Repository structure

```
configs/            task1.yaml, task2_classifier.yaml, task2_specialists.yaml, task3.yaml, task4.yaml
                    (base settings + smoke/dev/final modes + Optuna search spaces)
data/               README (download), manifests/ (splits + deterministic corruption manifests, committed)
src/corruptions/    corruption definitions (NumPy only; shared by training and backend)
src/data/           Oxford runtime/manifest datasets, balanced batch sampler, FS2K pairs, paired augmentation
src/models/         autoencoder, classifier, soft MoE + hard routing, style-conditioned pix2pix
src/losses/         SSIM, L1+SSIM, routing-balance losses
src/training/       training loops (AE, classifier, MoE, GAN) + shared utilities
src/evaluation/     metrics, restoration/classification evaluation, plots, LaTeX tables
src/optimization/   Optuna search-space handling and resumable studies
src/export/         ONNX export + validation
src/utils/          config, seeds, MLflow tracker, environment info, image I/O
scripts/            prepare_*, generate_manifests, optimize_*, train_*, evaluate_*, export_onnx,
                    run_pipeline, make_report_artifacts, download/package models
backend/            FastAPI app, Dockerfile, demo samples
frontend/           React + Tailwind app, Dockerfile, nginx.conf
report/             IEEE LaTeX report (main.tex), generated figures/ and tables/, stitch/ evidence
optuna_studies/     SQLite Optuna studies (committed)
tests/              pytest suite
```

---

## Reproducing the experiments

### 1. Python environment (Python 3.11)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126   # or /whl/cpu
pip install -r requirements.txt
```

### 2. Datasets
Follow [data/README.md](data/README.md) (Oxford-IIIT Pet via official server or the verified mirror;
FS2K from the official Google Drive link), then:
```bash
python scripts/prepare_oxford.py          # RGB 128×128 cache + 80/20 split, seed 42
python scripts/generate_manifests.py      # ONCE: deterministic val/test corruption manifests
python scripts/prepare_fs2k.py            # official split + 15 % stratified validation, seed 42
```

### 3. Tests and smoke test
```bash
pytest                                     # unit + API tests (~1 min)
python scripts/run_pipeline.py --mode smoke   # every script on a tiny subset (~5 min)
```

### 4. Full pipeline (or step by step)
```bash
python scripts/run_pipeline.py --mode final   # all tasks, logs in outputs/logs/final/
```
Every script accepts `--mode smoke|dev|final`, `--smoke-test`, `--epochs`, `--batch-size`, `--num-workers`,
`--device`, `--limit-data`, `--num-trials`, `--no-tracking`, `--seed`.

| Step | Command |
|---|---|
| Task 1 Optuna / train / eval | `python scripts/optimize_task1.py --mode final` · `python scripts/train_task1.py --mode final --use-best` · `python scripts/evaluate_task1.py --mode final` |
| Task 1 limited-skip ablation | `python scripts/train_task1.py --mode final --use-best --skip-levels 0 --tag skip0` · `python scripts/evaluate_task1.py --mode final --tag skip0` |
| Task 2 classifier | `python scripts/optimize_task2_classifier.py --mode final` · `python scripts/train_task2_classifier.py --mode final --use-best` |
| Task 2 specialists | `python scripts/optimize_task2_specialists.py --mode final` · `python scripts/train_task2_specialists.py --mode final --use-best` |
| Task 2 eval (oracle vs predicted) | `python scripts/evaluate_task2.py --mode final` |
| Task 3 | `python scripts/optimize_task3.py --mode final` · `python scripts/train_task3.py --mode final --use-best` · `python scripts/evaluate_task3.py --mode final` |
| Task 3 balance ablation | `python scripts/train_task3.py --mode final --use-best --balance entropy --tag balance_ablation` |
| Task 4 | `python scripts/optimize_task4.py --mode final` · `python scripts/train_task4.py --mode final --use-best` · `python scripts/evaluate_task4.py --mode final --split test [--lpips]` |
| ONNX export + validation | `python scripts/export_onnx.py --mode final` → `models/`, `outputs/onnx/validation_report.md` |
| Report figures/tables | `python scripts/make_report_artifacts.py --mode final` |

Optuna studies are stored in `optuna_studies/*.db`; study names get the mode as suffix, `n_trials`
is the total wanted (re-running resumes). Trial history, best params, search space and plots:
`outputs/<task>/optuna/<mode>/`. The PDF baseline values are always trial 0.

### 5. Experiment tracking (MLflow)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```
Experiments: `task1-universal-dae`, `task2-classifier`, `task2-specialists`, `task3-soft-moe`,
`task4-face2sketch` (and `smoke-tests`). Each run stores config, hyperparameters, per-epoch losses/metrics,
figures, checkpoints and the Optuna best configuration.

### 6. Model distribution
`python scripts/package_models.py` → `dist/models.zip`; attach it to a GitHub Release (see
[models/README.md](models/README.md)).

---

## Development mode (without Docker)
```bash
uvicorn backend.app.main:app --reload --port 8000        # from the repo root, venv active
cd frontend && npm install && npm run dev                # http://localhost:5173 (proxies /api)
```

## Report
`report/main.tex` (IEEEtran). Build with `latexmk -pdf main.tex` in `report/` or upload the `report/`
folder together with `docs/references.bib` to Overleaf. Tables in `report/tables/` and figures in
`report/figures/` are generated from FINAL-mode artifacts only; red TODO / MISSING markers show what is
still to be written or generated.

## Demo checklist (5–7 min video, PDF p.2)
1. `docker compose up --build` and the status page · 2. image upload · 3. runtime corruption with severity ·
4. Universal Restoration · 5. Hard routing with the four probabilities and selected expert ·
6. Soft MoE weights · 7. Face-to-Sketch with webcam/upload and the three styles · 8. result download ·
9. MLflow UI records (runs, metrics, artifacts) and the Optuna plots.

## Troubleshooting
| Problem | Fix |
|---|---|
| Status page shows models "missing" | put the 7 `.onnx` files + `model_card.json` in `./models` (or set `MODEL_PATH`) and restart |
| `503` from an endpoint | that model file is missing (see `/health`) |
| Webcam does not start | browsers allow cameras only on `https://` or `localhost`; grant permission |
| ONNX export prints UnicodeEncodeError on Windows | run with `PYTHONIOENCODING=utf-8` (the export script also reconfigures stdout) |
| CUDA out of memory | lower `--batch-size`, or `--device cpu` for small runs |
| DataLoader slow / hangs on Windows | try `--num-workers 0` |
| Official Oxford server very slow | use the verified mirror path in data/README.md |

## Licences / attribution
Oxford-IIIT Pet: CC BY-SA 4.0 (Parkhi et al., 2012). FS2K: see the official repository (Fan et al., 2022).
Code written with AI assistance — see [docs/AI_USE.md](docs/AI_USE.md).
