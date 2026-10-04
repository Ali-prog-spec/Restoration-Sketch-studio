# Final compliance checklist (audit against `GenAI_Assignment#1.pdf`)

Audit date: 2026-10-04, after the FINAL pipeline completed. Statuses:
**EXPERIMENTALLY VERIFIED** (ran on real data, artifact exists) · **TESTED** (automated test) ·
**IMPLEMENTED** (code exists, not executed in the target environment) · **MANUAL ACTION REQUIRED** · **NOT STARTED**.

Requirement IDs refer to `REQUIREMENTS.md`.

## Summary

| Area | Status |
|---|---|
| Tasks 1–4: models trained, tuned, evaluated on the official test sets | EXPERIMENTALLY VERIFIED |
| Optuna in all four tasks (5 studies) | EXPERIMENTALLY VERIFIED |
| MLflow tracking | EXPERIMENTALLY VERIFIED (local `mlflow.db`) |
| ONNX export of all 7 deployed models + numerical check | EXPERIMENTALLY VERIFIED (all PASS, max diff ≤ 8.1e-6) |
| FastAPI backend + React/Tailwind frontend | TESTED (67 tests) + verified in a browser with the real models |
| Docker / Docker Compose | TESTED (`docker compose up --build`, app verified at localhost:8080) |
| Google Stitch evidence | PARTLY DONE (all 4 workspaces + System status + logo; Stitch prompt/project screenshot missing) |
| IEEE LaTeX report | IMPLEMENTED skeleton + generated tables/figures; **interpretation text and compilation: MANUAL** |
| GitHub repo + model release | DONE |
| YouTube demo link, author details, AI-use appendix review | **MANUAL ACTION REQUIRED** |

**The assignment is NOT complete yet**: the manual items below must be done by the student before submission.

---

## A. Global

| Req | Requirement (page) | Evidence / location | Status |
|---|---|---|---|
| A1 | Four systems designed, trained, evaluated, deployed (p.1) | `src/`, `outputs/task1..4/`, `models/` | EXPERIMENTALLY VERIFIED |
| A2–A5 | Research component; alternatives, evidence (p.1) | `docs/research_notes.md`, `docs/references.bib` (verified) | IMPLEMENTED — student must write it into the report |
| A6–A7 | Student understands / can modify the system (p.1) | readable modular code, configs | MANUAL (student preparation) |
| A8 | Browser application for all tasks (p.1) | `frontend/`, `backend/` | TESTED (browser-verified) |
| A9 | PyTorch implementation (p.2) | torch 2.14 | DONE |
| A10 | Submit via Google Classroom before deadline (p.1) | — | MANUAL |

## B. Datasets

| Req | Requirement (page) | Evidence | Status |
|---|---|---|---|
| B1–B6 | Oxford official trainval → 80/20 seed 42; test untouched; RGB 128×128; same split for Tasks 1–3 (p.2–3) | `scripts/prepare_oxford.py`, `data/manifests/oxford_split.json` (2,944/736/3,669); `oxford_source_verification.json` | EXPERIMENTALLY VERIFIED |
| B7–B10 | Runtime corruption in loader, equal probability, label known (p.3) | `RuntimeCorruptionDataset`, `BalancedBatchSampler`; `tests/test_manifests_and_data.py` | TESTED |
| B11–B14 | Exact corruption definitions (p.3) | `src/corruptions/`; `tests/test_corruptions.py` | TESTED |
| B15 | Deterministic val/test manifests with type, severity, coords, blur settings, seed (p.3) | `data/manifests/oxford_{val,test}_manifest.json`; reproducibility test | TESTED |
| B16–B19 | Test levels 0.03/0.08/0.15; (3,0.7)/(5,1.5)/(7,2.5); ≈10/20/35 % with 1/2/3 rects (p.3) | `outputs/tables/manifest_summary_test.csv` (coverage means 0.100 / 0.198 / 0.345) | EXPERIMENTALLY VERIFIED |
| B30–B34 | FS2K official split; 15 % val stratified by style, seed 42; 128×128; pairing (p.7) | `data/manifests/fs2k_split.json` (899/159/1,046) | EXPERIMENTALLY VERIFIED |
| B35 | Identical paired augmentation (p.8) | `src/data/paired_transforms.py`; `tests/test_paired_augmentation.py` | TESTED |

## C. Task 1

| Req | Requirement | Evidence | Status |
|---|---|---|---|
| C1–C7 | Single conv DAE, genuine bottleneck, no unrestricted skips (p.3–4) | `src/models/autoencoder.py`; selected bottleneck 16×16×8 (24× compression); `test_no_skip_by_default…` | TESTED + EXPERIMENTALLY VERIFIED |
| C6 | Limited-skip investigation (p.4) | ablation `checkpoints/task1/universal_dae_skip0.pt`, `outputs/task1/metrics_by_type_skip0.csv` | EXPERIMENTALLY VERIFIED — discuss in report |
| C8–C9 | α·L1 + (1−α)(1−SSIM), α tuned (p.4) | `src/losses/`; best α = 0.53 | EXPERIMENTALLY VERIFIED |
| C10–C12 | Optuna: lr, batch, bottleneck, channels, dropout, α; objective = recon + SSIM; report space/trials/best (p.4) | `outputs/task1/optuna/final/` (30 trials: 20 complete, 10 pruned), `report/tables/optuna_task1.tex` | EXPERIMENTALLY VERIFIED |
| C13–C14 | Results per condition and per severity (p.4) | `outputs/task1/metrics_by_type.csv`, `metrics_by_level.csv`, `report/tables/task1_*.tex` | EXPERIMENTALLY VERIFIED |
| C15–C16 | Target/input/output/error map; ≥ 12 examples, ≥ 4 failures (p.4) | `report/figures/task1_representative_{1,2}.png` (12), `task1_failures_{1,2}.png` (7) | EXPERIMENTALLY VERIFIED — **discussion text MANUAL** |
| C17 | ONNX export (p.4) | `models/task1_universal_dae.onnx` | EXPERIMENTALLY VERIFIED |
| C18–C20 | "Universal Restoration" workspace: upload or sample + corruption; input, output, settings, time (p.4) | `frontend/src/workspaces/Workspaces.jsx`, `report/figures/app_universal.png` | TESTED |

## D. Task 2

| Req | Requirement | Evidence | Status |
|---|---|---|---|
| D1–D5 | 4-class CNN, runtime labels, balanced batches, CE (p.4–5) | `src/models/classifier.py`, `src/training/train_classifier.py` | EXPERIMENTALLY VERIFIED |
| D6 | Optuna: lr, batch, channels, dropout, weight decay (p.5) | `outputs/task2/classifier/optuna/final/` (25 trials) | EXPERIMENTALLY VERIFIED |
| D7 | Accuracy, macro P/R/F1, per-class, normalised confusion matrix (p.5) | `outputs/task2/test_summary.json` (acc 0.9942, macro-F1 0.9904), `classifier_per_class.csv`, `report/figures/task2_confusion_matrix.png` | EXPERIMENTALLY VERIFIED |
| D8–D12 | 3 specialists, own corruption only, shared Optuna search, independent training (p.5) | `checkpoints/task2/*_expert.pt`, `outputs/task2/specialists/optuna/final/` (20 trials) | EXPERIMENTALLY VERIFIED |
| D13–D14 | Hard routing with identity bypass (p.5) | `src/models/moe.py::hard_route`, `backend/app/inference.py`; test | TESTED |
| D15–D17 | Oracle vs predicted; classifier-caused failures identified (p.5) | `outputs/task2/oracle_vs_predicted_by_level.csv`, `classifier_caused_failures.csv` (34), `report/figures/task2_routing_failures.png` | EXPERIMENTALLY VERIFIED — **discussion MANUAL** |
| D18 | "Hard-Routed Restoration" workspace (p.5) | `report/figures/app_hard.png` | TESTED |
| D19 | ONNX: classifier + 3 specialists (p.5) | `models/task2_*.onnx` | EXPERIMENTALLY VERIFIED |

## E. Task 3

| Req | Requirement | Evidence | Status |
|---|---|---|---|
| E1–E6 | Soft MoE, identity branch, w = softmax(G/τ), weighted sum (p.5–6) | `src/models/moe.py`; tests (weights sum to 1, identity) | TESTED |
| E7–E9 | Init from Task 2; frozen-expert warm-up; joint fine-tune with smaller lr (p.6) | `src/training/train_moe.py`; `outputs/task3/history.csv` (stage column); initial J 0.0963 → best 0.0798 | EXPERIMENTALLY VERIFIED |
| E10–E14 | Joint loss with L1, SSIM, CE, balance; PDF start values (p.6) | `moe_loss`; trial 0 = PDF values | EXPERIMENTALLY VERIFIED |
| E13 | Alternative balance regulariser, research-supported (p.6) | entropy balance + CV² equivalence (`docs/research_notes.md` §4.4); ablation identical result because tuned λ_b ≈ 2e-4 makes the term negligible | EXPERIMENTALLY VERIFIED — **must be discussed honestly** |
| E15–E16 | Optuna: joint lr, τ, CE weight, balance weight, recon weighting; collapse pruning (p.6) | `outputs/task3/optuna/final/` (20 trials, 5 pruned) | EXPERIMENTALLY VERIFIED |
| E17–E20 | Weights per corruption and severity; dominant/mixed examples; heatmap; inactive/dominance check (p.7) | `routing_mean_weights_by_*.csv`, `report/figures/task3_routing_heatmap.png`, `task3_dominant_examples.png`, `task3_mixed_examples.png`, `routing_diagnostics.json` (no inactive expert; identity takes ~0.39 weight on blur/occlusion) | EXPERIMENTALLY VERIFIED — **discussion MANUAL** |
| E21 | "Soft Mixture-of-Experts Restoration" workspace (p.7) | `report/figures/app_soft_moe.png` | TESTED |
| E22 | Complete MoE pipeline in ONNX (p.7) | `models/task3_soft_moe.onnx` (outputs restored + weights) | EXPERIMENTALLY VERIFIED |

## F. Task 4

| Req | Requirement | Evidence | Status |
|---|---|---|---|
| F1–F8 | U-Net G(x,s), PatchGAN D(x,y,s), learned embedding in G and D (p.7) | `src/models/pix2pix.py`; style-influence test; `test_summary.json → style_influence` (mean |ΔG| between styles 0.097–0.116); `report/figures/task4_style_swap.png` | TESTED + EXPERIMENTALLY VERIFIED |
| F9–F11 | L_adv + λ·L1, BCE-with-logits, λ tuned (p.7–8) | `train_gan.py`; best λ_L1 = 148.7 | EXPERIMENTALLY VERIFIED |
| F12–F13 | Optuna on 7 params, short trials, full retrain (p.8) | `outputs/task4/optuna/final/` (20 finished trials × 25 epochs + 2 trials interrupted by a process kill, left in state RUNNING); retrain 200 epochs | EXPERIMENTALLY VERIFIED |
| F14 | Paired augmentation (p.8) | see B35 | TESTED |
| F15–F16 | Separate D-real/D-fake/G-adv/G-L1 + val metrics; fixed val photos over epochs (p.8) | `outputs/task4/history.csv`, `report/figures/task4_gan_losses.png`, `task4_progression.png`, `outputs/task4/progress/` | EXPERIMENTALLY VERIFIED |
| — | Test results (official FS2K test) | `outputs/task4/test_metrics_by_style.csv` (overall L1 0.111, SSIM 0.462, PSNR 15.0 dB) | EXPERIMENTALLY VERIFIED |
| F17 | "Face-to-Sketch Generator" workspace: upload/webcam, Style 1/2/3, side by side, download (p.8) | `frontend/src/workspaces/FaceToSketch.jsx` | TESTED (API) — **webcam to be tried manually in a browser** |
| F19 | Generator exported to ONNX (p.8) | `models/task4_generator.onnx` | EXPERIMENTALLY VERIFIED |

## G–J. Optuna, tracking, evaluation, ONNX

| Req | Requirement | Evidence | Status |
|---|---|---|---|
| G1–G5 | Optuna in all four tasks, studies in repo, results reported | `optuna_studies/*.db`, `outputs/*/optuna/final/`, `report/tables/optuna_*.tex` | EXPERIMENTALLY VERIFIED |
| H1–H2 | MLflow: params, losses, metrics, checkpoints, visuals | `mlflow.db` + `mlartifacts/` (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) | EXPERIMENTALLY VERIFIED |
| H3 | Tracking records shown in the video | — | MANUAL |
| J1–J2 | All inference models in ONNX, consistency verified | `outputs/onnx/validation_report.md` (8 outputs, all PASS) | EXPERIMENTALLY VERIFIED |
| J3 | ONNX files included or downloadable | GitHub Release v1.0 (`models.zip` + `models.sha256`); download + checksum + unzip verified | DONE |

## K–L. Application and deployment

| Req | Requirement | Evidence | Status |
|---|---|---|---|
| K1, K4–K10 | One React/Tailwind app, FastAPI backend, validation, ONNX inference, timing, required operations | `frontend/`, `backend/`, `tests/test_api.py` | TESTED |
| K2–K3 | Design first developed in Google Stitch; evidence in report | `report/stitch/` (Tasks 1–4, System status, logo); report section with honest note on Stitch placeholder content | PARTLY DONE — **Stitch prompt/project screenshot MANUAL** |
| L1–L3 | Docker containers, one-command Compose, clone → models → up → browser | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` | TESTED (both containers built and ran; all 7 models loaded) |

## M–P. Report, demo, repository, AI use

| Req | Requirement | Evidence | Status |
|---|---|---|---|
| M1–M3, M6–M8 | IEEE LaTeX report with all sections, diagrams, curves, tables, screenshots | `report/main.tex`, `report/figures/` (51 files), `report/tables/` (17 tables) | IMPLEMENTED — **interpretation text (red TODOs) MANUAL; compile on Overleaf** |
| M4 | Repository URL in report | `report/main.tex` author block | DONE |
| M5, N1–N2 | 5–7 min YouTube demo, link in report | demo checklist in `README.md` | MANUAL |
| M9–M10 | Interpretation of every figure/table; alternatives and difficulties | research notes + PROGRESS problem log as source material | MANUAL |
| O1 | GitHub repository | https://github.com/Ali-prog-spec/Restoration-Sketch-studio | DONE |
| O2 | Repo contents (code, configs, deps, data prep, train/eval, Optuna studies, ONNX code, app, Dockerfiles, Compose, README) | all present | DONE |
| O3 | No datasets / large models in Git | `.gitignore` (data, checkpoints, `*.onnx`) | DONE |
| P1–P2 | AI-use appendix | `docs/AI_USE.md` draft | **MANUAL: student must edit to reflect actual use** |
