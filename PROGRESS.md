# PROGRESS

Status vocabulary: NOT STARTED · IMPLEMENTED · TESTED · EXPERIMENTALLY VERIFIED · MANUAL ACTION REQUIRED · BLOCKED

_Last update: 2026-10-04 — FINAL pipeline complete. Full audit: [docs/final_compliance_checklist.md](docs/final_compliance_checklist.md)._

## Phase summary

| Phase | Content | Status |
|---|---|---|
| 0 | PDF read, REQUIREMENTS.md, research notes, verified bibliography | DONE |
| 1 | Repo structure, configs (smoke/dev/final), seeds, env recording, MLflow | TESTED |
| 2 | Oxford pipeline, runtime corruptions, balanced batches, deterministic manifests | EXPERIMENTALLY VERIFIED |
| 3 | Task 1 universal DAE (+ limited-skip ablation) | EXPERIMENTALLY VERIFIED |
| 4 | Task 2 classifier + specialists + hard routing (oracle vs predicted) | EXPERIMENTALLY VERIFIED |
| 5 | Task 3 soft MoE (+ balance-loss ablation) | EXPERIMENTALLY VERIFIED |
| 6 | Task 4 style-conditioned pix2pix on FS2K | EXPERIMENTALLY VERIFIED |
| 7 | FastAPI backend | TESTED + browser-verified with real models |
| 8 | React + Tailwind UI (restyled to the student's Google Stitch design) | TESTED + browser-verified |
| 9 | Docker / Compose | TESTED (built and run with Docker Desktop) |
| 10 | Report artifacts (51 figures, 17 LaTeX tables) + IEEE skeleton | IMPLEMENTED — text and compilation MANUAL |
| 11 | Final PDF compliance audit | DONE |

## Headline results (official test sets; every number comes from `outputs/`)

| System | Result |
|---|---|
| Task 1 universal DAE | overall SSIM 0.786 (corrupted input 0.672), PSNR 25.05 dB; limited-skip ablation SSIM 0.839 |
| Task 2 classifier | accuracy 0.9942, macro-F1 0.9904 |
| Task 2 hard routing | SSIM oracle 0.8018 vs predicted 0.8023; 214 misrouted, 34 classifier-caused failures |
| Task 3 soft MoE | overall SSIM 0.838, PSNR 27.57 dB; no inactive expert; validation J 0.0963 (Task 2 init) → 0.0798 |
| Task 4 generator | test L1 0.111, SSIM 0.462, PSNR 15.05 dB (Style 1 / 2 / 3 SSIM 0.507 / 0.372 / 0.606) |
| ONNX | 7 models, 8 outputs, all PASS (max abs diff ≤ 8.1e-6) |

## Findings to discuss in the report
- The no-skip bottleneck costs fidelity on clean inputs (SSIM 0.83 instead of 1.0); the limited skip improves every condition.
- Balance-loss ablation gives identical results: Optuna chose λ_b ≈ 2e-4, so the balance term is negligible.
- The MoE gives the identity branch ~0.39 weight on blur/occlusion inputs (soft hedging, not collapse).
- FS2K: Style 2 is hardest (SSIM 0.372); the test set is style-imbalanced (619/381/46); style is confounded with photo source.
- Task 4 Optuna study holds 2 trials left in state RUNNING: they were killed when the process was closed and never finished.

## Problems encountered (and fixes)
| Problem | Fix |
|---|---|
| System Python 3.14 with CPU-only torch | project venv with Python 3.11 + torch 2.14 cu126 |
| Official Oxford server ~150 KB/s, no HTTP range support | verified HF mirror (`scripts/fetch_oxford_mirror.py`) |
| `gdown --id` removed in gdown 6 | positional id |
| ONNX dynamo exporter crashes on Windows console encoding | UTF-8 stdout reconfiguration |
| First Task 1 trial: low SSIM with an 8×8 latent only | search over bottleneck shapes; study restarted |
| Smoke tables written into `report/tables` | only FINAL mode writes into `report/` |
| pandas 3 `groupby.apply` drops the grouping column (Task 3 evaluation crash) | explicit per-group sampling |
| Background training killed by session time limits / closed console window | relaunched as hidden detached processes; Optuna studies resumed |

## Manual steps for the student (required before submission)
1. Google Stitch: add a **Stitch prompt/project screenshot** (`report/stitch/stitch_prompt.png`). (All 4 workspace designs are in.)
2. ~~Docker~~ done.
3. ~~GitHub repo + release v1.0~~ done.
4. Write the report interpretation (all red TODOs), compile on Overleaf, add repo URL.
5. Record the 5–7 min demo video (checklist in README), upload to YouTube, put the link in the report.
6. Edit `docs/AI_USE.md` to reflect actual AI usage; copy it into the report appendix.
7. Submit on Google Classroom.

## Commands
See `commands.txt` (run the app, check training, MLflow, tests, export, release).
