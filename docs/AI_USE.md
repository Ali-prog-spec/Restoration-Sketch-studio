# AI-use appendix (draft — the student must edit this to reflect actual usage)

The assignment (PDF p.1–2) allows AI assistants but requires an appendix naming the tools,
what they were used for, and how their output was tested or corrected.

## Tools

| Tool | Version / model | Used for |
|---|---|---|
| Claude Code (Anthropic) | Claude Opus 5.5, Sept 2026 | Requirement extraction from the PDF, repository architecture, implementation of data pipeline / models / training / evaluation / ONNX / FastAPI / React code, test generation, documentation drafts, research discovery (citation verification via web search) |
| *(add others, e.g. ChatGPT, Copilot, Google Stitch AI generation)* | | |

## What the AI produced vs. what was verified

| Component | AI contribution | Verification performed |
|---|---|---|
| Requirement checklist (`REQUIREMENTS.md`) | extracted from the PDF page by page | *student: re-read the PDF and check every row* |
| Corruptions + manifests | implementation | unit tests (ranges, determinism, blur vs SciPy, coverage); manifest summary CSV inspected |
| SSIM / losses | implementation | SSIM compared with scikit-image (`tests/test_losses.py`); CV² equivalence test |
| Models (DAE, classifier, MoE, pix2pix) | implementation | shape tests, no-skip test, identity-branch test, style-influence test |
| Training / Optuna / MLflow | implementation | smoke pipeline (`scripts/run_pipeline.py --mode smoke`), final training logs, MLflow UI inspection |
| ONNX export | implementation | numerical comparison PyTorch vs ONNX Runtime (`outputs/onnx/validation_report.md`) |
| Backend / frontend | implementation | API tests incl. validation errors (`tests/test_api.py`), manual browser testing *(student)* |
| Citations | discovered and **verified** against DOI/arXiv/proceedings pages | *student: open each reference and confirm it supports the claim* |
| Report text | skeleton + auto-generated tables | all numbers come from `outputs/` artifacts; *student writes the interpretation* |

## Corrections made during development (examples, from the development log)

- Float-equality bug in a test → replaced with `pytest.approx`.
- ONNX dynamo exporter failed on Windows due to console encoding → UTF-8 stdout reconfiguration.
- Official dataset server ignored HTTP range requests → switched to a verified mirror (byte-level MD5 check).
- Initial Task 1 search space only varied latent channels at 8×8 → replaced by a bottleneck-shape search (12×–48× compression) after the first real trial showed low SSIM.
- Smoke-test LaTeX tables were written into `report/` → routed non-final outputs elsewhere.

