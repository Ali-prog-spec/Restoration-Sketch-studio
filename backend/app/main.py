"""FastAPI backend for the four workspaces.

Endpoints
  GET  /health                      service + model availability
  GET  /api/info                    model card (classes, styles, tau, file hashes)
  GET  /api/samples                 clean sample images the user can pick
  GET  /api/samples/{sample_id}     one sample image (PNG/JPEG)
  POST /api/universal-restoration   Task 1
  POST /api/hard-routing            Task 2
  POST /api/soft-mixture            Task 3
  POST /api/face-to-sketch          Task 4

Restoration endpoints take multipart/form-data:
  file (optional upload) OR sample_id,
  corruption  = none | salt_pepper | blur | occlusion   (applied server-side at runtime)
  severity    = low | medium | high | custom
  prob / kernel / sigma / coverage / num_rects          (used when severity = custom)
  seed        (optional; returned so a result can be reproduced)

Images are returned inside the JSON as base64 PNG data URLs ("data:image/png;base64,...").
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import fastapi
import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.corruptions import apply_corruption, level_params
from src.corruptions.constants import CLASS_DISPLAY, SEVERITY_LEVELS, STYLE_NAMES, TEST_LEVELS
from src.corruptions.corruptions import sample_rectangles
from src.utils.image_io import resize_rgb

from . import inference as inf
from .imaging import data_url, decode_image, preprocess_restoration, psnr, read_upload, ssim, to_nchw, from_nchw

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", ROOT / "backend" / "samples"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))
MAX_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
CORS_ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",") if o]
VERSION = "1.0.0"

app = FastAPI(title="GenAI Assignment 1 — Restoration & Face-to-Sketch API", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
registry = inf.ModelRegistry(MODEL_DIR, threads=int(os.environ.get("ORT_THREADS", "0")))
SAMPLE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@app.exception_handler(inf.ModelUnavailable)
async def _missing_model(_, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """The backend has no web page; point people to the app and the API docs."""
    return {"service": "GenAI Assignment 1 backend (FastAPI + ONNX Runtime)",
            "web_app": "http://localhost:8080 (Docker) or http://localhost:5173 (npm run dev)",
            "api_docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    status = registry.status()
    files = {}
    for key in status:
        p = registry.path(key)
        files[key] = {"file": p.name, "size_mb": round(p.stat().st_size / 1024**2, 2) if p.exists() else None,
                      "sha256": registry.model_info(key)["sha256"]}
    return {"status": "ok", "version": VERSION, "model_dir": str(MODEL_DIR), "models": status,
            "model_files": files,
            "all_models_available": all(v != "missing" for v in status.values()),
            "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
            "python": platform.python_version(), "fastapi": fastapi.__version__}


@app.get("/api/info")
def info():
    return {"model_card": registry.card, "classes": CLASS_DISPLAY, "styles": STYLE_NAMES,
            "test_levels": TEST_LEVELS, "max_upload_mb": MAX_UPLOAD_MB}


def _samples():
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in SAMPLE_EXT)


@app.get("/api/samples")
def samples():
    return [{"id": p.stem, "name": p.stem.replace("_", " "), "url": f"/api/samples/{p.stem}"} for p in _samples()]


@app.get("/api/samples/{sample_id}")
def sample(sample_id: str):
    for p in _samples():
        if p.stem == sample_id:
            return FileResponse(p)
    raise HTTPException(404, "sample not found")


# ---------------------------------------------------------------------------
# shared request handling for Tasks 1-3
# ---------------------------------------------------------------------------
def build_corruption(corruption: str, severity: str, seed: int, prob, kernel, sigma, coverage, num_rects) -> dict | None:
    if corruption in ("none", "", None):
        return None
    if corruption not in ("salt_pepper", "blur", "occlusion", "clean"):
        raise HTTPException(422, f"unknown corruption '{corruption}'")
    if corruption == "clean":
        return None
    if severity in SEVERITY_LEVELS:
        return {**level_params(corruption, severity, seed), "severity": severity}
    if severity != "custom":
        raise HTTPException(422, "severity must be low, medium, high or custom")
    if corruption == "salt_pepper":
        p = 0.08 if prob is None else float(prob)
        if not 0.0 < p <= 0.5:
            raise HTTPException(422, "prob must be in (0, 0.5]")
        return {"type": "salt_pepper", "prob": p, "seed": seed, "severity": "custom"}
    if corruption == "blur":
        k = 5 if kernel is None else int(kernel)
        s = 1.5 if sigma is None else float(sigma)
        if k not in (3, 5, 7, 9, 11) or not 0.1 <= s <= 5.0:
            raise HTTPException(422, "kernel must be odd in 3..11 and sigma in [0.1, 5]")
        return {"type": "blur", "kernel": k, "sigma": s, "severity": "custom"}
    cov = 0.2 if coverage is None else float(coverage)
    n = 2 if num_rects is None else int(num_rects)
    if not 0.01 <= cov <= 0.6 or not 1 <= n <= 5:
        raise HTTPException(422, "coverage must be in [0.01, 0.6] and num_rects in 1..5")
    rects, achieved = sample_rectangles(np.random.default_rng(seed), 128, 128, n, cov)
    return {"type": "occlusion", "rects": rects, "coverage": achieved, "num_rects": n, "severity": "custom"}


async def prepare_input(file, sample_id, corruption, severity, seed, prob, kernel, sigma, coverage, num_rects):
    t0 = time.perf_counter()
    if file is not None and getattr(file, "filename", ""):
        img = decode_image(await read_upload(file, MAX_BYTES))
        source = {"kind": "upload", "filename": file.filename, "original_size": list(img.size)}
    elif sample_id:
        match = [p for p in _samples() if p.stem == sample_id]
        if not match:
            raise HTTPException(404, "sample not found")
        img = decode_image(match[0].read_bytes())
        source = {"kind": "sample", "sample_id": sample_id, "original_size": list(img.size)}
    else:
        raise HTTPException(422, "provide an image file or a sample_id")
    clean = preprocess_restoration(img)
    t_pre = (time.perf_counter() - t0) * 1000
    seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
    params = build_corruption(corruption, severity, seed, prob, kernel, sigma, coverage, num_rects)
    t1 = time.perf_counter()
    x = apply_corruption(clean, params) if params else clean
    t_cor = (time.perf_counter() - t1) * 1000
    return clean, x, params, source, {"preprocess": t_pre, "corruption": t_cor}


def restoration_response(task, clean, x, y, params, source, timing, result):
    out = np.clip(from_nchw(y), 0, 1)
    resp = {
        "task": task,
        "source": source,
        "corruption": params,
        "corruption_display": CLASS_DISPLAY[params["type"]] if params else "None (input used as given)",
        "input_image": data_url(x),
        "output_image": data_url(out),
        "reference_image": data_url(clean) if params else None,
        "timing_ms": {**{k: round(v, 2) for k, v in timing.items()}, "inference": round(result["inference_ms"], 2)},
        "models": result["models"],
    }
    resp["timing_ms"]["total"] = round(sum(v for k, v in resp["timing_ms"].items() if k != "total"), 2)
    if params:  # a clean reference exists -> report quality numbers
        resp["metrics"] = {"input_psnr": psnr(x, clean), "output_psnr": psnr(out, clean),
                           "input_ssim": ssim(x, clean), "output_ssim": ssim(out, clean)}
    for k in ("probabilities", "predicted_class", "predicted_display", "selected_expert", "selected_expert_display",
              "timing_detail_ms", "weights", "ranking", "dominant_branch", "entropy_nats", "tau"):
        if k in result:
            resp[k] = result[k]
    return resp


class RestorationForm:
    """Multipart fields shared by the three restoration endpoints."""

    def __init__(self, sample_id: str | None = Form(None), corruption: str = Form("none"),
                 severity: str = Form("medium"), seed: int | None = Form(None), prob: float | None = Form(None),
                 kernel: int | None = Form(None), sigma: float | None = Form(None),
                 coverage: float | None = Form(None), num_rects: int | None = Form(None)):
        self.sample_id, self.corruption, self.severity, self.seed = sample_id, corruption, severity, seed
        self.prob, self.kernel, self.sigma, self.coverage, self.num_rects = prob, kernel, sigma, coverage, num_rects


async def _restore(task: str, runner, file: UploadFile | None, f: RestorationForm):
    clean, x, params, source, timing = await prepare_input(file, f.sample_id, f.corruption, f.severity, f.seed, f.prob,
                                                           f.kernel, f.sigma, f.coverage, f.num_rects)
    res = runner(registry, to_nchw(x))
    return restoration_response(task, clean, x, res["output"], params, source, timing, res)


@app.post("/api/universal-restoration")
async def universal_restoration(file: UploadFile | None = File(None), form: RestorationForm = Depends()):
    return await _restore("universal-restoration", inf.universal, file, form)


@app.post("/api/hard-routing")
async def hard_routing(file: UploadFile | None = File(None), form: RestorationForm = Depends()):
    return await _restore("hard-routing", inf.hard_routing, file, form)


@app.post("/api/soft-mixture")
async def soft_mixture(file: UploadFile | None = File(None), form: RestorationForm = Depends()):
    return await _restore("soft-mixture", inf.soft_mixture, file, form)


@app.post("/api/face-to-sketch")
async def face_to_sketch(file: UploadFile = File(...), style: int = Form(1)):
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2 or 3")
    t0 = time.perf_counter()
    img = decode_image(await read_upload(file, MAX_BYTES))
    photo = resize_rgb(img, 128)
    x = to_nchw(photo.astype(np.float32) / 127.5 - 1.0)
    t_pre = (time.perf_counter() - t0) * 1000
    res = inf.face_to_sketch(registry, x, style - 1)
    sketch = np.clip((res["output"][0].transpose(1, 2, 0) + 1) / 2, 0, 1)
    return {
        "task": "face-to-sketch",
        "source": {"kind": "upload", "filename": file.filename, "original_size": list(img.size)},
        "style": res["style"], "style_index": style,
        "input_image": data_url(photo),
        "output_image": data_url(sketch),
        "timing_ms": {"preprocess": round(t_pre, 2), "inference": round(res["inference_ms"], 2),
                      "total": round(t_pre + res["inference_ms"], 2)},
        "models": res["models"],
    }
