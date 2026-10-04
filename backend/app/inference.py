"""ONNX Runtime model registry and the four inference pipelines."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.corruptions.constants import CLASS_DISPLAY, CLASS_NAMES, STYLE_NAMES

MODEL_FILES = {
    "task1": "task1_universal_dae.onnx",
    "task2_classifier": "task2_classifier.onnx",
    "task2_salt_pepper": "task2_salt_expert.onnx",
    "task2_blur": "task2_blur_expert.onnx",
    "task2_occlusion": "task2_occlusion_expert.onnx",
    "task3": "task3_soft_moe.onnx",
    "task4": "task4_generator.onnx",
}
BRANCHES = ["identity", "salt_pepper", "blur", "occlusion"]
BRANCH_DISPLAY = {"identity": "Identity (clean bypass)", "salt_pepper": "Salt-and-pepper expert",
                  "blur": "Blur expert", "occlusion": "Occlusion expert"}


class ModelUnavailable(RuntimeError):
    pass


class ModelRegistry:
    """Loads ONNX sessions lazily (first request) and caches them."""

    def __init__(self, model_dir: str | Path, threads: int = 0):
        self.model_dir = Path(model_dir)
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._lock = threading.Lock()
        self._opts = ort.SessionOptions()
        if threads:
            self._opts.intra_op_num_threads = threads
        card = self.model_dir / "model_card.json"
        self.card = json.loads(card.read_text(encoding="utf-8")) if card.exists() else {}

    def path(self, key: str) -> Path:
        return self.model_dir / MODEL_FILES[key]

    def available(self, key: str) -> bool:
        return self.path(key).exists()

    def session(self, key: str) -> ort.InferenceSession:
        if key not in self._sessions:
            with self._lock:
                if key not in self._sessions:
                    p = self.path(key)
                    if not p.exists():
                        raise ModelUnavailable(f"model file '{p.name}' not found in {self.model_dir} — "
                                               "download the trained models (see models/README.md)")
                    self._sessions[key] = ort.InferenceSession(str(p), self._opts, providers=["CPUExecutionProvider"])
        return self._sessions[key]

    def status(self) -> dict:
        return {k: ("loaded" if k in self._sessions else "available" if self.available(k) else "missing")
                for k in MODEL_FILES}

    def model_info(self, key: str) -> dict:
        info = self.card.get("models", {}).get(key, {})
        return {"key": key, "file": MODEL_FILES[key], "sha256": info.get("sha256")}


def _run(sess: ort.InferenceSession, feeds: dict):
    t0 = time.perf_counter()
    out = sess.run(None, feeds)
    return out, (time.perf_counter() - t0) * 1000.0


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def universal(reg: ModelRegistry, x: np.ndarray) -> dict:
    (y,), ms = _run(reg.session("task1"), {"input": x})
    return {"output": y, "inference_ms": ms, "models": [reg.model_info("task1")]}


def hard_routing(reg: ModelRegistry, x: np.ndarray) -> dict:
    (logits,), ms_c = _run(reg.session("task2_classifier"), {"input": x})
    probs = softmax(logits)[0]
    pred = int(np.argmax(probs))
    name = CLASS_NAMES[pred]
    models = [reg.model_info("task2_classifier")]
    if name == "clean":  # identity bypass — no expert is executed
        y, ms_e, expert = x, 0.0, "identity"
    else:
        key = f"task2_{name}"
        (y,), ms_e = _run(reg.session(key), {"input": x})
        expert = name
        models.append(reg.model_info(key))
    return {
        "output": y, "inference_ms": ms_c + ms_e,
        "timing_detail_ms": {"classifier": ms_c, "expert": ms_e},
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "predicted_class": name, "predicted_display": CLASS_DISPLAY[name],
        "selected_expert": expert, "selected_expert_display": BRANCH_DISPLAY[expert],
        "models": models,
    }


def soft_mixture(reg: ModelRegistry, x: np.ndarray) -> dict:
    (y, w), ms = _run(reg.session("task3"), {"input": x})
    w = w[0]
    order = np.argsort(-w)
    return {
        "output": y, "inference_ms": ms,
        "weights": {b: float(v) for b, v in zip(BRANCHES, w)},
        "ranking": [{"branch": BRANCHES[i], "display": BRANCH_DISPLAY[BRANCHES[i]], "weight": float(w[i])} for i in order],
        "dominant_branch": BRANCHES[int(order[0])],
        "entropy_nats": float(-(w * np.log(w + 1e-12)).sum()),
        "tau": reg.card.get("task3_tau"),
        "models": [reg.model_info("task3")],
    }


def face_to_sketch(reg: ModelRegistry, photo_pm1: np.ndarray, style_index: int) -> dict:
    (y,), ms = _run(reg.session("task4"), {"photo": photo_pm1, "style": np.array([style_index], dtype=np.int64)})
    return {"output": y, "inference_ms": ms, "style": STYLE_NAMES[style_index], "style_index": style_index,
            "models": [reg.model_info("task4")]}
