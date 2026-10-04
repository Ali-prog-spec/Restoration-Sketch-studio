"""Corruption functions for Tasks 1-3 (NumPy only, so the web backend can reuse them).

Images are float32 arrays of shape (H, W, 3) with values in [0, 1].

Every corruption is fully described by a JSON-serialisable ``params`` dict, e.g.

    {"type": "clean"}
    {"type": "salt_pepper", "prob": 0.08, "seed": 1234}
    {"type": "blur", "kernel": 5, "sigma": 1.5}
    {"type": "occlusion", "rects": [[x0, y0, x1, y1], ...], "coverage": 0.2}

``apply_corruption(img, params)`` is deterministic: the same params always give
the same corrupted image. This is what makes the validation/test manifests
reproducible (the salt-and-pepper pixel pattern is regenerated from its seed,
the rectangle coordinates are stored explicitly).
"""

from __future__ import annotations

import numpy as np

from .constants import (
    CLASS_NAMES,
    OCC_COVERAGE_TOLERANCE,
    TEST_LEVELS,
    TRAIN_BLUR_KERNELS,
    TRAIN_BLUR_SIGMA_RANGE,
    TRAIN_OCC_COVERAGE_RANGE,
    TRAIN_OCC_NUM_RECTS,
    TRAIN_SALT_PROB_RANGE,
)


# ----------------------------------------------------------------------------
# Individual corruptions
# ----------------------------------------------------------------------------
def salt_and_pepper(img: np.ndarray, prob: float, seed: int) -> np.ndarray:
    """Each pixel is selected with probability ``prob``; a selected pixel becomes
    black or white with equal probability (all three channels together, i.e. a
    pixel-level corruption as described in the PDF)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    selected = rng.random((h, w)) < prob
    white = rng.random((h, w)) < 0.5
    out = img.copy()
    out[selected & white] = 1.0
    out[selected & ~white] = 0.0
    return out


def gaussian_kernel1d(kernel: int, sigma: float) -> np.ndarray:
    x = np.arange(kernel, dtype=np.float64) - (kernel - 1) / 2.0
    g = np.exp(-(x**2) / (2.0 * sigma**2))
    return (g / g.sum()).astype(np.float32)


def gaussian_blur(img: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (no dark borders)."""
    if kernel % 2 != 1:
        raise ValueError("kernel size must be odd")
    g = gaussian_kernel1d(kernel, sigma)
    r = kernel // 2
    h, w = img.shape[:2]
    padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")
    tmp = np.zeros((h + 2 * r, w, img.shape[2]), dtype=np.float32)
    for i in range(kernel):  # horizontal pass
        tmp += g[i] * padded[:, i : i + w, :]
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(kernel):  # vertical pass
        out += g[i] * tmp[i : i + h, :, :]
    return np.clip(out, 0.0, 1.0)


def occlusion(img: np.ndarray, rects) -> np.ndarray:
    """Paint black rectangles. ``rects`` = [[x0, y0, x1, y1], ...] (x1/y1 exclusive)."""
    out = img.copy()
    for x0, y0, x1, y1 in rects:
        out[y0:y1, x0:x1, :] = 0.0
    return out


def union_coverage(rects, h: int, w: int) -> float:
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in rects:
        mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def sample_rectangles(
    rng: np.random.Generator,
    h: int,
    w: int,
    num_rects: int,
    target_coverage: float,
    tolerance: float = OCC_COVERAGE_TOLERANCE,
    bounds: tuple[float, float] | None = None,
    max_attempts: int = 200,
):
    """Sample ``num_rects`` axis-aligned rectangles at random positions whose
    UNION covers ``target_coverage`` of the image (within ``tolerance``).

    The target area is split randomly between the rectangles (Dirichlet), each
    rectangle gets a random aspect ratio in [0.5, 2] and a random position.
    Overlaps shrink the union, so we rejection-sample and keep the closest
    attempt if the tolerance is never met (very rare).
    Returns (rects, achieved_coverage).
    """
    total = h * w
    best, best_err = None, np.inf
    for _ in range(max_attempts):
        shares = rng.dirichlet(np.full(num_rects, 3.0)) if num_rects > 1 else np.ones(1)
        rects = []
        for share in shares:
            area = share * target_coverage * total
            aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))  # width / height
            rw = int(round(np.sqrt(area * aspect)))
            rh = int(round(area / max(rw, 1)))
            rw, rh = int(np.clip(rw, 4, w)), int(np.clip(rh, 4, h))
            x0 = int(rng.integers(0, w - rw + 1))
            y0 = int(rng.integers(0, h - rh + 1))
            rects.append([x0, y0, x0 + rw, y0 + rh])
        cov = union_coverage(rects, h, w)
        err = abs(cov - target_coverage)
        in_bounds = bounds is None or (bounds[0] <= cov <= bounds[1])
        if err <= tolerance and in_bounds:
            return rects, cov
        if in_bounds and err < best_err:
            best, best_err = (rects, cov), err
    if best is None:  # pragma: no cover - practically unreachable
        raise RuntimeError("could not sample occlusion rectangles")
    return best


# ----------------------------------------------------------------------------
# Parameter sampling
# ----------------------------------------------------------------------------
def sample_params(
    rng: np.random.Generator,
    corruption_type: str | None = None,
    h: int = 128,
    w: int = 128,
    allowed_types=None,
) -> dict:
    """Sample one TRAINING corruption (PDF p.3 ranges).

    If ``corruption_type`` is None the type is drawn uniformly from
    ``allowed_types`` (default: all four conditions, i.e. equal probability).
    """
    if corruption_type is None:
        types = list(allowed_types) if allowed_types else CLASS_NAMES
        corruption_type = types[int(rng.integers(len(types)))]

    if corruption_type == "clean":
        return {"type": "clean"}
    if corruption_type == "salt_pepper":
        return {
            "type": "salt_pepper",
            "prob": float(rng.uniform(*TRAIN_SALT_PROB_RANGE)),
            "seed": int(rng.integers(0, 2**31 - 1)),
        }
    if corruption_type == "blur":
        return {
            "type": "blur",
            "kernel": int(TRAIN_BLUR_KERNELS[int(rng.integers(len(TRAIN_BLUR_KERNELS)))]),
            "sigma": float(rng.uniform(*TRAIN_BLUR_SIGMA_RANGE)),
        }
    if corruption_type == "occlusion":
        n = int(TRAIN_OCC_NUM_RECTS[int(rng.integers(len(TRAIN_OCC_NUM_RECTS)))])
        target = float(rng.uniform(*TRAIN_OCC_COVERAGE_RANGE))
        rects, cov = sample_rectangles(rng, h, w, n, target, bounds=TRAIN_OCC_COVERAGE_RANGE)
        return {"type": "occlusion", "rects": rects, "coverage": cov, "num_rects": n}
    raise ValueError(f"unknown corruption type: {corruption_type}")


def level_params(
    corruption_type: str, level: str, seed: int, h: int = 128, w: int = 128
) -> dict:
    """Fixed FINAL-TEST severity level (PDF p.3). ``seed`` drives the salt
    pattern / rectangle positions so the record is reproducible."""
    if corruption_type == "clean":
        return {"type": "clean"}
    spec = TEST_LEVELS[corruption_type][level]
    if corruption_type == "salt_pepper":
        return {"type": "salt_pepper", "prob": spec["prob"], "seed": int(seed)}
    if corruption_type == "blur":
        return {"type": "blur", "kernel": spec["kernel"], "sigma": spec["sigma"]}
    if corruption_type == "occlusion":
        rng = np.random.default_rng(seed)
        rects, cov = sample_rectangles(rng, h, w, spec["num_rects"], spec["coverage"])
        return {
            "type": "occlusion",
            "rects": rects,
            "coverage": cov,
            "num_rects": spec["num_rects"],
        }
    raise ValueError(f"unknown corruption type: {corruption_type}")


def severity_value(params: dict) -> float:
    """A single scalar describing the severity (for logging/plots)."""
    t = params["type"]
    if t == "salt_pepper":
        return float(params["prob"])
    if t == "blur":
        return float(params["sigma"])
    if t == "occlusion":
        return float(params["coverage"])
    return 0.0


# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------
def apply_corruption(img: np.ndarray, params: dict) -> np.ndarray:
    t = params["type"]
    if t == "clean":
        return img.copy()
    if t == "salt_pepper":
        return salt_and_pepper(img, params["prob"], params["seed"])
    if t == "blur":
        return gaussian_blur(img, int(params["kernel"]), float(params["sigma"]))
    if t == "occlusion":
        return occlusion(img, params["rects"])
    raise ValueError(f"unknown corruption type: {t}")
