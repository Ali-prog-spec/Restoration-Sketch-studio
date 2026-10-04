"""Upload validation, preprocessing, encoding and light-weight metrics (NumPy only)."""

from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.utils.image_io import load_rgb, png_bytes, resize_rgb, to_float, to_uint8

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "MPO"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/jpg", "application/octet-stream"}
MIN_SIDE, MAX_SIDE = 16, 8000


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"unsupported content type '{file.content_type}' (use JPEG, PNG, WEBP or BMP)")
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"file too large (limit {max_bytes // (1024 * 1024)} MB)")
    if not data:
        raise HTTPException(400, "empty file")
    return data


def decode_image(data: bytes) -> Image.Image:
    """Decode and validate by CONTENT (not by file extension / declared type)."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(400, f"not a valid image: {exc}") from exc
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, f"unsupported image format {fmt}")
    img = load_rgb(data)
    w, h = img.size
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE:
        raise HTTPException(400, f"image size {w}x{h} outside the supported range [{MIN_SIDE}, {MAX_SIDE}] px")
    return img


def preprocess_restoration(img: Image.Image) -> np.ndarray:
    """RGB -> 128x128 -> float32 HWC in [0, 1] (identical to the training pipeline)."""
    return to_float(resize_rgb(img, 128))


def to_nchw(img_hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img_hwc.transpose(2, 0, 1)[None], dtype=np.float32)


def from_nchw(t: np.ndarray) -> np.ndarray:
    return t[0].transpose(1, 2, 0)


def data_url(img: np.ndarray) -> str:
    """float [0,1] HWC / HW or uint8 -> 'data:image/png;base64,...'."""
    u8 = img if img.dtype == np.uint8 else to_uint8(img)
    if u8.ndim == 3 and u8.shape[2] == 1:
        u8 = u8[:, :, 0]
    return "data:image/png;base64," + base64.b64encode(png_bytes(u8)).decode("ascii")


# ---------------------------------------------------------------------------
# metrics (only when a clean reference exists, i.e. the user corrupted a clean image)
# ---------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


def _gauss_valid(img: np.ndarray, g: np.ndarray) -> np.ndarray:
    k = len(g)
    h, w = img.shape[:2]
    tmp = sum(g[i] * img[:, i : w - k + 1 + i] for i in range(k))
    return sum(g[i] * tmp[i : h - k + 1 + i] for i in range(k))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Same definition as the training metric (Gaussian 11, sigma 1.5, valid region)."""
    x = np.arange(11) - 5
    g = np.exp(-(x**2) / (2 * 1.5**2))
    g = g / g.sum()
    c1, c2 = 0.01**2, 0.03**2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = _gauss_valid(a, g), _gauss_valid(b, g)
    saa = _gauss_valid(a * a, g) - mu_a**2
    sbb = _gauss_valid(b * b, g) - mu_b**2
    sab = _gauss_valid(a * b, g) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2))
    return float(m.mean())
