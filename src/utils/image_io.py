"""Image loading/preprocessing shared by training and the backend (PIL + NumPy only)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

# Bicubic with PIL's built-in antialiasing when downsampling.
RESAMPLE = Image.BICUBIC


def load_rgb(path_or_bytes) -> Image.Image:
    """Open an image from a path or raw bytes, honour EXIF rotation, convert to RGB.
    Handles grayscale, CMYK, palette and RGBA inputs (RGBA is composited on white)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(path_or_bytes))
    else:
        img = Image.open(path_or_bytes)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def resize_rgb(img: Image.Image, size: int = 128) -> np.ndarray:
    """Resize (no crop, as specified by the PDF) and return uint8 HxWx3."""
    return np.asarray(img.resize((size, size), RESAMPLE), dtype=np.uint8)


def to_float(img_uint8: np.ndarray) -> np.ndarray:
    return img_uint8.astype(np.float32) / 255.0


def to_uint8(img_float: np.ndarray) -> np.ndarray:
    return (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def png_bytes(img_uint8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")  # mode inferred: L or RGB
    return buf.getvalue()
