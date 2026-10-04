"""FS2K dataset for Task 4 (Fan et al., 2022).

Official layout (github.com/DengPingFan/FS2K):
    FS2K/photo/photo{1,2,3}/image*.{jpg,png}
    FS2K/sketch/sketch{1,2,3}/sketch*.{jpg,png}
    FS2K/anno_train.json, FS2K/anno_test.json   (official train/test definition)

Each annotation entry has "image_name" like "photo1/image0110" (no extension) and
"style" in {0, 1, 2}. The sketch path is obtained, as in the official
tools/split_train_test.py, by replacing 'photo' -> 'sketch' and 'image' -> 'sketch'
in the RELATIVE name; the extension is resolved per file (.jpg first, then .png).

scripts/prepare_fs2k.py resolves every pair, splits the official TRAIN portion
85/15 stratified by style with seed 42, and writes data/manifests/fs2k_split.json.
The official TEST portion is only used by the final evaluation script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.paired_transforms import PairedTransform
from src.utils.config import resolve

SPLIT_FILE = "data/manifests/fs2k_split.json"
EXTENSIONS = (".jpg", ".png", ".jpeg", ".JPG", ".PNG")


def resolve_file(root: Path, rel_stem: str) -> Path | None:
    for ext in EXTENSIONS:
        p = root / (rel_stem + ext)
        if p.exists():
            return p
    return None


def pair_paths(root: Path, image_name: str):
    photo = resolve_file(root / "photo", image_name)
    sketch_rel = image_name.replace("photo", "sketch").replace("image", "sketch")
    sketch = resolve_file(root / "sketch", sketch_rel)
    return photo, sketch


def load_split() -> dict:
    with open(resolve(SPLIT_FILE), "r", encoding="utf-8") as f:
        return json.load(f)


def load_photo(path) -> Image.Image:
    from src.utils.image_io import load_rgb

    return load_rgb(path)


def load_sketch(path, channels: int) -> Image.Image:
    from src.utils.image_io import load_rgb

    img = load_rgb(path)  # handles RGBA / palette sketches
    return img.convert("L") if channels == 1 else img


def to_tensor_pm1(img: Image.Image) -> torch.Tensor:
    """PIL -> float tensor in [-1, 1], CHW."""
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    if a.ndim == 2:
        a = a[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


class FS2KPairs(Dataset):
    def __init__(self, records: list[dict], root: str, sketch_channels: int = 1, train: bool = False,
                 load_size: int = 143, crop_size: int = 128, max_rotation: float = 0.0, seed: int = 42,
                 limit: int | None = None):
        self.records = records[: limit or None]
        self.root = Path(resolve(root))
        self.sketch_channels = sketch_channels
        self.tf = PairedTransform(load_size, crop_size, max_rotation=max_rotation, train=train)
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        photo = load_photo(self.root / r["photo"])
        sketch = load_sketch(self.root / r["sketch"], self.sketch_channels)
        photo, sketch, _ = self.tf(photo, sketch, self.rng)
        return {
            "photo": to_tensor_pm1(photo),
            "sketch": to_tensor_pm1(sketch),
            "style": int(r["style"]),
            "name": r["image_name"],
        }
