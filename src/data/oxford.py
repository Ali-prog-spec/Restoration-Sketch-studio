"""Oxford-IIIT Pet pipeline for Tasks 1-3.

Preparation (scripts/prepare_oxford.py) reads the OFFICIAL split files
(annotations/trainval.txt, annotations/test.txt), converts every image to RGB,
resizes it to 128x128 and caches the CLEAN images as one uint8 .npy array per split.
Caching clean resized images is only a speed-up (JPEG decoding dominates otherwise);
no corrupted image is ever written to disk.

The 80/20 train/validation split of trainval is made once with seed 42 and stored in
data/manifests/oxford_split.json, which Tasks 1, 2 and 3 all read.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.corruptions import CLASS_TO_INDEX, apply_corruption, sample_params, severity_value
from src.utils.config import resolve

PROCESSED = "data/processed"
SPLIT_FILE = "data/manifests/oxford_split.json"


def read_split_file(path) -> list[str]:
    """Official split file lines: 'Image CLASS-ID SPECIES BREED-ID'."""
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line.split()[0])
    return ids


def load_cache(split: str):
    """split in {'trainval', 'test'} -> (path of the uint8 [N,128,128,3] cache, ids list).
    The path is passed to the datasets, which open it lazily (see _LazyImages)."""
    with open(resolve(f"{PROCESSED}/oxford_{split}_ids.json"), "r", encoding="utf-8") as f:
        ids = json.load(f)
    return f"{PROCESSED}/oxford_{split}_128.npy", ids


def load_split():
    with open(resolve(SPLIT_FILE), "r", encoding="utf-8") as f:
        return json.load(f)


def to_tensor(img_float: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_float.transpose(2, 0, 1)))


class _LazyImages:
    """Holds either an in-memory array or the path of a cached .npy file.

    With a path, every DataLoader worker opens its own read-only memmap on first
    access (cheap to pickle on Windows' spawn start method)."""

    def __init__(self, images):
        self._path = images if isinstance(images, str) else None
        self._arr = None if self._path else images

    @property
    def images(self):
        if self._arr is None:
            self._arr = np.load(resolve(self._path), mmap_mode="r")
        return self._arr

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._path:
            state["_arr"] = None
        return state


def cache_path(split: str) -> str:
    return f"{PROCESSED}/oxford_{split}_128.npy"


class _Base(Dataset, _LazyImages):
    def __init__(self, images, all_ids, subset_ids, limit=None):
        _LazyImages.__init__(self, images)
        index_of = {name: i for i, name in enumerate(all_ids)}
        ids = list(subset_ids)[: limit or None]
        self.ids = ids
        self.rows = np.array([index_of[i] for i in ids], dtype=np.int64)

    def clean(self, i: int) -> np.ndarray:
        return np.asarray(self.images[self.rows[i]], dtype=np.float32) / 255.0


class RuntimeCorruptionDataset(_Base):
    """TRAINING dataset: a new corruption type + severity is sampled every time an
    image is loaded (PDF p.3). ``allowed_types`` restricts the types (the Task 2
    specialists see only their own corruption).

    ``__getitem__`` accepts an int (type sampled uniformly from allowed_types — equal
    probability) or a tuple (index, type_index) coming from BalancedBatchSampler.
    """

    def __init__(self, images, all_ids, subset_ids, allowed_types=None, limit=None, seed=42):
        super().__init__(images, all_ids, subset_ids, limit)
        self.allowed_types = list(allowed_types) if allowed_types else None
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:  # called from worker_init_fn
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        from src.corruptions import CLASS_NAMES

        if isinstance(item, (tuple, list)):
            i, t = int(item[0]), CLASS_NAMES[int(item[1])]
        else:
            i, t = int(item), None
        clean = self.clean(i)
        params = sample_params(self.rng, t, allowed_types=self.allowed_types)
        corrupted = apply_corruption(clean, params)
        return {
            "input": to_tensor(corrupted),
            "target": to_tensor(clean),
            "label": CLASS_TO_INDEX[params["type"]],
            "severity": severity_value(params),
            "params": json.dumps(params),
            "image_id": self.ids[i],
        }


class BalancedBatchSampler(Sampler):
    """Yields batches of (image_index, class_index) with exactly batch_size/4 samples
    of every class, in random order. Each image still receives a uniformly random
    class (so the per-image condition keeps equal probability), but every batch is
    perfectly balanced — required for the classifier (PDF p.4) and for the MoE
    balance loss, which is defined over a balanced batch (PDF p.6)."""

    def __init__(self, n_items: int, batch_size: int, num_classes: int = 4, seed: int = 42, drop_last: bool = True):
        if batch_size % num_classes:
            raise ValueError("batch_size must be a multiple of the number of classes")
        self.n, self.bs, self.k, self.seed, self.drop_last = n_items, batch_size, num_classes, seed, drop_last
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(self.n)
        for start in range(0, self.n, self.bs):
            idx = order[start : start + self.bs]
            if len(idx) < self.bs and self.drop_last:
                break
            labels = np.resize(np.arange(self.k), len(idx))
            rng.shuffle(labels)
            yield [(int(i), int(c)) for i, c in zip(idx, labels)]

    def __len__(self):
        return self.n // self.bs if self.drop_last else -(-self.n // self.bs)


class ManifestDataset(Dataset, _LazyImages):
    """VALIDATION/TEST dataset: corruption parameters come from a stored manifest,
    so the corrupted images are identical on every run."""

    def __init__(self, images, all_ids, manifest: dict, types=None, limit_images=None):
        _LazyImages.__init__(self, images)
        index_of = {name: i for i, name in enumerate(all_ids)}
        records = manifest["records"]
        if types is not None:
            records = [r for r in records if r["type"] in types]
        if limit_images:
            keep = set(list(dict.fromkeys(r["image_id"] for r in records))[:limit_images])
            records = [r for r in records if r["image_id"] in keep]
        self.records = records
        self.index_of = index_of

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        clean = np.asarray(self.images[self.index_of[r["image_id"]]], dtype=np.float32) / 255.0
        corrupted = apply_corruption(clean, r)
        return {
            "input": to_tensor(corrupted),
            "target": to_tensor(clean),
            "label": CLASS_TO_INDEX[r["type"]],
            "severity": severity_value(r),
            "level": r.get("level", "sampled"),
            "record": i,
            "image_id": r["image_id"],
        }


def load_manifest(name: str) -> dict:
    with open(resolve(f"data/manifests/oxford_{name}_manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)
