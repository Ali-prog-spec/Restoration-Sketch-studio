"""Helpers shared by all training loops."""

from __future__ import annotations

import csv
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.oxford import BalancedBatchSampler, ManifestDataset, RuntimeCorruptionDataset, load_cache, load_manifest, load_split
from src.utils.config import resolve
from src.utils.seed import make_generator, worker_init_fn


# ----------------------------------------------------------------------------
# Data loaders for Tasks 1-3
# ----------------------------------------------------------------------------
def oxford_loaders(cfg: dict, batch_size: int, allowed_types=None, balanced: bool = True,
                   val_types=None):
    """Training loader (runtime corruptions) and validation loader (manifest)."""
    dcfg = cfg.get("data", {})
    limit = dcfg.get("limit_data")
    nw = int(dcfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 42))
    split = load_split()
    images, ids = load_cache("trainval")

    train_ds = RuntimeCorruptionDataset(images, ids, split["train"], allowed_types=allowed_types,
                                        limit=limit, seed=seed)
    common = dict(num_workers=nw, worker_init_fn=worker_init_fn, pin_memory=torch.cuda.is_available(),
                  persistent_workers=nw > 0)
    if balanced and not allowed_types:
        sampler = BalancedBatchSampler(len(train_ds), batch_size, seed=seed)
        train_dl = DataLoader(train_ds, batch_sampler=sampler, generator=make_generator(seed), **common)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=make_generator(seed), **common)

    val_ds = ManifestDataset(images, ids, load_manifest("val"), types=val_types, limit_images=limit)
    val_dl = DataLoader(val_ds, batch_size=max(batch_size, 64), shuffle=False, **common)
    return train_dl, val_dl


def test_loader(cfg: dict, batch_size: int = 128, types=None):
    dcfg = cfg.get("data", {})
    images, ids = load_cache("test")
    ds = ManifestDataset(images, ids, load_manifest("test"), types=types, limit_images=dcfg.get("limit_data"))
    nw = int(dcfg.get("num_workers", 0))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)


# ----------------------------------------------------------------------------
# Mixed precision
# ----------------------------------------------------------------------------
def autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def grad_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


# ----------------------------------------------------------------------------
# Optimiser / scheduler
# ----------------------------------------------------------------------------
def make_scheduler(opt, cfg_train: dict, epochs: int):
    kind = cfg_train.get("scheduler", "cosine")
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=opt.param_groups[0]["lr"] * 0.01)
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    return None


def step_scheduler(sched, metric=None):
    if sched is None:
        return
    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        sched.step(metric)
    else:
        sched.step()


# ----------------------------------------------------------------------------
# Checkpoints / history
# ----------------------------------------------------------------------------
def save_checkpoint(path, model: torch.nn.Module, extra: dict) -> Path:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": getattr(model, "config", None), **extra}, path)
    return path


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(resolve(path), map_location=map_location, weights_only=False)


class History:
    """Accumulates one row per epoch and writes it as CSV."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, row: dict) -> None:
        self.rows.append({k: (float(v) if isinstance(v, (int, float, torch.Tensor)) else v) for k, v in row.items()})

    def save(self, path) -> Path:
        path = resolve(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        return path


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


class EarlyStopping:
    def __init__(self, patience: int | None):
        self.patience = patience
        self.best = float("inf")
        self.bad = 0

    def step(self, value: float) -> tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        if value < self.best - 1e-6:
            self.best, self.bad = value, 0
            return True, False
        self.bad += 1
        return False, bool(self.patience and self.bad >= self.patience)
