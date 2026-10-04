"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    ``deterministic=True`` additionally forces deterministic cuDNN kernels
    (slower). We keep cudnn.benchmark off by default so repeated runs are close;
    bit-exact GPU reproducibility is not guaranteed by PyTorch in general
    (see https://pytorch.org/docs/stable/notes/randomness.html).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own reproducible NumPy generator.

    ``torch.initial_seed()`` inside a worker equals base_seed + worker_id, and
    base_seed is drawn from the main process RNG (seeded by ``set_seed``), so the
    runtime corruption stream is reproducible for a fixed seed and worker count.
    """
    info = torch.utils.data.get_worker_info()
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
