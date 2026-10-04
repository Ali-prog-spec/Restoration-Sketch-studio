"""Rebuild models from checkpoints (architecture config is stored inside each checkpoint)."""

from __future__ import annotations

import torch

from src.models.autoencoder import DenoisingAutoencoder
from src.models.classifier import CorruptionClassifier
from src.models.moe import SoftMoE
from src.utils.config import resolve

EXPERT_ORDER = ["salt_pepper", "blur", "occlusion"]


def _ck(path):
    return torch.load(resolve(path), map_location="cpu", weights_only=False)


def load_autoencoder(path) -> DenoisingAutoencoder:
    ck = _ck(path)
    m = DenoisingAutoencoder(**ck["model_config"])
    m.load_state_dict(ck["model_state"])
    return m.eval()


def load_classifier(path) -> CorruptionClassifier:
    ck = _ck(path)
    m = CorruptionClassifier(**ck["model_config"])
    m.load_state_dict(ck["model_state"])
    return m.eval()


def load_experts(checkpoint_dir) -> list[DenoisingAutoencoder]:
    return [load_autoencoder(f"{checkpoint_dir}/{c}_expert.pt") for c in EXPERT_ORDER]


def load_moe(path) -> SoftMoE:
    ck = _ck(path)
    cfgs = ck["model_config"]
    gate = CorruptionClassifier(**cfgs["gate"])
    experts = [DenoisingAutoencoder(**c) for c in cfgs["experts"]]
    moe = SoftMoE(gate, *experts, tau=cfgs["tau"])
    moe.load_state_dict(ck["model_state"])
    return moe.eval()
