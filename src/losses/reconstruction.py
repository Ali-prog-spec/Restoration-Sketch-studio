"""Reconstruction losses for Tasks 1-3."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ssim import ssim


def l1_ssim_loss(x_hat: torch.Tensor, x: torch.Tensor, alpha: float = 0.8):
    """L = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))   (PDF p.4).

    Returns (loss, parts) where parts holds detached L1 and SSIM for logging.
    """
    x_hat = x_hat.float()
    x = x.float()
    l1 = F.l1_loss(x_hat, x)
    s = ssim(x_hat, x)
    loss = alpha * l1 + (1.0 - alpha) * (1.0 - s)
    return loss, {"l1": l1.detach(), "ssim": s.detach()}


def selection_objective(l1: float, ssim_value: float) -> float:
    """Fixed model-selection objective J = 0.5 * L1 + 0.5 * (1 - SSIM) (lower is better).

    It is deliberately independent of the trainable loss weight alpha: if Optuna
    minimised the *training* loss, it could "improve" simply by changing alpha
    (e.g. alpha -> 1 makes the loss the small L1 term). Using one fixed criterion
    makes trials with different alpha comparable while still combining
    reconstruction quality (L1) and structural similarity (SSIM) as required.
    """
    return 0.5 * float(l1) + 0.5 * (1.0 - float(ssim_value))
