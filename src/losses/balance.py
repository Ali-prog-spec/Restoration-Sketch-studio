"""Routing-balance regularisers for the soft mixture-of-experts (Task 3).

``squared`` (assignment baseline, PDF p.6):
    L = sum_k (w_bar_k - 1/K)^2, with w_bar_k the batch-mean weight of branch k.

    Note: for a dense softmax the batch-mean weights sum to 1, so their mean is
    exactly 1/K and Shazeer et al.'s (2017) importance loss CV(w_bar)^2 =
    Var(w_bar)/mean^2 = K * sum_k (w_bar_k - 1/K)^2 — i.e. the baseline is the
    importance loss up to a constant factor. We therefore do not implement CV^2
    separately.

``entropy`` (research-supported alternative):
    L = log K - H(w_bar), the gap between the maximum entropy and the entropy of
    the batch-mean routing distribution (marginal-entropy / "usage" balance).
    It is also minimised by uniform usage but penalises near-collapse much more
    strongly (its gradient grows like -log w_bar_k as a branch dies out),
    while — like the baseline — still allowing every *individual* sample to be
    routed sharply.

Both are computed on BALANCED batches (equal number of each true class), so the
uniform target 1/K is the correct expected usage.
"""

from __future__ import annotations

import math

import torch


def squared_balance(weights: torch.Tensor) -> torch.Tensor:
    k = weights.shape[1]
    w_bar = weights.float().mean(dim=0)
    return ((w_bar - 1.0 / k) ** 2).sum()


def entropy_balance(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    k = weights.shape[1]
    w_bar = weights.float().mean(dim=0)
    h = -(w_bar * torch.log(w_bar + eps)).sum()
    return math.log(k) - h


BALANCE_LOSSES = {"squared": squared_balance, "entropy": entropy_balance}


def routing_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample routing entropy (nats) — used for analysis, not as a loss."""
    w = weights.float()
    return -(w * torch.log(w + eps)).sum(dim=1)
