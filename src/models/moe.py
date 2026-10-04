"""Soft mixture-of-experts restoration (Task 3) and hard routing (Task 2).

Branch order is fixed: 0 identity (clean), 1 salt-and-pepper, 2 blur, 3 occlusion,
which is exactly the classifier's class order — so the Task 2 classifier can be
used directly as the gate G.

    w  = softmax(G(x~) / tau)
    x^ = w0 * x~ + w1 * A_salt(x~) + w2 * A_blur(x~) + w3 * A_occ(x~)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SoftMoE(nn.Module):
    def __init__(self, gate: nn.Module, expert_salt: nn.Module, expert_blur: nn.Module,
                 expert_occ: nn.Module, tau: float = 1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList([expert_salt, expert_blur, expert_occ])
        # tau is a fixed hyperparameter (tuned by Optuna), stored as a buffer so it is
        # saved in checkpoints and baked into the ONNX graph.
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        weights = torch.softmax(logits / self.tau, dim=1)                 # (B, 4)
        branches = torch.stack([x] + [e(x) for e in self.experts], dim=1)  # (B, 4, C, H, W)
        out = (weights[:, :, None, None, None] * branches).sum(dim=1)
        return out, weights, logits

    def set_experts_trainable(self, trainable: bool) -> None:
        for p in self.experts.parameters():
            p.requires_grad_(trainable)


class SoftMoEExport(nn.Module):
    """ONNX wrapper: returns (restored image, routing weights) only."""

    def __init__(self, moe: SoftMoE):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        out, weights, _ = self.moe(x)
        return out, weights


@torch.no_grad()
def hard_route(x: torch.Tensor, route: torch.Tensor, experts) -> torch.Tensor:
    """Hard routing (PDF p.5): route 0 -> identity bypass (no expert is run),
    1/2/3 -> the corresponding specialist. ``route`` is (B,) long: either the
    oracle label from the manifest or the classifier argmax."""
    out = x.clone()
    for k, expert in enumerate(experts, start=1):
        idx = (route == k).nonzero(as_tuple=True)[0]
        if idx.numel():
            out[idx] = expert(x[idx]).to(out.dtype)
    return out
