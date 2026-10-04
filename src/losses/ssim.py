"""Differentiable SSIM (Wang et al., 2004).

Gaussian window 11x11, sigma 1.5, K1=0.01, K2=0.03, data range 1 — the settings
of the original paper. Computed per channel with a grouped convolution over the
'valid' region and averaged, which matches
``skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5,
use_sample_covariance=False, data_range=1, channel_axis=-1)`` (checked in tests).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_WINDOW_CACHE: dict = {}


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    key = (size, sigma, channels, device, dtype)
    if key not in _WINDOW_CACHE:
        x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
        g = torch.exp(-(x**2) / (2 * sigma**2))
        g = g / g.sum()
        w2d = (g[:, None] @ g[None, :]).to(device=device, dtype=dtype)
        _WINDOW_CACHE[key] = w2d.expand(channels, 1, size, size).contiguous()
    return _WINDOW_CACHE[key]


def ssim_map(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
             data_range: float = 1.0) -> torch.Tensor:
    """Return the SSIM map, shape (B, C, H-10, W-10)."""
    x = x.float()
    y = y.float()
    c = x.shape[1]
    w = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x**2
    syy = F.conv2d(y * y, w, groups=c) - mu_y**2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    return num / den


def ssim(x: torch.Tensor, y: torch.Tensor, reduction: str = "mean", **kw) -> torch.Tensor:
    """SSIM averaged over channels and pixels. ``reduction='none'`` gives one value per image."""
    m = ssim_map(x, y, **kw).mean(dim=(1, 2, 3))
    return m.mean() if reduction == "mean" else m
