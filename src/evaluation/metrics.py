"""Image-restoration metrics (per image).

* L1 / MAE — named in the PDF (reconstruction term); mean absolute error in [0, 1] units.
* SSIM — named in the PDF; Wang et al. (2004), Gaussian window 11, sigma 1.5.
* PSNR — additional standard restoration metric: 10 log10(1 / MSE) for data range 1.
  Reported because it is the most common restoration benchmark number and is
  complementary to SSIM (Hore & Ziou, 2010): PSNR measures pixel error energy,
  SSIM measures local structure.
"""

from __future__ import annotations

import torch

from src.losses.ssim import ssim


@torch.no_grad()
def batch_metrics(x_hat: torch.Tensor, x: torch.Tensor) -> dict:
    """Return per-image tensors (B,) of l1, mse, psnr, ssim for images in [0, 1]."""
    x_hat = x_hat.float().clamp(0, 1)
    x = x.float()
    l1 = (x_hat - x).abs().mean(dim=(1, 2, 3))
    mse = ((x_hat - x) ** 2).mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    s = ssim(x_hat, x, reduction="none")
    return {"l1": l1, "mse": mse, "psnr": psnr, "ssim": s}
