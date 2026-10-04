import math

import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

from src.losses import entropy_balance, l1_ssim_loss, selection_objective, squared_balance, ssim
from src.evaluation.metrics import batch_metrics


def test_ssim_matches_skimage():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.1, a.shape), 0, 1).astype(np.float32)
    ref = structural_similarity(a, b, channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
    ours = ssim(torch.from_numpy(a.transpose(2, 0, 1))[None], torch.from_numpy(b.transpose(2, 0, 1))[None])
    assert abs(float(ours) - ref) < 2e-3, (float(ours), ref)


def test_ssim_identity_and_loss_zero():
    x = torch.rand(2, 3, 64, 64)
    assert abs(float(ssim(x, x)) - 1) < 1e-5
    loss, parts = l1_ssim_loss(x, x, alpha=0.8)
    assert float(loss) < 1e-5 and float(parts["l1"]) == 0


def test_l1_ssim_weighting():
    x = torch.rand(1, 3, 64, 64)
    y = (x + 0.1).clamp(0, 1)
    l_a1, p = l1_ssim_loss(y, x, alpha=1.0)
    l_a0, _ = l1_ssim_loss(y, x, alpha=0.0)
    assert torch.isclose(l_a1, p["l1"]) and torch.isclose(l_a0, 1 - p["ssim"])


def test_loss_is_differentiable():
    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64, requires_grad=True)
    l1_ssim_loss(y, x)[0].backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()


def test_balance_losses():
    uniform = torch.full((8, 4), 0.25)
    collapsed = torch.tensor([[1.0, 0, 0, 0]] * 8)
    assert float(squared_balance(uniform)) < 1e-9 and float(entropy_balance(uniform)) < 1e-6
    assert abs(float(squared_balance(collapsed)) - 0.75) < 1e-6
    assert abs(float(entropy_balance(collapsed)) - math.log(4)) < 1e-4
    # per-sample one-hot but balanced over the batch -> no penalty (specialisation allowed)
    onehot_balanced = torch.eye(4).repeat(2, 1)
    assert float(squared_balance(onehot_balanced)) < 1e-9


def test_squared_balance_equals_scaled_cv2():
    w = torch.softmax(torch.randn(16, 4), dim=1)
    w_bar = w.mean(0)
    cv2 = w_bar.var(unbiased=False) / w_bar.mean() ** 2
    assert torch.isclose(cv2, 4 * squared_balance(w), atol=1e-6)


def test_metrics_and_objective():
    x = torch.rand(2, 3, 64, 64)
    m = batch_metrics(x, x)
    assert torch.all(m["l1"] == 0) and torch.all(m["psnr"] > 90)
    assert selection_objective(0.0, 1.0) == 0.0 and selection_objective(0.1, 0.8) == pytest.approx(0.15)
