"""Corruption definitions must match the PDF (p.3) exactly."""

import numpy as np
import pytest
from scipy.ndimage import convolve1d

from src.corruptions import CLASS_NAMES, apply_corruption, gaussian_blur, level_params, sample_params, union_coverage
from src.corruptions.constants import TEST_LEVELS
from src.corruptions.corruptions import gaussian_kernel1d


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.uniform(0.2, 0.8, size=(128, 128, 3)).astype(np.float32)  # no pure black/white pixels


def test_training_types_equal_probability():
    rng = np.random.default_rng(42)
    counts = {c: 0 for c in CLASS_NAMES}
    n = 20000
    for _ in range(n):
        counts[sample_params(rng)["type"]] += 1
    for c in CLASS_NAMES:
        assert abs(counts[c] / n - 0.25) < 0.015, counts


def test_training_parameter_ranges():
    rng = np.random.default_rng(1)
    for _ in range(3000):
        p = sample_params(rng)
        if p["type"] == "salt_pepper":
            assert 0.02 <= p["prob"] <= 0.15
        elif p["type"] == "blur":
            assert p["kernel"] in (3, 5, 7) and 0.5 <= p["sigma"] <= 2.5
        elif p["type"] == "occlusion":
            assert 1 <= len(p["rects"]) <= 3 and p["num_rects"] == len(p["rects"])
            cov = union_coverage(p["rects"], 128, 128)
            assert 0.10 <= cov <= 0.35, cov
            assert abs(cov - p["coverage"]) < 1e-9


def test_salt_and_pepper_pixels(img):
    p = {"type": "salt_pepper", "prob": 0.1, "seed": 7}
    out = apply_corruption(img, p)
    changed = np.any(out != img, axis=2)
    assert abs(changed.mean() - 0.1) < 0.01
    vals = out[changed]
    # every corrupted pixel is black or white on ALL channels, ~half each
    assert np.all((vals == 0).all(1) | (vals == 1).all(1))
    white = (vals == 1).all(1).mean()
    assert 0.45 < white < 0.55


def test_gaussian_blur_matches_scipy(img):
    for k, s in [(3, 0.7), (5, 1.5), (7, 2.5)]:
        g = gaussian_kernel1d(k, s)
        ref = convolve1d(convolve1d(img, g, axis=1, mode="mirror"), g, axis=0, mode="mirror")
        np.testing.assert_allclose(gaussian_blur(img, k, s), np.clip(ref, 0, 1), atol=1e-5)
        assert abs(g.sum() - 1) < 1e-6 and len(g) == k


def test_occlusion_black_rectangles(img):
    p = level_params("occlusion", "medium", seed=3)
    out = apply_corruption(img, p)
    mask = np.zeros((128, 128), bool)
    for x0, y0, x1, y1 in p["rects"]:
        mask[y0:y1, x0:x1] = True
    assert np.all(out[mask] == 0) and np.all(out[~mask] == img[~mask])


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_fixed_test_levels(level):
    assert level_params("salt_pepper", level, 1)["prob"] == TEST_LEVELS["salt_pepper"][level]["prob"]
    b = level_params("blur", level, 1)
    assert (b["kernel"], b["sigma"]) == (TEST_LEVELS["blur"][level]["kernel"], TEST_LEVELS["blur"][level]["sigma"])
    target = TEST_LEVELS["occlusion"][level]
    for seed in range(50):
        o = level_params("occlusion", level, seed)
        assert len(o["rects"]) == target["num_rects"]
        assert abs(o["coverage"] - target["coverage"]) <= 0.015 + 1e-9


def test_test_level_values_are_the_pdf_values():
    assert [TEST_LEVELS["salt_pepper"][l]["prob"] for l in ("low", "medium", "high")] == [0.03, 0.08, 0.15]
    assert [(TEST_LEVELS["blur"][l]["kernel"], TEST_LEVELS["blur"][l]["sigma"]) for l in ("low", "medium", "high")] == \
        [(3, 0.7), (5, 1.5), (7, 2.5)]
    assert [(TEST_LEVELS["occlusion"][l]["coverage"], TEST_LEVELS["occlusion"][l]["num_rects"])
            for l in ("low", "medium", "high")] == [(0.10, 1), (0.20, 2), (0.35, 3)]


def test_same_params_same_image(img):
    rng = np.random.default_rng(5)
    for _ in range(40):
        p = sample_params(rng)
        np.testing.assert_array_equal(apply_corruption(img, p), apply_corruption(img, p))


def test_clean_is_identity(img):
    np.testing.assert_array_equal(apply_corruption(img, {"type": "clean"}), img)
