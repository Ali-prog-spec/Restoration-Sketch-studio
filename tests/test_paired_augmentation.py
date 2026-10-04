"""Paired augmentation must keep photo/sketch pixel correspondence (PDF p.8)."""

import numpy as np
from PIL import Image

from src.data.paired_transforms import PairedTransform


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    base = (rng.random((200, 180)) * 255).astype(np.uint8)
    photo = Image.fromarray(np.stack([base] * 3, axis=2))  # RGB copy of the same content
    sketch = Image.fromarray(base)                          # grayscale
    return photo, sketch


def test_same_transform_applied_to_both():
    tf = PairedTransform(load_size=143, crop_size=128, flip_prob=0.5, max_rotation=10.0, train=True)
    rng = np.random.default_rng(42)
    photo, sketch = _pair()
    for _ in range(25):
        p_out, s_out, params = tf(photo, sketch, rng)
        assert p_out.size == s_out.size == (128, 128)
        p = np.asarray(p_out.convert("L"), dtype=np.int16)
        s = np.asarray(s_out, dtype=np.int16)
        # identical geometry -> identical pixels (up to rounding of RGB->L conversion)
        assert np.abs(p - s).max() <= 1, params


def test_independent_transforms_would_break_pairing():
    """Sanity check that the test above is meaningful."""
    tf = PairedTransform(143, 128, 0.5, 10.0, train=True)
    photo, sketch = _pair()
    p_out, _, _ = tf(photo, sketch, np.random.default_rng(1))
    _, s_out, _ = tf(photo, sketch, np.random.default_rng(2))
    diff = np.abs(np.asarray(p_out.convert("L"), np.int16) - np.asarray(s_out, np.int16))
    assert diff.mean() > 5


def test_parameters_vary_and_cover_flip():
    tf = PairedTransform(143, 128, 0.5, 0.0, train=True)
    rng = np.random.default_rng(0)
    ps = [tf.sample(rng) for _ in range(200)]
    assert 0.35 < np.mean([p.flip for p in ps]) < 0.65
    assert len({(p.crop_x, p.crop_y) for p in ps}) > 50
    assert all(0 <= p.crop_x <= 15 and 0 <= p.crop_y <= 15 for p in ps)


def test_eval_mode_is_deterministic_resize():
    tf = PairedTransform(143, 128, train=False)
    photo, sketch = _pair()
    a = tf(photo, sketch, np.random.default_rng(0))
    b = tf(photo, sketch, np.random.default_rng(99))
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])) and a[2] is None
