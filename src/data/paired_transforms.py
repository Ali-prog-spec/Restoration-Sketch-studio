"""Synchronised (paired) augmentation for photo/sketch pairs (PDF p.8).

The random parameters are sampled ONCE per pair and applied to both images, so
pixel (i, j) of the photo still corresponds to pixel (i, j) of the sketch.

Augmentations (pix2pix-style "jitter", Isola et al. 2017):
  * resize both to `load_size` (e.g. 143) then random-crop `crop_size` (128) at the SAME offset
  * horizontal flip of BOTH with probability 0.5
  * optional small rotation of BOTH by the same angle
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class PairParams:
    crop_x: int
    crop_y: int
    flip: bool
    angle: float


class PairedTransform:
    def __init__(self, load_size: int = 143, crop_size: int = 128, flip_prob: float = 0.5,
                 max_rotation: float = 0.0, train: bool = True):
        self.load_size, self.crop_size = load_size, crop_size
        self.flip_prob, self.max_rotation, self.train = flip_prob, max_rotation, train

    def sample(self, rng: np.random.Generator) -> PairParams:
        m = self.load_size - self.crop_size
        return PairParams(
            crop_x=int(rng.integers(0, m + 1)) if m > 0 else 0,
            crop_y=int(rng.integers(0, m + 1)) if m > 0 else 0,
            flip=bool(rng.random() < self.flip_prob),
            angle=float(rng.uniform(-self.max_rotation, self.max_rotation)) if self.max_rotation > 0 else 0.0,
        )

    def apply(self, img: Image.Image, p: PairParams, fill) -> Image.Image:
        img = img.resize((self.load_size, self.load_size), Image.BICUBIC)
        if p.angle:
            img = img.rotate(p.angle, resample=Image.BILINEAR, fillcolor=fill)
        img = img.crop((p.crop_x, p.crop_y, p.crop_x + self.crop_size, p.crop_y + self.crop_size))
        if p.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __call__(self, photo: Image.Image, sketch: Image.Image, rng: np.random.Generator):
        if not self.train:  # validation / test: deterministic plain resize
            size = (self.crop_size, self.crop_size)
            return photo.resize(size, Image.BICUBIC), sketch.resize(size, Image.BICUBIC), None
        p = self.sample(rng)
        white_photo = (255,) * len(photo.getbands())
        white_sketch = (255,) * len(sketch.getbands()) if len(sketch.getbands()) > 1 else 255
        return self.apply(photo, p, white_photo), self.apply(sketch, p, white_sketch), p
