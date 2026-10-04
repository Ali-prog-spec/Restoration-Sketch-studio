"""Copy a few Oxford-IIIT Pet TEST images into backend/samples for the "clean sample" picker.

Images are CC BY-SA 4.0 (Parkhi et al., 2012). They are only used for the demo UI; the
test images are never used for training or model selection.

    python scripts/make_samples.py [--n 8]
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from src.data.oxford import load_split
from src.utils.config import resolve
from src.utils.image_io import load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    test = load_split()["test"]
    rng = np.random.default_rng(42)
    out = resolve("backend/samples")
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(rng.choice(test, size=args.n, replace=False)):
        img = load_rgb(resolve(f"data/raw/oxford_pets/images/{name}.jpg"))
        img.thumbnail((256, 256))
        img.save(out / f"{name}.jpg", quality=92)
        print(name)
    (out / "ATTRIBUTION.md").write_text(
        "Images from the Oxford-IIIT Pet Dataset (Parkhi, Vedaldi, Zisserman, Jawahar, CVPR 2012), "
        "https://www.robots.ox.ac.uk/~vgg/data/pets/ — licensed CC BY-SA 4.0. Resized to max 256 px.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
