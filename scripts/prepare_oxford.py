"""Prepare the Oxford-IIIT Pet dataset for Tasks 1-3.

Steps
  1. (optional --download) download images.tar.gz + annotations.tar.gz from the official site
  2. extract them into data/raw/oxford_pets/
  3. read the OFFICIAL split files annotations/trainval.txt and annotations/test.txt
  4. convert every image to RGB, resize to 128x128, cache as uint8 .npy (clean images only)
  5. split trainval 80/20 with seed 42 -> data/manifests/oxford_split.json

Usage:  python scripts/prepare_oxford.py [--download]
"""

import _bootstrap  # noqa: F401

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.corruptions.constants import IMAGE_SIZE, SEED
from src.data.oxford import SPLIT_FILE, read_split_file
from src.utils.config import resolve, save_json
from src.utils.image_io import load_rgb, resize_rgb

URLS = {
    "images.tar.gz": "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz",
    "annotations.tar.gz": "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz",
}


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        target = raw / name
        if target.exists():
            print(f"[skip] {target} exists")
            continue
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, target)


def extract(raw: Path) -> None:
    for name, folder in [("images.tar.gz", "images"), ("annotations.tar.gz", "annotations")]:
        if (raw / folder).exists():
            continue
        archive = raw / name
        if not archive.exists():
            raise FileNotFoundError(f"{archive} not found — run with --download or place it there manually")
        print(f"extracting {archive}")
        with tarfile.open(archive) as tar:
            tar.extractall(raw, filter="data")


def build_cache(raw: Path, split: str, out: Path) -> list[str]:
    ids = read_split_file(raw / "annotations" / f"{split}.txt")
    arr = np.zeros((len(ids), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    kept, failed = [], []
    for name in tqdm(ids, desc=f"{split}"):
        try:
            img = load_rgb(raw / "images" / f"{name}.jpg")
            arr[len(kept)] = resize_rgb(img, IMAGE_SIZE)
            kept.append(name)
        except Exception as exc:  # corrupt / unreadable file: report, do not crash
            failed.append({"image_id": name, "error": repr(exc)})
    arr = arr[: len(kept)]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"oxford_{split}_128.npy", arr)
    with open(out / f"oxford_{split}_ids.json", "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"{split}: {len(kept)} images cached, {len(failed)} failed")
    return kept, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/oxford_pets")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    raw = resolve(args.raw)
    if args.download:
        download(raw)
    extract(raw)

    out = resolve("data/processed")
    trainval, fail_tv = build_cache(raw, "trainval", out)
    test, fail_te = build_cache(raw, "test", out)

    # 80/20 split of the development data, seed 42 (PDF p.2). Shared by Tasks 1-3.
    train_ids, val_ids = train_test_split(trainval, test_size=0.2, random_state=args.seed, shuffle=True)
    split = {
        "seed": args.seed,
        "source": "official annotations/trainval.txt (development) and annotations/test.txt (test)",
        "image_size": IMAGE_SIZE,
        "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test)},
        "failed_images": fail_tv + fail_te,
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": test,
    }
    save_json(split, SPLIT_FILE)
    print(f"split saved to {SPLIT_FILE}: {split['counts']}")


if __name__ == "__main__":
    main()
