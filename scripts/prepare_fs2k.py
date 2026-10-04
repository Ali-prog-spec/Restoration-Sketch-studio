"""Prepare FS2K for Task 4.

1. Download FS2K manually (Google Drive link in data/README.md) or with
   ``--download`` (needs ``pip install gdown``), and unzip so that
   data/raw/FS2K/anno_train.json exists.
2. Resolve every (photo, sketch) pair from the OFFICIAL anno_train.json / anno_test.json.
3. Split the official train portion into 85% train / 15% validation, stratified by
   sketch style, seed 42. The official test portion is kept separate and untouched.

Usage:  python scripts/prepare_fs2k.py [--root data/raw/FS2K] [--download]
"""

import _bootstrap  # noqa: F401

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

from src.corruptions.constants import SEED
from src.data.fs2k import SPLIT_FILE, pair_paths
from src.utils.config import resolve, save_json

GDRIVE_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"


def download(raw_parent: Path) -> None:
    import gdown  # optional dependency

    raw_parent.mkdir(parents=True, exist_ok=True)
    zpath = raw_parent / "FS2K.zip"
    if not zpath.exists():
        gdown.download(id=GDRIVE_ID, output=str(zpath), quiet=False)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw_parent)


def find_root(root: Path) -> Path:
    """Accept either .../FS2K or a parent folder that contains it."""
    if (root / "anno_train.json").exists():
        return root
    for cand in root.rglob("anno_train.json"):
        return cand.parent
    raise FileNotFoundError(f"anno_train.json not found under {root}")


def collect(root: Path, split: str) -> tuple[list[dict], list[dict]]:
    with open(root / f"anno_{split}.json", "r", encoding="utf-8") as f:
        anno = json.load(f)
    records, missing = [], []
    for a in anno:
        photo, sketch = pair_paths(root, a["image_name"])
        if photo is None or sketch is None:
            missing.append({"image_name": a["image_name"], "photo": str(photo), "sketch": str(sketch)})
            continue
        records.append({
            "image_name": a["image_name"],
            "photo": photo.relative_to(root).as_posix(),
            "sketch": sketch.relative_to(root).as_posix(),
            "style": int(a["style"]),
        })
    return records, missing


def describe(root: Path, records: list[dict], n: int = 30) -> dict:
    """Record image modes/sizes of a few files (used to choose sketch channels)."""
    modes, sizes = Counter(), Counter()
    for r in records[:n]:
        for key in ("photo", "sketch"):
            with Image.open(root / r[key]) as im:
                modes[f"{key}:{im.mode}"] += 1
                sizes[f"{key}:{im.size}"] += 1
    return {"modes": dict(modes), "sizes_sample": dict(sizes.most_common(10))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/FS2K")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    root = resolve(args.root)
    if args.download:
        download(root.parent)
    root = find_root(root)

    train_all, miss_tr = collect(root, "train")
    test, miss_te = collect(root, "test")
    styles = [r["style"] for r in train_all]
    train, val = train_test_split(train_all, test_size=args.val_fraction, random_state=args.seed,
                                  shuffle=True, stratify=styles)
    split = {
        "seed": args.seed,
        "root": root.relative_to(resolve(".")).as_posix() if root.is_relative_to(resolve(".")) else str(root),
        "val_fraction": args.val_fraction,
        "stratified_by": "style",
        "counts": {"official_train": len(train_all), "train": len(train), "val": len(val), "test": len(test)},
        "style_counts": {
            "train": dict(Counter(r["style"] for r in train)),
            "val": dict(Counter(r["style"] for r in val)),
            "test": dict(Counter(r["style"] for r in test)),
        },
        "missing_pairs": miss_tr + miss_te,
        "image_info": describe(root, train_all),
        "train": train,
        "val": val,
        "test": test,
    }
    save_json(split, SPLIT_FILE)
    print(json.dumps({k: split[k] for k in ("counts", "style_counts", "image_info")}, indent=2))
    if split["missing_pairs"]:
        print(f"WARNING: {len(split['missing_pairs'])} pairs could not be resolved (see {SPLIT_FILE})")


if __name__ == "__main__":
    main()
