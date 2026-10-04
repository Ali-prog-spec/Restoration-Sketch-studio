"""Alternative download path for Oxford-IIIT Pet via the timm Hugging Face mirror
(https://huggingface.co/datasets/timm/oxford-iiit-pet, CC BY-SA 4.0 like the original).

Why: the official server (thor.robots.ox.ac.uk) throttles each connection to ~150 KB/s
and does not support HTTP range requests, so the 792 MB archive can take >1.5 h.

The mirror stores the ORIGINAL image bytes plus an ``image_id`` field. This script
  1. writes every image to data/raw/oxford_pets/images/<image_id>.jpg (bytes unchanged)
  2. VERIFIES the mirror against the official source:
       a. the image ids of the mirror's train/test splits equal the official
          annotations/trainval.txt and annotations/test.txt (official annotations.tar.gz)
       b. every image that can be read from a (possibly partial) official
          images.tar.gz is byte-identical (MD5) to the mirror copy
  and writes the verification result to data/manifests/oxford_source_verification.json.

Usage:
  curl -L -o data/raw/oxford_pets/hf/train.parquet https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/main/data/train-00000-of-00001.parquet
  curl -L -o data/raw/oxford_pets/hf/test.parquet  https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/main/data/test-00000-of-00001.parquet
  python scripts/fetch_oxford_mirror.py [--official-archive data/raw/oxford_pets/official_images_partial.tar.gz]
"""

import _bootstrap  # noqa: F401

import argparse
import hashlib
import tarfile
import zlib
from pathlib import Path

import pyarrow.parquet as pq

from src.data.oxford import read_split_file
from src.utils.config import resolve, save_json


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/oxford_pets")
    ap.add_argument("--official-archive", default="data/raw/oxford_pets/official_images_partial.tar.gz")
    args = ap.parse_args()
    raw = resolve(args.raw)
    out = raw / "images"
    out.mkdir(parents=True, exist_ok=True)

    hashes, split_ids = {}, {}
    for split in ("train", "test"):
        table = pq.read_table(raw / "hf" / f"{split}.parquet", columns=["image", "image_id"])
        ids = table.column("image_id").to_pylist()
        imgs = table.column("image").to_pylist()
        split_ids[split] = ids
        for image_id, img in zip(ids, imgs):
            data = img["bytes"]
            (out / f"{image_id}.jpg").write_bytes(data)
            hashes[image_id] = md5(data)
        print(f"{split}: wrote {len(ids)} images")

    official_tv = read_split_file(raw / "annotations" / "trainval.txt")
    official_te = read_split_file(raw / "annotations" / "test.txt")
    ids_ok = {"train_equals_trainval_txt": sorted(split_ids["train"]) == sorted(official_tv),
              "test_equals_test_txt": sorted(split_ids["test"]) == sorted(official_te),
              "n_trainval_txt": len(official_tv), "n_test_txt": len(official_te),
              "n_mirror_train": len(split_ids["train"]), "n_mirror_test": len(split_ids["test"])}

    # byte-level comparison against whatever part of the official archive is available
    compared, identical, mismatched = 0, 0, []
    arch = resolve(args.official_archive)
    if arch.exists():
        try:
            with tarfile.open(arch, mode="r|gz") as tar:
                for m in tar:
                    if not m.isfile() or not m.name.endswith(".jpg"):
                        continue
                    data = tar.extractfile(m).read()
                    image_id = Path(m.name).stem
                    if image_id in hashes:
                        compared += 1
                        if md5(data) == hashes[image_id]:
                            identical += 1
                        else:
                            mismatched.append(image_id)
        except (EOFError, tarfile.ReadError, zlib.error, OSError) as exc:
            print(f"(official archive ends early — partial download: {type(exc).__name__})")
    result = {"mirror": "https://huggingface.co/datasets/timm/oxford-iiit-pet", "split_ids": ids_ok,
              "official_archive": str(args.official_archive), "bytes_compared": compared,
              "bytes_identical": identical, "bytes_mismatched": mismatched[:50],
              "verified": ids_ok["train_equals_trainval_txt"] and ids_ok["test_equals_test_txt"]
              and compared > 0 and not mismatched}
    save_json(result, "data/manifests/oxford_source_verification.json")
    print(result)


if __name__ == "__main__":
    main()
