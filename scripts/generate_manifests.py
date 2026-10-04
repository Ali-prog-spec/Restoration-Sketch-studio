"""Generate the deterministic validation and test corruption manifests ONCE.

    python scripts/generate_manifests.py            # refuses to overwrite existing manifests
    python scripts/generate_manifests.py --force    # regenerate (changes the benchmark!)

Outputs data/manifests/oxford_val_manifest.json and oxford_test_manifest.json,
plus a small CSV summary of the realised severities.
"""

import _bootstrap  # noqa: F401

import argparse
from collections import Counter

import pandas as pd

from src.corruptions.constants import SEED
from src.data.manifests import build_test_manifest, build_validation_manifest
from src.data.oxford import load_split
from src.utils.config import resolve, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    split = load_split()
    paths = {
        "val": resolve("data/manifests/oxford_val_manifest.json"),
        "test": resolve("data/manifests/oxford_test_manifest.json"),
    }
    if not args.force and any(p.exists() for p in paths.values()):
        raise SystemExit("manifests already exist — they must be generated once. Use --force to regenerate.")

    val = build_validation_manifest(split["val"], seed=args.seed)
    test = build_test_manifest(split["test"], seed=args.seed)
    save_json(val, paths["val"])
    save_json(test, paths["test"])

    for name, m in [("val", val), ("test", test)]:
        df = pd.DataFrame(m["records"])
        print(f"{name}: {len(df)} records, types={dict(Counter(df['type']))}")
        summary = []
        for (t, lvl), g in df.groupby(["type", "level"]):
            row = {"split": name, "type": t, "level": lvl, "n": len(g)}
            if "coverage" in g:
                row["coverage_mean"] = g["coverage"].mean()
                row["coverage_min"] = g["coverage"].min()
                row["coverage_max"] = g["coverage"].max()
            if "prob" in g:
                row["prob_mean"] = g["prob"].mean()
            if "sigma" in g:
                row["sigma_mean"] = g["sigma"].mean()
            summary.append(row)
        out = resolve(f"outputs/tables/manifest_summary_{name}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary).to_csv(out, index=False)
        print(f"  summary -> {out}")


if __name__ == "__main__":
    main()
