"""Zip models/*.onnx + model_card.json into dist/models.zip for a GitHub Release upload.

    python scripts/package_models.py
Then create a release (e.g. v1.0) on GitHub and attach dist/models.zip (and dist/models.sha256).
"""

import _bootstrap  # noqa: F401

import hashlib
import zipfile

from src.utils.config import resolve


def main():
    src = resolve("models")
    files = sorted(src.glob("*.onnx")) + [src / "model_card.json"]
    missing = [f.name for f in files if not f.exists()]
    if missing or len(files) < 8:
        raise SystemExit(f"models incomplete (missing {missing}); run scripts/export_onnx.py --mode final first")
    dist = resolve("dist")
    dist.mkdir(exist_ok=True)
    zpath = dist / "models.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (dist / "models.sha256").write_text(f"{digest}  models.zip\n")
    print(f"{zpath} ({zpath.stat().st_size / 1e6:.1f} MB) sha256={digest}")


if __name__ == "__main__":
    main()
