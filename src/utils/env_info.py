"""Record the software/hardware environment of an experiment."""

from __future__ import annotations

import platform
import sys
from importlib import metadata

PACKAGES = [
    "torch", "torchvision", "numpy", "optuna", "mlflow", "onnx", "onnxruntime",
    "scikit-learn", "scikit-image", "pillow", "fastapi",
]


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {},
    }
    for p in PACKAGES:
        try:
            info["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info["packages"][p] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
    except Exception:  # pragma: no cover
        pass
    return info


def get_device(requested: str | None = "auto"):
    import torch

    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
