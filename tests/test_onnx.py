"""ONNX export + PyTorch/ONNX Runtime consistency utilities."""

import torch

from src.export.onnx_utils import export_onnx, validate_onnx
from src.models import CorruptionClassifier, DenoisingAutoencoder, SoftMoE, SoftMoEExport, StyleUNetGenerator


def test_autoencoder_export_matches(tmp_path):
    m = DenoisingAutoencoder(base_channels=8, latent_channels=4).eval()
    x = torch.rand(3, 3, 128, 128)
    export_onnx(m, (x[:2],), tmp_path / "ae.onnx", ["input"], ["restored"])
    r = validate_onnx(m, tmp_path / "ae.onnx", (x,))  # batch 3 != export batch 2 -> dynamic batch works
    assert r["passed"] and r["outputs"][0]["shape_match"] and r["outputs"][0]["max_abs_diff"] < 1e-4


def test_moe_pipeline_export_matches(tmp_path):
    moe = SoftMoE(CorruptionClassifier(base_channels=8), *[DenoisingAutoencoder(base_channels=8) for _ in range(3)], tau=1.3)
    wrapper = SoftMoEExport(moe).eval()
    x = torch.rand(2, 3, 128, 128)
    export_onnx(wrapper, (x,), tmp_path / "moe.onnx", ["input"], ["restored", "weights"])
    r = validate_onnx(wrapper, tmp_path / "moe.onnx", (x,))
    assert r["passed"] and len(r["outputs"]) == 2


def test_generator_export_with_style_input(tmp_path):
    g = StyleUNetGenerator(base_channels=8, style_dim=4).eval()
    x, s = torch.rand(3, 3, 128, 128) * 2 - 1, torch.tensor([0, 1, 2])
    export_onnx(g, (x[:2], s[:2]), tmp_path / "g.onnx", ["photo", "style"], ["sketch"])
    r = validate_onnx(g, tmp_path / "g.onnx", (x, s))
    assert r["passed"] and [i["name"] for i in r["inputs"]] == ["photo", "style"]


def test_validation_detects_mismatch(tmp_path):
    m = DenoisingAutoencoder(base_channels=8, latent_channels=4).eval()
    x = torch.rand(2, 3, 128, 128)
    export_onnx(m, (x,), tmp_path / "ae.onnx", ["input"], ["restored"])
    other = DenoisingAutoencoder(base_channels=8, latent_channels=4).eval()  # different weights
    assert not validate_onnx(other, tmp_path / "ae.onnx", (x,))["passed"]
