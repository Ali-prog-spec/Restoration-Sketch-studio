"""FastAPI backend: health, validation and all four inference endpoints.

A fixture exports tiny RANDOM-weight models to a temporary MODEL_DIR so the full
request -> preprocess -> ONNX Runtime -> response path is exercised without trained
weights (numerical quality is not tested here, only the pipeline)."""

import base64
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import inference as inf
from backend.app import main
from src.export.onnx_utils import export_onnx
from src.models import CorruptionClassifier, DenoisingAutoencoder, SoftMoE, SoftMoEExport, StyleUNetGenerator


def _png(size=(200, 150), color=(120, 80, 200)) -> bytes:
    buf = io.BytesIO()
    arr = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    x = torch.rand(1, 3, 128, 128)
    ae = lambda: DenoisingAutoencoder(base_channels=8, latent_channels=4).eval()  # noqa: E731
    export_onnx(ae(), (x,), d / inf.MODEL_FILES["task1"], ["input"], ["restored"])
    export_onnx(CorruptionClassifier(base_channels=8).eval(), (x,), d / inf.MODEL_FILES["task2_classifier"], ["input"], ["logits"])
    for k in ("task2_salt_pepper", "task2_blur", "task2_occlusion"):
        export_onnx(ae(), (x,), d / inf.MODEL_FILES[k], ["input"], ["restored"])
    moe = SoftMoE(CorruptionClassifier(base_channels=8), ae(), ae(), ae(), tau=1.0)
    export_onnx(SoftMoEExport(moe).eval(), (x,), d / inf.MODEL_FILES["task3"], ["input"], ["restored", "weights"])
    export_onnx(StyleUNetGenerator(base_channels=8, style_dim=4).eval(), (x * 2 - 1, torch.tensor([0])),
                d / inf.MODEL_FILES["task4"], ["photo", "style"], ["sketch"])
    return d


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "cat_1.png").write_bytes(_png())
    monkeypatch.setattr(main, "registry", inf.ModelRegistry(model_dir))
    monkeypatch.setattr(main, "SAMPLES_DIR", samples)
    return TestClient(main.app)


def _decode(url: str) -> Image.Image:
    assert url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["all_models_available"]


def test_health_reports_missing_models(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "registry", inf.ModelRegistry(tmp_path))
    c = TestClient(main.app)
    assert c.get("/health").json()["all_models_available"] is False
    r = c.post("/api/universal-restoration", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 503


def test_samples(client):
    s = client.get("/api/samples").json()
    assert s[0]["id"] == "cat_1"
    assert client.get("/api/samples/cat_1").status_code == 200
    assert client.get("/api/samples/nope").status_code == 404


@pytest.mark.parametrize("endpoint", ["/api/universal-restoration", "/api/hard-routing", "/api/soft-mixture"])
@pytest.mark.parametrize("corruption", ["none", "salt_pepper", "blur", "occlusion"])
def test_restoration_endpoints(client, endpoint, corruption):
    r = client.post(endpoint, files={"file": ("x.png", _png(), "image/png")},
                    data={"corruption": corruption, "severity": "high", "seed": "7"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert _decode(b["input_image"]).size == (128, 128) and _decode(b["output_image"]).size == (128, 128)
    assert b["timing_ms"]["inference"] >= 0 and b["timing_ms"]["total"] >= b["timing_ms"]["inference"]
    if corruption != "none":
        assert b["corruption"]["type"] == corruption and "metrics" in b
    if endpoint == "/api/hard-routing":
        assert set(b["probabilities"]) == {"clean", "salt_pepper", "blur", "occlusion"}
        assert abs(sum(b["probabilities"].values()) - 1) < 1e-4
        assert b["selected_expert"] in ("identity", "salt_pepper", "blur", "occlusion")
    if endpoint == "/api/soft-mixture":
        assert abs(sum(b["weights"].values()) - 1) < 1e-4 and len(b["ranking"]) == 4


def test_same_seed_same_corruption(client):
    kw = dict(data={"corruption": "occlusion", "severity": "medium", "seed": "11", "sample_id": "cat_1"})
    a = client.post("/api/universal-restoration", **kw).json()
    b = client.post("/api/universal-restoration", **kw).json()
    assert a["corruption"]["rects"] == b["corruption"]["rects"] and a["input_image"] == b["input_image"]


def test_custom_corruption_and_validation(client):
    ok = client.post("/api/universal-restoration", data={"sample_id": "cat_1", "corruption": "blur",
                                                          "severity": "custom", "kernel": "7", "sigma": "2.0"})
    assert ok.status_code == 200 and ok.json()["corruption"]["kernel"] == 7
    bad = client.post("/api/universal-restoration", data={"sample_id": "cat_1", "corruption": "blur",
                                                           "severity": "custom", "kernel": "4"})
    assert bad.status_code == 422
    assert client.post("/api/universal-restoration", data={"corruption": "blur"}).status_code == 422  # no image
    assert client.post("/api/universal-restoration", data={"sample_id": "cat_1", "corruption": "rain"}).status_code == 422


def test_upload_validation(client):
    r = client.post("/api/universal-restoration", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
    r = client.post("/api/universal-restoration", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    big = b"0" * (int(main.MAX_BYTES) + 10)
    r = client.post("/api/universal-restoration", files={"file": ("x.png", big, "image/png")})
    assert r.status_code == 413


def test_face_to_sketch(client):
    for style in (1, 2, 3):
        r = client.post("/api/face-to-sketch", files={"file": ("f.png", _png((300, 400)), "image/png")},
                        data={"style": str(style)})
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["style_index"] == style and _decode(b["output_image"]).size == (128, 128)
    assert client.post("/api/face-to-sketch", files={"file": ("f.png", _png(), "image/png")},
                       data={"style": "4"}).status_code == 422
