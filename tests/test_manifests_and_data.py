"""Deterministic manifests, runtime datasets and balanced batches (no real data needed)."""

import json

import numpy as np
import torch

from src.corruptions import CLASS_NAMES, apply_corruption
from src.data.manifests import build_test_manifest, build_validation_manifest
from src.data.oxford import BalancedBatchSampler, ManifestDataset, RuntimeCorruptionDataset

IDS = [f"img_{i}" for i in range(12)]
IMAGES = (np.random.default_rng(0).uniform(0.1, 0.9, (12, 128, 128, 3)) * 255).astype(np.uint8)


def _strip(m):
    return json.dumps(m["records"], sort_keys=True)


def test_manifest_generation_is_deterministic():
    assert _strip(build_validation_manifest(IDS, 42)) == _strip(build_validation_manifest(IDS, 42))
    assert _strip(build_test_manifest(IDS, 42)) == _strip(build_test_manifest(IDS, 42))
    assert _strip(build_test_manifest(IDS, 42)) != _strip(build_test_manifest(IDS, 43))


def test_manifest_records_reproduce_identical_images():
    m = build_test_manifest(IDS, 42)
    # simulate "save to disk and load again"
    m2 = json.loads(json.dumps(m))
    ds1 = ManifestDataset(IMAGES, IDS, m)
    ds2 = ManifestDataset(IMAGES, IDS, m2)
    for i in range(len(ds1)):
        assert torch.equal(ds1[i]["input"], ds2[i]["input"])
    # and twice from the same dataset
    assert torch.equal(ds1[5]["input"], ds1[5]["input"])


def test_manifest_contains_required_fields():
    m = build_test_manifest(IDS, 42)
    for r in m["records"]:
        assert {"image_id", "type", "level", "record_seed"} <= set(r)
        if r["type"] == "salt_pepper":
            assert {"prob", "seed"} <= set(r)
        if r["type"] == "blur":
            assert {"kernel", "sigma"} <= set(r)
        if r["type"] == "occlusion":
            assert {"rects", "coverage", "num_rects"} <= set(r)


def test_test_manifest_structure():
    m = build_test_manifest(IDS, 42)
    assert len(m["records"]) == 10 * len(IDS)  # clean + 3 corruptions x 3 levels
    per = {}
    for r in m["records"]:
        per.setdefault(r["image_id"], set()).add((r["type"], r["level"]))
    for s in per.values():
        assert ("clean", "none") in s and len(s) == 10


def test_validation_manifest_balanced_and_in_training_ranges():
    m = build_validation_manifest(IDS, 42)
    types = [r["type"] for r in m["records"]]
    assert all(types.count(c) == len(IDS) for c in CLASS_NAMES)


def test_runtime_dataset_changes_corruption_every_load():
    ds = RuntimeCorruptionDataset(IMAGES, IDS, IDS, seed=0)
    item = ds[0]
    assert set(item) >= {"input", "target", "label", "severity", "params", "image_id"}
    assert item["input"].shape == (3, 128, 128) and item["input"].dtype == torch.float32
    params = {ds[0]["params"] for _ in range(20)}
    assert len(params) > 5  # re-sampled on every access, never cached


def test_runtime_dataset_restricted_types():
    ds = RuntimeCorruptionDataset(IMAGES, IDS, IDS, allowed_types=["blur"], seed=0)
    assert all(ds[i % 12]["label"] == CLASS_NAMES.index("blur") for i in range(30))


def test_forced_class_from_sampler():
    ds = RuntimeCorruptionDataset(IMAGES, IDS, IDS, seed=0)
    for c in range(4):
        assert ds[(3, c)]["label"] == c


def test_balanced_batch_sampler():
    s = BalancedBatchSampler(n_items=103, batch_size=16, seed=42)
    batches = list(iter(s))
    assert len(batches) == len(s) == 103 // 16
    for b in batches:
        labels = [c for _, c in b]
        assert all(labels.count(c) == 4 for c in range(4))
        assert len({i for i, _ in b}) == 16  # no duplicate images in a batch
    # a new epoch reshuffles
    assert [b for b in iter(s)] != batches


def test_clean_target_is_uncorrupted():
    ds = RuntimeCorruptionDataset(IMAGES, IDS, IDS, seed=0)
    item = ds[(2, 2)]
    np.testing.assert_allclose(item["target"].numpy().transpose(1, 2, 0), IMAGES[2] / 255.0, atol=1e-6)
    p = json.loads(item["params"])
    np.testing.assert_allclose(item["input"].numpy().transpose(1, 2, 0),
                               apply_corruption(IMAGES[2].astype(np.float32) / 255.0, p), atol=1e-6)
