"""Deterministic validation / test corruption manifests (PDF p.3).

Every record stores the image id, corruption type, severity level, the per-record
random seed and every parameter needed to regenerate the corruption exactly:
salt probability (+ seed for the pixel pattern), blur kernel and sigma, rectangle
coordinates and achieved coverage.

* Validation: four records per validation image (one per condition, so the set is
  exactly balanced — equal probability in expectation becomes equal counts), with
  severities sampled from the TRAINING ranges using a per-record seed. Four records
  per image also give each Task 2 specialist a full-size validation set.
* Test: ten records per test image — clean + 3 corruptions x 3 fixed levels
  (salt 0.03/0.08/0.15; blur (3,0.7)/(5,1.5)/(7,2.5); occlusion ~10/20/35% with
  1/2/3 rectangles).
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from src.corruptions import CLASS_NAMES, level_params, sample_params
from src.corruptions.constants import IMAGE_SIZE, SEVERITY_LEVELS, TEST_LEVELS


def _record_seed(base_seed: int, *keys: int) -> int:
    """A stable, collision-free per-record seed derived with NumPy's SeedSequence."""
    return int(np.random.SeedSequence([base_seed, *keys]).generate_state(1)[0] & 0x7FFFFFFF)


def build_validation_manifest(val_ids, seed: int = 42) -> dict:
    records = []
    for i, image_id in enumerate(val_ids):
        for t_idx, ctype in enumerate(CLASS_NAMES):
            rs = _record_seed(seed, 0, i, t_idx)
            rng = np.random.default_rng(rs)
            params = sample_params(rng, ctype, IMAGE_SIZE, IMAGE_SIZE)
            records.append({"image_id": image_id, "level": "sampled", "record_seed": rs, **params})
    return {"meta": _meta("validation", seed, len(val_ids)), "records": records}


def build_test_manifest(test_ids, seed: int = 42) -> dict:
    records = []
    for i, image_id in enumerate(test_ids):
        records.append({"image_id": image_id, "level": "none", "record_seed": None, "type": "clean"})
        for t_idx, ctype in enumerate(CLASS_NAMES[1:], start=1):
            for l_idx, level in enumerate(SEVERITY_LEVELS):
                rs = _record_seed(seed, 1, i, t_idx, l_idx)
                params = level_params(ctype, level, rs, IMAGE_SIZE, IMAGE_SIZE)
                records.append({"image_id": image_id, "level": level, "record_seed": rs, **params})
    return {"meta": _meta("test", seed, len(test_ids)), "records": records}


def _meta(name, seed, n_images):
    return {
        "name": name,
        "base_seed": seed,
        "num_images": n_images,
        "image_size": IMAGE_SIZE,
        "classes": CLASS_NAMES,
        "test_levels": TEST_LEVELS,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "note": "Regenerate a corrupted image with src.corruptions.apply_corruption(clean, record).",
    }
