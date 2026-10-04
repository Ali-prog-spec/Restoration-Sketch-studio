"""Export every model needed by the application to ONNX and validate it.

    python scripts/export_onnx.py --mode final                 # -> models/*.onnx
    python scripts/export_onnx.py --mode smoke                 # -> outputs/smoke/models/*.onnx
    python scripts/export_onnx.py --mode final --only task1 task3

Writes models/model_card.json (metadata read by the backend) and
outputs/onnx/validation_report.{json,md}. Exits with code 1 if any model fails
the PyTorch-vs-ONNX Runtime comparison.
"""

import _bootstrap  # noqa: F401

import argparse
import hashlib
import sys

import numpy as np
import torch

from src.corruptions.constants import CLASS_NAMES, STYLE_NAMES
from src.export.onnx_utils import export_onnx, validate_onnx
from src.models import SoftMoEExport, StyleUNetGenerator
from src.models.loading import EXPERT_ORDER, load_autoencoder, load_classifier, load_moe
from src.utils.config import load_config, resolve, save_json
from src.utils.env_info import environment_info

FILES = {
    "task1": "task1_universal_dae.onnx",
    "task2_classifier": "task2_classifier.onnx",
    "task2_salt_pepper": "task2_salt_expert.onnx",
    "task2_blur": "task2_blur_expert.onnx",
    "task2_occlusion": "task2_occlusion_expert.onnx",
    "task3": "task3_soft_moe.onnx",
    "task4": "task4_generator.onnx",
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def oxford_samples(n_per_class: int = 4) -> torch.Tensor:
    """Real validation inputs (from the deterministic manifest) + a random batch."""
    from src.data.oxford import ManifestDataset, load_cache, load_manifest

    xs = []
    try:
        images, ids = load_cache("trainval")
        ds = ManifestDataset(images, ids, load_manifest("val"))
        per = {c: 0 for c in CLASS_NAMES}
        for i in range(len(ds)):
            t = ds.records[i]["type"]
            if per[t] < n_per_class:
                xs.append(ds[i]["input"])
                per[t] += 1
            if all(v >= n_per_class for v in per.values()):
                break
    except FileNotFoundError:
        pass
    g = torch.Generator().manual_seed(42)
    xs += list(torch.rand(4, 3, 128, 128, generator=g))
    return torch.stack(xs)


def fs2k_samples(n: int = 6):
    xs, ss = [], []
    try:
        from src.data.fs2k import FS2KPairs, load_split

        split = load_split()
        ds = FS2KPairs(split["val"], split["root"], 1, train=False)
        for i in range(min(n, len(ds))):
            it = ds[i]
            xs.append(it["photo"])
            ss.append(it["style"])
    except FileNotFoundError:
        pass
    g = torch.Generator().manual_seed(42)
    xs += list(torch.rand(3, 3, 128, 128, generator=g) * 2 - 1)
    ss += [0, 1, 2]
    return torch.stack(xs), torch.tensor(ss, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="final", choices=["smoke", "dev", "final"])
    ap.add_argument("--only", nargs="*", default=["task1", "task2", "task3", "task4"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()
    mode = args.mode
    out = resolve(args.out or ("models" if mode != "smoke" else "outputs/smoke/models"))
    out.mkdir(parents=True, exist_ok=True)
    c1 = load_config("configs/task1.yaml", mode)
    c2c = load_config("configs/task2_classifier.yaml", mode)
    c2s = load_config("configs/task2_specialists.yaml", mode)
    c3 = load_config("configs/task3.yaml", mode)
    c4 = load_config("configs/task4.yaml", mode)

    jobs = []  # (key, model, example_inputs, input_names, output_names, validation_inputs)
    x_ox = oxford_samples()
    if "task1" in args.only:
        jobs.append(("task1", load_autoencoder(c1["output"]["checkpoint"]), ["input"], ["restored"], (x_ox,)))
    if "task2" in args.only:
        jobs.append(("task2_classifier", load_classifier(c2c["output"]["checkpoint"]), ["input"], ["logits"], (x_ox,)))
        for c in EXPERT_ORDER:
            jobs.append((f"task2_{c}", load_autoencoder(f"{c2s['output']['checkpoint_dir']}/{c}_expert.pt"),
                         ["input"], ["restored"], (x_ox,)))
    tau = None
    if "task3" in args.only:
        moe = load_moe(c3["output"]["checkpoint"])
        tau = float(moe.tau)
        jobs.append(("task3", SoftMoEExport(moe), ["input"], ["restored", "weights"], (x_ox,)))
    g_cfg = None
    if "task4" in args.only:
        ck = torch.load(resolve(c4["output"]["checkpoint"]), map_location="cpu", weights_only=False)
        G = StyleUNetGenerator(**ck["model_config"])
        G.load_state_dict(ck["model_state"])
        g_cfg = ck["model_config"]
        x4, s4 = fs2k_samples()
        jobs.append(("task4", G.eval(), ["photo", "style"], ["sketch"], (x4, s4)))

    report, card_models, all_ok = {}, {}, True
    for key, model, in_names, out_names, val_inputs in jobs:
        path = out / FILES[key]
        example = tuple(v[:2] for v in val_inputs)
        print(f"[export] {key} -> {path}")
        info = export_onnx(model, example, path, in_names, out_names)
        val = validate_onnx(model, path, val_inputs, atol=args.atol)
        all_ok &= val["passed"]
        report[key] = {**info, **val}
        card_models[key] = {"file": FILES[key], "sha256": sha256(path), "inputs": val["inputs"],
                            "outputs": out_names, "exporter": info["exporter"], "opset": info["opset"]}
        print(f"   passed={val['passed']} max|diff|={[o['max_abs_diff'] for o in val['outputs']]}")

    report["summary"] = {"all_passed": bool(all_ok), "n_models": len(jobs), "environment": environment_info()}
    save_json(report, f"outputs/onnx/validation_report{'_smoke' if mode == 'smoke' else ''}.json")
    lines = ["# ONNX validation report", "", f"Mode: `{mode}` — tolerance atol={args.atol}, rtol=1e-3 (np.allclose)", "",
             "| Model | File | Exporter | Batch | Output | Shape match | max abs diff | mean abs diff | Passed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, r in report.items():
        if key == "summary":
            continue
        for o in r["outputs"]:
            lines.append(f"| {key} | {FILES[key]} | {r['exporter']} | {r['batch_size']} | {o['name']} | {o['shape_match']} | "
                         f"{o['max_abs_diff']:.3e} | {o['mean_abs_diff']:.3e} | {'PASS' if o['allclose'] else 'FAIL'} |")
    lines += ["", f"**All passed: {all_ok}**"]
    p = resolve(f"outputs/onnx/validation_report{'_smoke' if mode == 'smoke' else ''}.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")

    card_path = out / "model_card.json"
    card = {}
    if card_path.exists():
        import json

        card = json.loads(card_path.read_text(encoding="utf-8"))
    card.setdefault("models", {}).update(card_models)
    card.update({"image_size": 128, "class_names": CLASS_NAMES, "style_names": STYLE_NAMES,
                 "restoration_input": "float32 RGB in [0,1], NCHW", "sketch_input": "float32 RGB in [-1,1], NCHW; style int64 in {0,1,2}",
                 "sketch_output": "float32 in [-1,1]", "mode": mode})
    if tau is not None:
        card["task3_tau"] = tau
    if g_cfg is not None:
        card["task4_generator_config"] = g_cfg
    save_json(card, card_path)
    print(f"model card -> {card_path}\nall passed: {all_ok}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
