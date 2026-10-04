"""Final TEST evaluation of the Task 3 soft MoE + routing analysis.

    python scripts/evaluate_task3.py --mode final

Produces (outputs/task3/):
  per_record.csv (with the 4 routing weights of every record), metrics by type/level
  routing_mean_weights_by_type.csv, routing_mean_weights_by_level.csv  (PDF p.7)
  figures/routing_heatmap.png, weight_distribution.png, expert_utilisation.png
  figures/dominant_examples.png, mixed_examples.png
  routing_diagnostics.json  (inactive experts, dominance over unrelated inputs, entropy)
  system_comparison.csv  (Task 1 vs Task 2 oracle/predicted vs Task 3, if available)
  mixed_corruption_stress.csv  (extra analysis: images with TWO corruptions)
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.corruptions import apply_corruption, level_params
from src.corruptions.constants import CLASS_DISPLAY, CLASS_NAMES
from src.data.oxford import to_tensor
from src.evaluation.metrics import batch_metrics
from src.evaluation.plots import (BRANCH_NAMES, compare_systems_plot, image_row_grid, routing_heatmap,
                                  utilization_plot, weight_distribution_plot)
from src.evaluation.reporting import report_dir, example_figures, latex_table, save_tables
from src.evaluation.restoration import run_restoration_eval, select_failures, select_representative, summarize
from src.losses import routing_entropy
from src.models import hard_route
from src.models.loading import load_classifier, load_experts, load_moe
from src.training.common import autocast, test_loader
from src.utils.config import add_common_args, config_from_args, resolve, save_json
from src.utils.env_info import get_device
from src.utils.tracking import Tracker


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--tag", default="")
    ap.add_argument("--dominant-threshold", type=float, default=0.9)
    ap.add_argument("--mixed-threshold", type=float, default=0.6)
    ap.add_argument("--stress-images", type=int, default=300, help="images for the two-corruption stress test")
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task3.yaml")
    device = get_device(cfg.get("device"))
    tag = f"_{args.tag}" if args.tag else ""
    moe = load_moe(cfg["output"]["checkpoint"].replace(".pt", f"{tag}.pt")).to(device)
    out = cfg["output"]["dir"]
    loader = test_loader(cfg, batch_size=64)
    ds = loader.dataset

    @torch.no_grad()
    def restore(x, batch):
        with autocast(device, True):
            y, w, _ = moe(x)
        w = w.float()
        return {"output": y.float(), "weights": w, "entropy": routing_entropy(w), "dominant": w.argmax(1)}

    df = run_restoration_eval(restore, loader, device, desc="soft MoE")
    df.to_csv(resolve(f"{out}/per_record{tag}.csv"), index=False)
    by_type, by_level, by_sev = summarize(df)
    save_tables({f"metrics_by_type{tag}": by_type, f"metrics_by_level{tag}": by_level,
                 f"metrics_by_severity{tag}": by_sev}, out)
    cols = ["l1", "psnr", "ssim", "input_psnr", "input_ssim", "n"]
    fmt = {"psnr": "{:.2f}", "input_psnr": "{:.2f}", "n": "{:.0f}"}
    latex_table(by_level, f"{report_dir(cfg)}/task3_by_level{tag}.tex", "Task 3 soft MoE test results.",
                f"tab:task3_level{tag}", cols, fmt)

    # ---------------- routing analysis (PDF p.7) ----------------
    wcols = [f"weights_{k}" for k in range(4)]
    mw_type = df.groupby("type", sort=False)[wcols].mean().reindex(CLASS_NAMES)
    mw_type.columns = BRANCH_NAMES
    mw_level = df.groupby(["type", "level"], sort=False)[wcols].mean()
    mw_level.columns = BRANCH_NAMES
    mw_type.to_csv(resolve(f"{out}/routing_mean_weights_by_type{tag}.csv"))
    mw_level.to_csv(resolve(f"{out}/routing_mean_weights_by_level{tag}.csv"))
    latex_table(mw_level, f"{report_dir(cfg)}/task3_routing_weights{tag}.tex",
                "Average soft-MoE routing weight per true condition and severity (test).",
                f"tab:task3_routing{tag}", BRANCH_NAMES, {c: "{:.3f}" for c in BRANCH_NAMES})
    heat = mw_level.copy()
    heat.index = [f"{CLASS_DISPLAY[t].split()[0]} / {lv}" for t, lv in heat.index]
    routing_heatmap(heat, f"{out}/figures/routing_heatmap{tag}.png", "Task 3: mean routing weight (test)")
    weight_distribution_plot(df, f"{out}/figures/weight_distribution{tag}.png")
    util = pd.DataFrame({"mean_weight": df[wcols].mean().values,
                         "argmax_share": np.bincount(df["dominant"].astype(int), minlength=4) / len(df)},
                        index=BRANCH_NAMES)
    util.to_csv(resolve(f"{out}/expert_utilisation{tag}.csv"))
    utilization_plot(util, f"{out}/figures/expert_utilisation{tag}.png")

    mw = mw_type.values  # rows true class, cols branch
    inactive = [BRANCH_NAMES[k] for k in range(4) if mw[:, k].max() < 0.05]
    dominance = [{"branch": BRANCH_NAMES[k], "true_class": CLASS_NAMES[c], "mean_weight": float(mw[c, k])}
                 for c in range(4) for k in range(4) if k != c and mw[c, k] > 0.3]
    diagnostics = {
        "inactive_experts(max mean weight < 0.05)": inactive,
        "unrelated_dominance(mean weight > 0.3 on another class)": dominance,
        "correct_branch_mean_weight": {CLASS_NAMES[c]: float(mw[c, c]) for c in range(4)},
        "routing_accuracy(argmax == true)": float((df["dominant"].astype(int) == df["type"].map(CLASS_NAMES.index)).mean()),
        "mean_entropy_nats": float(df["entropy"].mean()), "max_entropy_nats": float(np.log(4)),
        "share_dominant(max w > %.2f)" % args.dominant_threshold: float((df[wcols].max(axis=1) > args.dominant_threshold).mean()),
        "share_mixed(max w < %.2f)" % args.mixed_threshold: float((df[wcols].max(axis=1) < args.mixed_threshold).mean()),
        "entropy_by_type": df.groupby("type")["entropy"].mean().to_dict(),
    }
    save_json(diagnostics, f"{out}/routing_diagnostics{tag}.json")

    # dominant vs mixed routing examples
    def weight_grid(sel: pd.DataFrame, name: str, title: str):
        rows, caps = [], []
        for _, r in sel.iterrows():
            item = ds[int(r["record"])]
            with torch.no_grad():
                y, w, _ = moe(item["input"][None].to(device))
            rows.append([item["target"], item["input"], y[0].float().cpu()])
            ws = " ".join(f"{v:.2f}" for v in w[0].float().cpu().tolist())
            caps.append(f"{CLASS_DISPLAY[r['type']].split()[0]} ({r['level']})\nw=[{ws}]\nSSIM {r['ssim']:.3f}")
        if rows:
            image_row_grid(rows, ["Clean target", "Input", "MoE output"], f"{out}/figures/{name}{tag}.png", caps, title=title)
    dom_all = df[df[wcols].max(axis=1) > args.dominant_threshold]
    # up to 2 random examples per true condition (keeps the 'type' column, unlike groupby.apply in pandas 3)
    dom = pd.concat([g.sample(min(2, len(g)), random_state=42) for _, g in dom_all.groupby("type")]) if len(dom_all) else dom_all
    mix = df[df[wcols].max(axis=1) < args.mixed_threshold].sort_values("entropy", ascending=False).drop_duplicates("image_id").head(8)
    weight_grid(dom, "dominant_examples", "Task 3: one expert dominates (max weight > %.2f)" % args.dominant_threshold)
    weight_grid(mix, "mixed_examples", "Task 3: weights distributed over several branches")
    dom.to_csv(resolve(f"{out}/dominant_examples{tag}.csv"), index=False)
    mix.to_csv(resolve(f"{out}/mixed_examples{tag}.csv"), index=False)

    # representative / failures
    @torch.no_grad()
    def restore_one(x):
        return moe(x[None].to(device))[0][0].float().cpu()

    cap = lambda r: "w=[" + " ".join(f"{r[c]:.2f}" for c in wcols) + "]"  # noqa: E731
    example_figures(ds, df, select_representative(df), restore_one, f"{out}/figures/representative{tag}",
                    "Task 3: representative test examples", extra_caption=cap)
    fails = select_failures(df)
    fails.to_csv(resolve(f"{out}/failure_cases{tag}.csv"), index=False)
    example_figures(ds, df, [int(r) for r in fails["record"]], restore_one, f"{out}/figures/failures{tag}",
                    "Task 3: failure cases", extra_caption=cap)

    # ---------------- comparison with Tasks 1 and 2 ----------------
    systems = {"Task 3 soft MoE": by_level}
    base = "outputs" if cfg["mode"] != "smoke" else "outputs/smoke"
    for name, path in [("Task 1 universal", f"{base}/task1/metrics_by_level.csv"),
                       ("Task 2 oracle", f"{base}/task2/routing_oracle_by_level.csv"),
                       ("Task 2 predicted", f"{base}/task2/routing_predicted_by_level.csv")]:
        if resolve(path).exists():
            systems[name] = pd.read_csv(resolve(path), index_col=[0, 1])
    if len(systems) > 1:
        comp = pd.DataFrame({f"{n}_ssim": t["ssim"] for n, t in systems.items()})
        comp_psnr = pd.DataFrame({f"{n}_psnr": t["psnr"] for n, t in systems.items()})
        comp = comp.join(comp_psnr)
        comp.to_csv(resolve(f"{out}/system_comparison{tag}.csv"))
        compare_systems_plot(systems, f"{out}/figures/system_comparison_ssim{tag}.png", "ssim",
                             "Test SSIM of all restoration systems")

    # ---------------- extra: two-corruption stress test ----------------
    stress = two_corruption_stress(moe, cfg, device, args.stress_images)
    if stress is not None:
        stress.to_csv(resolve(f"{out}/mixed_corruption_stress{tag}.csv"))

    tracker = Tracker(cfg)
    with tracker.run(f"task3-test-eval-{cfg['mode']}{tag}", tags={"task": "task3", "split": "test"}):
        tracker.log_metrics({f"test_{k}": v for k, v in by_type.loc["overall"].items()})
        tracker.log_metrics({"routing_accuracy": diagnostics["routing_accuracy(argmax == true)"],
                             "mean_entropy": diagnostics["mean_entropy_nats"]})
        tracker.log_artifacts(resolve(f"{out}/figures"), "figures")
        tracker.log_artifact(resolve(f"{out}/routing_diagnostics{tag}.json"))
    print(by_type.round(4).to_string())
    print(mw_level.round(3).to_string())
    print(diagnostics)


@torch.no_grad()
def two_corruption_stress(moe, cfg, device, n_images: int):
    """Motivating case of Task 3 (PDF p.5): an image with TWO corruptions. Compares soft
    MoE, hard predicted routing (Task 2) and the corrupted input. Clearly an EXTRA
    analysis — not part of the official test protocol."""
    from src.data.oxford import load_cache

    t2_clf, t2_dir = Path("checkpoints/task2/classifier.pt"), "checkpoints/task2"
    if cfg["mode"] == "smoke":
        t2_clf, t2_dir = Path("checkpoints/smoke/classifier.pt"), "checkpoints/smoke/task2"
    if not resolve(t2_clf).exists():
        return None
    clf = load_classifier(t2_clf).to(device)
    experts = [e.to(device) for e in load_experts(t2_dir)]
    path, ids = load_cache("test")
    images = np.load(resolve(path), mmap_mode="r")
    n = min(n_images, len(ids))
    combos = [("blur", "salt_pepper"), ("occlusion", "salt_pepper"), ("blur", "occlusion")]
    rows = []
    for a, b in combos:
        for level in ("low", "medium"):
            xs, cs = [], []
            for i in range(n):
                clean = images[i].astype(np.float32) / 255.0
                seed = 10_000 * (combos.index((a, b)) + 1) + i
                x = apply_corruption(apply_corruption(clean, level_params(a, level, seed)), level_params(b, level, seed + 1))
                xs.append(to_tensor(x))
                cs.append(to_tensor(clean))
            x = torch.stack(xs).to(device)
            c = torch.stack(cs).to(device)
            y_moe = torch.cat([moe(x[k:k + 64])[0] for k in range(0, n, 64)]).float()
            y_hard = torch.cat([hard_route(x[k:k + 64], clf(x[k:k + 64]).argmax(1), experts) for k in range(0, n, 64)]).float()
            for name, y in [("input", x), ("task2_hard_predicted", y_hard), ("task3_soft_moe", y_moe)]:
                m = batch_metrics(y, c)
                rows.append({"combo": f"{a}+{b}", "level": level, "system": name,
                             "ssim": float(m["ssim"].mean()), "psnr": float(m["psnr"].mean()), "l1": float(m["l1"].mean())})
    return pd.DataFrame(rows).pivot_table(index=["combo", "level"], columns="system", values=["ssim", "psnr"])


if __name__ == "__main__":
    main()
