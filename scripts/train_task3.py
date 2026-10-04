"""Train the final Task 3 soft MoE (warm-up + joint fine-tuning).

    python scripts/train_task3.py --mode final --use-best
    python scripts/train_task3.py --mode final --use-best --balance entropy --tag entropy   # balance ablation
"""

import _bootstrap  # noqa: F401

import argparse

from src.evaluation.plots import training_curves
from src.optimization.runner import config_with_best
from src.training.train_moe import train_moe
from src.utils.config import add_common_args, config_from_args, resolve, save_json
from src.utils.env_info import environment_info
from src.utils.tracking import Tracker


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--use-best", action="store_true")
    ap.add_argument("--best-from", default=None)
    ap.add_argument("--balance", choices=["squared", "entropy"], default=None, help="override balance loss")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task3.yaml")
    best = None
    if args.use_best:
        cfg, best = config_with_best(cfg, args.best_from)
        if best is None:
            raise SystemExit("no best_params.json — run scripts/optimize_task3.py first")
    if args.balance:
        cfg["loss"]["balance"] = args.balance
    tag = f"_{args.tag}" if args.tag else ""
    out = cfg["output"]["dir"]
    ckpt = cfg["output"]["checkpoint"].replace(".pt", f"{tag}.pt")
    tracker = Tracker(cfg)
    with tracker.run(f"task3-train-{cfg['mode']}{tag}", tags={"task": "task3"}):
        tracker.log_params(cfg)
        tracker.log_params({"env": environment_info(), "optuna_best": best or "none"})
        res = train_moe(cfg, tracker=tracker, checkpoint_path=ckpt, history_path=f"{out}/history{tag}.csv")
        fig = training_curves(f"{out}/history{tag}.csv", f"{out}/figures/training_curves{tag}.png",
                              pairs=[("train_loss", "val_loss", "Joint loss"), ("train_ssim", "val_ssim", "SSIM"),
                                     ("train_ce", "val_ce", "Gate CE")],
                              title="Task 3 soft MoE (warm-up then joint)")
        save_json({"best": res["best"], "initial": res["initial"], "config": cfg, "optuna_best_params": best,
                   "checkpoint": ckpt, "environment": environment_info()}, f"{out}/train_summary{tag}.json")
        tracker.log_artifact(fig)
        tracker.log_artifact(resolve(f"{out}/history{tag}.csv"))
        tracker.log_artifact(resolve(ckpt), "checkpoints")
    print(f"initial (Task 2 weights): {res['initial']['objective']:.4f}  best: {res['best']['objective']:.4f}")


if __name__ == "__main__":
    main()
