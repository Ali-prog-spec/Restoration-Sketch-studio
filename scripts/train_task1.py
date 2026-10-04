"""Train the final Task 1 universal DAE.

    python scripts/train_task1.py --mode smoke
    python scripts/train_task1.py --mode final --use-best          # best params of the 'final' study
    python scripts/train_task1.py --mode final --use-best --best-from dev
    python scripts/train_task1.py --mode final --skip-levels 0     # limited-skip ablation
"""

import _bootstrap  # noqa: F401

import argparse

from src.evaluation.plots import training_curves
from src.optimization.runner import config_with_best
from src.training.train_autoencoder import train_autoencoder
from src.utils.config import add_common_args, config_from_args, resolve, save_json
from src.utils.env_info import environment_info
from src.utils.tracking import Tracker


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--use-best", action="store_true", help="merge best Optuna params")
    ap.add_argument("--best-from", default=None, help="mode whose study provides the best params")
    ap.add_argument("--skip-levels", type=int, nargs="*", default=None, help="enable limited skips (ablation)")
    ap.add_argument("--tag", default="", help="suffix for output names (e.g. ablation)")
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task1.yaml")
    best = None
    if args.use_best:
        cfg, best = config_with_best(cfg, args.best_from)
        if best is None:
            raise SystemExit("no best_params.json found — run scripts/optimize_task1.py first")
    if args.skip_levels is not None:
        cfg["model"]["skip_levels"] = args.skip_levels
    tag = f"_{args.tag}" if args.tag else ""
    out = cfg["output"]["dir"]
    ckpt = cfg["output"]["checkpoint"].replace(".pt", f"{tag}.pt")

    tracker = Tracker(cfg)
    with tracker.run(f"task1-train-{cfg['mode']}{tag}", tags={"task": "task1"}):
        tracker.log_params(cfg)
        tracker.log_params({"env": environment_info(), "optuna_best": best or "none"})
        res = train_autoencoder(cfg, None, tracker=tracker, checkpoint_path=ckpt, desc="Task1",
                                history_path=f"{out}/history{tag}.csv")
        fig = training_curves(f"{out}/history{tag}.csv", f"{out}/figures/training_curves{tag}.png",
                              title="Task 1 universal DAE")
        summary = {"best": res["best"], "bottleneck_dim": res["bottleneck_dim"], "config": cfg,
                   "optuna_best_params": best, "checkpoint": ckpt, "environment": environment_info()}
        save_json(summary, f"{out}/train_summary{tag}.json")
        tracker.log_metrics({f"best_val_{k}": v for k, v in res["best"].items() if isinstance(v, float)})
        tracker.log_artifact(fig)
        tracker.log_artifact(resolve(f"{out}/history{tag}.csv"))
        tracker.log_artifact(resolve(f"{out}/train_summary{tag}.json"))
        tracker.log_artifact(resolve(ckpt), "checkpoints")
    print(f"best validation: {res['best']}")


if __name__ == "__main__":
    main()
