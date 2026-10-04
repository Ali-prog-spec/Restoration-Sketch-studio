"""Train the final Task 2 corruption classifier.

    python scripts/train_task2_classifier.py --mode final --use-best
"""

import _bootstrap  # noqa: F401

import argparse

from src.evaluation.plots import training_curves
from src.optimization.runner import config_with_best
from src.training.train_classifier import train_classifier
from src.utils.config import add_common_args, config_from_args, resolve, save_json
from src.utils.env_info import environment_info
from src.utils.tracking import Tracker


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--use-best", action="store_true")
    ap.add_argument("--best-from", default=None)
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task2_classifier.yaml")
    best = None
    if args.use_best:
        cfg, best = config_with_best(cfg, args.best_from)
        if best is None:
            raise SystemExit("no best_params.json — run scripts/optimize_task2_classifier.py first")
    out, ckpt = cfg["output"]["dir"], cfg["output"]["checkpoint"]
    tracker = Tracker(cfg)
    with tracker.run(f"task2-classifier-train-{cfg['mode']}", tags={"task": "task2", "part": "classifier"}):
        tracker.log_params(cfg)
        tracker.log_params({"env": environment_info(), "optuna_best": best or "none"})
        res = train_classifier(cfg, tracker=tracker, checkpoint_path=ckpt, history_path=f"{out}/history.csv")
        fig = training_curves(f"{out}/history.csv", f"{out}/figures/training_curves.png",
                              pairs=[("train_loss", "val_loss", "Cross-entropy"),
                                     ("train_accuracy", "val_accuracy", "Accuracy"),
                                     ("val_macro_f1", "val_macro_f1", "Val. macro-F1")],
                              title="Task 2 corruption classifier")
        save_json({"best": res["best"], "config": cfg, "optuna_best_params": best, "checkpoint": ckpt,
                   "environment": environment_info()}, f"{out}/train_summary.json")
        tracker.log_artifact(fig)
        tracker.log_artifact(resolve(f"{out}/history.csv"))
        tracker.log_artifact(resolve(ckpt), "checkpoints")
    print(f"best validation: {res['best']}")


if __name__ == "__main__":
    main()
