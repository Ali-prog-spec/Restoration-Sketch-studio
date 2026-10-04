"""Retrain the selected Task 4 configuration with the FULL schedule.

    python scripts/train_task4.py --mode final --use-best
"""

import _bootstrap  # noqa: F401

import argparse

from src.evaluation.plots import training_curves
from src.optimization.runner import config_with_best
from src.training.train_gan import train_gan
from src.utils.config import add_common_args, config_from_args, resolve, save_json
from src.utils.env_info import environment_info
from src.utils.tracking import Tracker


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--use-best", action="store_true")
    ap.add_argument("--best-from", default=None)
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task4.yaml")
    best = None
    if args.use_best:
        cfg, best = config_with_best(cfg, args.best_from)
        if best is None:
            raise SystemExit("no best_params.json — run scripts/optimize_task4.py first")
    out, ckpt = cfg["output"]["dir"], cfg["output"]["checkpoint"]
    tracker = Tracker(cfg)
    with tracker.run(f"task4-train-{cfg['mode']}", tags={"task": "task4"}):
        tracker.log_params(cfg)
        tracker.log_params({"env": environment_info(), "optuna_best": best or "none"})
        res = train_gan(cfg, tracker=tracker, checkpoint_path=ckpt, history_path=f"{out}/history.csv",
                        progress_dir=f"{out}/progress")
        fig1 = training_curves(f"{out}/history.csv", f"{out}/figures/gan_losses.png",
                               pairs=[("train_d_real", "train_d_fake", "D loss (real / fake)"),
                                      ("train_g_adv", "train_g_adv", "G adversarial"),
                                      ("train_g_l1", "val_l1", "L1 (train G / val)")],
                               title="Task 4 GAN losses")
        fig2 = training_curves(f"{out}/history.csv", f"{out}/figures/val_metrics.png",
                               pairs=[("val_ssim", "val_ssim", "Val. SSIM"), ("val_psnr", "val_psnr", "Val. PSNR"),
                                      ("train_d_real_prob", "train_d_fake_prob", "D(real) / D(fake) prob")],
                               title="Task 4 validation")
        save_json({"best": res["best"], "g_params": res["g_params"], "d_params": res["d_params"], "config": cfg,
                   "optuna_best_params": best, "checkpoint": ckpt, "environment": environment_info()},
                  f"{out}/train_summary.json")
        for f in (fig1, fig2):
            tracker.log_artifact(f)
        tracker.log_artifact(resolve(f"{out}/history.csv"))
        tracker.log_artifact(resolve(f"{out}/progress/progression_summary.png"), "progress")
        tracker.log_artifact(resolve(ckpt), "checkpoints")
    print(f"best validation: {res['best']}")


if __name__ == "__main__":
    main()
