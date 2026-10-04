"""Optuna study for Task 1 (universal DAE).

    python scripts/optimize_task1.py --mode smoke
    python scripts/optimize_task1.py --mode final            # 30 trials x 15 epochs by default
    python scripts/optimize_task1.py --mode final --num-trials 40
"""

import _bootstrap  # noqa: F401

import argparse

from src.optimization.runner import run_study
from src.training.train_autoencoder import train_autoencoder
from src.utils.config import add_common_args, config_from_args


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args, "configs/task1.yaml")

    def train_fn(tcfg, trial):
        res = train_autoencoder(tcfg, allowed_types=None, trial=trial, desc=f"T1-trial{trial.number}")
        b = res["best"]
        return b["objective"], {"val_l1": b["l1"], "val_ssim": b["ssim"], "val_psnr": b["psnr"],
                                "best_epoch": b["epoch"], "bottleneck_dim": res["bottleneck_dim"]}

    m = cfg["model"]
    baseline = {"lr": cfg["train"]["lr"], "batch_size": cfg["train"]["batch_size"],
                "bottleneck": "8x8x16", "base_channels": m["base_channels"],
                "dropout": m["dropout"], "alpha": cfg["loss"]["alpha"]}
    run_study(cfg, train_fn, baseline_params=baseline)


if __name__ == "__main__":
    main()
