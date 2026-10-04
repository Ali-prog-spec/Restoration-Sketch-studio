"""SHARED Optuna study for the three Task 2 specialists (PDF p.5 allows this).

Each trial trains A_salt, A_blur and A_occlusion (each only on its own corruption)
with the SAME hyperparameters; the objective is the mean validation J of the three.
The final specialists are then trained independently with the best configuration
(scripts/train_task2_specialists.py).

    python scripts/optimize_task2_specialists.py --mode final
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from src.optimization.runner import run_study
from src.training.train_autoencoder import train_autoencoder
from src.utils.config import add_common_args, config_from_args


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args, "configs/task2_specialists.yaml")

    def train_fn(tcfg, trial):
        objs, attrs = [], {}
        epochs = int(tcfg["train"]["epochs"])
        for k, ctype in enumerate(tcfg["specialists"]):
            res = train_autoencoder(tcfg, allowed_types=[ctype], trial=trial, desc=f"T2-{ctype}-trial{trial.number}",
                                    step_offset=k * epochs)
            objs.append(res["best"]["objective"])
            attrs[f"{ctype}_val_objective"] = res["best"]["objective"]
            attrs[f"{ctype}_val_ssim"] = res["best"]["ssim"]
            attrs[f"{ctype}_val_psnr"] = res["best"]["psnr"]
        return float(np.mean(objs)), attrs

    m = cfg["model"]
    baseline = {"lr": cfg["train"]["lr"], "batch_size": 16 if cfg["mode"] == "smoke" else cfg["train"]["batch_size"],
                "bottleneck": "8x8x16", "base_channels": m["base_channels"],
                "alpha": cfg["loss"]["alpha"]}
    run_study(cfg, train_fn, baseline_params=baseline)


if __name__ == "__main__":
    main()
