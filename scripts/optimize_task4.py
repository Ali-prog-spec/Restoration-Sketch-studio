"""Optuna study for the Task 4 style-conditioned pix2pix (shortened trial schedule).

    python scripts/optimize_task4.py --mode final       # 20 trials x 25 epochs
"""

import _bootstrap  # noqa: F401

import argparse

from src.optimization.runner import run_study
from src.training.train_gan import train_gan
from src.utils.config import add_common_args, config_from_args


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args, "configs/task4.yaml")

    def train_fn(tcfg, trial):
        res = train_gan(tcfg, trial=trial)
        b = res["best"]
        return b["objective"], {"val_l1": b["l1"], "val_ssim": b["ssim"], "val_psnr": b["psnr"], "best_epoch": b["epoch"]}

    m, t = cfg["model"], cfg["train"]
    baseline = {"lr_g": t["lr_g"], "lr_d": t["lr_d"], "batch_size": 4 if cfg["mode"] == "smoke" else t["batch_size"],
                "base_channels": m["g_channels"], "dropout": m["dropout"], "style_dim": m["style_dim"],
                "lambda_l1": cfg["loss"]["lambda_l1"]}
    run_study(cfg, train_fn, baseline_params=baseline)


if __name__ == "__main__":
    main()
