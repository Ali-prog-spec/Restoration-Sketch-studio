"""Optuna study for the Task 3 soft MoE (joint lr, tau, CE weight, balance weight/type,
reconstruction weighting). Trials that show routing collapse are pruned.

    python scripts/optimize_task3.py --mode final
"""

import _bootstrap  # noqa: F401

import argparse

from src.optimization.runner import run_study
from src.training.train_moe import train_moe
from src.utils.config import add_common_args, config_from_args


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args, "configs/task3.yaml")

    def train_fn(tcfg, trial):
        tcfg["train"]["warmup_epochs"] = int(cfg["optuna"]["trial_warmup_epochs"])
        tcfg["train"]["joint_epochs"] = int(cfg["optuna"]["trial_joint_epochs"])
        res = train_moe(tcfg, trial=trial)
        b = res["best"]
        return b["objective"], {"val_ssim": b["ssim"], "val_psnr": b["psnr"], "val_route_acc": b["acc"],
                                "val_entropy": b["entropy"], "collapse": str(b["collapse"]),
                                "mean_weights": b["mean_weights"], "best_epoch": b["epoch"],
                                "initial_objective": res["initial"]["objective"]}

    L = cfg["loss"]
    baseline = {"lr_joint": cfg["train"]["lr_joint"], "tau": cfg["model"]["tau"], "ce_weight": L["ce"],
                "balance_weight": L["balance_weight"], "recon_l1": L["l1"], "balance": L["balance"]}
    run_study(cfg, train_fn, baseline_params=baseline)


if __name__ == "__main__":
    main()
