"""Optuna study for the Task 2 corruption classifier.

    python scripts/optimize_task2_classifier.py --mode final
"""

import _bootstrap  # noqa: F401

import argparse

from src.optimization.runner import run_study
from src.training.train_classifier import train_classifier
from src.utils.config import add_common_args, config_from_args


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args, "configs/task2_classifier.yaml")

    def train_fn(tcfg, trial):
        res = train_classifier(tcfg, trial=trial)
        b = res["best"]
        return b["objective"], {"val_accuracy": b["accuracy"], "val_macro_f1": b["macro_f1"],
                                "val_ce": b["ce"], "best_epoch": b["epoch"]}

    baseline = {"lr": cfg["train"]["lr"], "batch_size": cfg["train"]["batch_size"],
                "base_channels": cfg["model"]["base_channels"], "channel_plan": "standard",
                "dropout": cfg["model"]["dropout"], "weight_decay": cfg["train"]["weight_decay"]}
    if cfg["mode"] == "smoke":
        baseline["batch_size"] = 16
    run_study(cfg, train_fn, baseline_params=baseline)


if __name__ == "__main__":
    main()
