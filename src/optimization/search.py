"""Generic Optuna helpers driven by the ``optuna.search_space`` section of a config.

A search-space entry looks like

    lr:          {type: float, low: 1.0e-4, high: 3.0e-3, log: true, target: train.lr}
    batch_size:  {type: categorical, choices: [16, 32, 64], target: train.batch_size}

``target`` is the dotted config key that the sampled value overrides. Keeping the
space in YAML means the complete search space is saved with the study and can be
printed in the report exactly as it was searched.

Sampler: TPE (Bergstra et al., 2011) seeded with 42 -> reproducible suggestions.
Pruner: median pruning on the per-epoch validation objective (configurable), which
stops clearly poor trials early to save compute.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import optuna
import pandas as pd

from src.utils.config import resolve, save_json


def suggest(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif t == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"unknown search-space type {t}")
    return params


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_params(cfg: dict, params: dict, space: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for name, value in params.items():
        spec = space.get(name, {})
        # a categorical param may be decoded via 'map': name -> value, or name -> {dotted.key: value}
        if "map" in spec:
            value = spec["map"][value] if not isinstance(value, list) else value
            if isinstance(value, dict):
                for k, v in value.items():
                    set_dotted(cfg, k, v)
                continue
        for target in ([spec["target"]] if isinstance(spec.get("target"), str) else spec.get("target", [])):
            set_dotted(cfg, target, value)
        # e.g. reconstruction weighting r: loss.l1 = r and loss.ssim = 1 - r
        if "complement_target" in spec:
            set_dotted(cfg, spec["complement_target"], 1.0 - float(value))
    return cfg


def make_study(ocfg: dict, direction: str = "minimize") -> optuna.Study:
    storage = ocfg.get("storage")
    if storage and storage.startswith("sqlite:///"):
        db = resolve(storage[len("sqlite:///"):])
        db.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db.as_posix()}"
    pruner_kind = ocfg.get("pruner", "median")
    if pruner_kind == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=int(ocfg.get("pruner_startup_trials", 3)),
                                             n_warmup_steps=int(ocfg.get("pruner_warmup_steps", 2)))
    elif pruner_kind == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()
    return optuna.create_study(
        study_name=ocfg["study_name"], storage=storage, load_if_exists=True, direction=direction,
        sampler=optuna.samplers.TPESampler(seed=int(ocfg.get("seed", 42))), pruner=pruner,
    )


def save_study_artifacts(study: optuna.Study, space: dict, out_dir, extra: dict | None = None) -> dict:
    """Write trial history, best trial, search space and Optuna plots to out_dir."""
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(out / "trials.csv", index=False)
    states = df["state"].value_counts().to_dict() if len(df) else {}
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials_total": len(study.trials),
        "trial_states": states,
        "search_space": space,
        **(extra or {}),
    }
    if complete:
        best = study.best_trial
        summary.update({"best_trial_number": best.number, "best_value": best.value,
                        "best_params": best.params, "best_user_attrs": best.user_attrs})
    save_json(summary, out / "study_summary.json")
    if complete:
        save_json(study.best_trial.params, out / "best_params.json")
    _plots(study, out)
    return summary


def _plots(study: optuna.Study, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return
    try:
        ax = ovm.plot_optimization_history(study)
        ax.figure.savefig(out / "optimization_history.png", bbox_inches="tight", dpi=150)
        plt.close(ax.figure)
    except Exception as exc:  # noqa: BLE001
        print(f"[optuna] history plot failed: {exc}")
    if len(complete) >= 2:
        try:
            ax = ovm.plot_param_importances(study)
            ax.figure.savefig(out / "param_importances.png", bbox_inches="tight", dpi=150)
            plt.close(ax.figure)
        except Exception as exc:  # noqa: BLE001
            print(f"[optuna] importance plot failed: {exc}")
        try:
            ax = ovm.plot_intermediate_values(study)
            ax.figure.savefig(out / "intermediate_values.png", bbox_inches="tight", dpi=150)
            plt.close(ax.figure)
        except Exception as exc:  # noqa: BLE001
            print(f"[optuna] intermediate plot failed: {exc}")


def load_best_params(out_dir) -> dict | None:
    p = resolve(out_dir) / "best_params.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def trials_table(out_dir) -> pd.DataFrame:
    return pd.read_csv(resolve(out_dir) / "trials.csv")
