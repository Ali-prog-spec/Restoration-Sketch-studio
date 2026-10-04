"""Run a resumable Optuna study for any task.

* The study name and output folder are suffixed with the mode (smoke/dev/final) so
  quick experiments never pollute the final study.
* The first trial is the assignment's baseline configuration (e.g. alpha = 0.8),
  enqueued explicitly, so the report can compare "PDF baseline" vs "Optuna best".
* ``n_trials`` is the TOTAL number of finished trials wanted; rerunning the script
  resumes the SQLite study and only runs the missing trials.
"""

from __future__ import annotations

import copy
import time

import optuna

from src.optimization.search import apply_params, make_study, save_study_artifacts, suggest
from src.utils.config import flatten
from src.utils.env_info import environment_info
from src.utils.tracking import Tracker


def study_dir(cfg: dict) -> str:
    return f"{cfg['output']['optuna_dir']}/{cfg['mode']}"


def run_study(cfg: dict, train_fn, baseline_params: dict | None = None, direction: str = "minimize",
              trial_overrides: dict | None = None) -> dict:
    """train_fn(trial_cfg, trial) -> (objective_value, user_attrs dict)."""
    ocfg = copy.deepcopy(cfg["optuna"])
    ocfg["study_name"] = f"{ocfg['study_name']}_{cfg['mode']}"
    space = ocfg["search_space"]
    study = make_study(ocfg, direction)
    if baseline_params and len(study.trials) == 0:
        study.enqueue_trial(baseline_params, user_attrs={"baseline": True})

    tracker = Tracker(cfg)
    finished = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    remaining = max(int(ocfg["n_trials"]) - len(finished), 0)
    print(f"[optuna] study {ocfg['study_name']}: {len(finished)} finished trials, running {remaining} more")

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        tcfg = apply_params(cfg, params, space)
        tcfg["train"]["epochs"] = int(ocfg["epochs"])
        tcfg["progress"] = False
        for k, v in (trial_overrides or {}).items():
            tcfg[k] = v
        t0 = time.perf_counter()
        with tracker.run(f"{ocfg['study_name']}-trial{trial.number}", nested=True, tags={"optuna_trial": str(trial.number)}):
            tracker.log_params({"trial": params})
            try:
                value, attrs = train_fn(tcfg, trial)
            except optuna.TrialPruned:
                tracker.set_tags({"state": "pruned"})
                raise
            tracker.log_metrics({"objective": value})
        for k, v in (attrs or {}).items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("duration_s", round(time.perf_counter() - t0, 1))
        return value

    with tracker.run(f"{ocfg['study_name']}", tags={"type": "optuna_study"}):
        tracker.log_params({"optuna": {k: v for k, v in ocfg.items() if k != "search_space"},
                            "search_space": space, "base_config": flatten(cfg)})
        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        summary = save_study_artifacts(study, space, study_dir(cfg),
                                       extra={"mode": cfg["mode"], "trial_epochs": ocfg["epochs"],
                                              "environment": environment_info()})
        if "best_value" in summary:
            tracker.log_metrics({"best_objective": summary["best_value"]})
            tracker.log_params({"best": summary["best_params"]})
        tracker.log_artifacts(str(study_dir_path(cfg)), "optuna")
    print(f"[optuna] best value {summary.get('best_value')} params {summary.get('best_params')}")
    return summary


def study_dir_path(cfg: dict):
    from src.utils.config import resolve

    return resolve(study_dir(cfg))


def config_with_best(cfg: dict, best_mode: str | None = None) -> tuple[dict, dict | None]:
    """Merge the best Optuna params (from ``best_mode``'s study, default: cfg mode) into cfg."""
    from src.optimization.search import load_best_params

    mode = best_mode or cfg["mode"]
    params = load_best_params(f"{cfg['output']['optuna_dir']}/{mode}")
    if params is None:
        return cfg, None
    return apply_params(cfg, params, cfg["optuna"]["search_space"]), params
