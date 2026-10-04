"""Train the three Task 2 specialists independently (shared best architecture).

    python scripts/train_task2_specialists.py --mode final --use-best
    python scripts/train_task2_specialists.py --mode final --use-best --only blur
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
    ap.add_argument("--use-best", action="store_true")
    ap.add_argument("--best-from", default=None)
    ap.add_argument("--only", choices=["salt_pepper", "blur", "occlusion"], default=None)
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task2_specialists.yaml")
    best = None
    if args.use_best:
        cfg, best = config_with_best(cfg, args.best_from)
        if best is None:
            raise SystemExit("no best_params.json — run scripts/optimize_task2_specialists.py first")
    out = cfg["output"]["dir"]
    tracker = Tracker(cfg)
    summary = {}
    for ctype in ([args.only] if args.only else cfg["specialists"]):
        ckpt = f"{cfg['output']['checkpoint_dir']}/{ctype}_expert.pt"
        with tracker.run(f"task2-specialist-{ctype}-{cfg['mode']}", tags={"task": "task2", "part": "specialist",
                                                                           "corruption": ctype}):
            tracker.log_params(cfg)
            tracker.log_params({"env": environment_info(), "optuna_best": best or "none", "trained_on": ctype})
            res = train_autoencoder(cfg, allowed_types=[ctype], tracker=tracker, checkpoint_path=ckpt,
                                    desc=f"A_{ctype}", history_path=f"{out}/history_{ctype}.csv")
            fig = training_curves(f"{out}/history_{ctype}.csv", f"{out}/figures/training_curves_{ctype}.png",
                                  title=f"Task 2 specialist: {ctype}")
            tracker.log_artifact(fig)
            tracker.log_artifact(resolve(ckpt), "checkpoints")
            summary[ctype] = {"best": res["best"], "checkpoint": ckpt}
    save_json({"specialists": summary, "config": cfg, "optuna_best_params": best,
               "environment": environment_info()}, f"{out}/train_summary{'_' + args.only if args.only else ''}.json")
    print(summary)


if __name__ == "__main__":
    main()
