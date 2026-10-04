"""Run the complete experimental pipeline in the required order.

    python scripts/run_pipeline.py --mode smoke                    # ~3 min: verifies every script
    python scripts/run_pipeline.py --mode final                    # full experiments (hours on one GPU)
    python scripts/run_pipeline.py --mode final --stages task3 task4 export

Stages (each is a list of script invocations):
  task1   : Optuna -> final training with best params -> limited-skip ablation -> test evaluation
  task2   : classifier Optuna/train, specialists shared Optuna/train, hard-routing evaluation
  task3   : MoE Optuna -> final training -> balance-loss ablation -> evaluation
  task4   : GAN Optuna (short schedule) -> full retraining -> test evaluation
  export  : ONNX export + validation of all deployed models
  report  : figures/tables for the IEEE report

Every command's output is written to outputs/logs/<mode>/<stage>_<n>.log. The Optuna
studies are resumable, so re-running after an interruption continues where it stopped.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import os
import subprocess
import sys
import time

from src.utils.config import resolve


def stages(mode: str) -> dict:
    m = ["--mode", mode]
    smoke = mode == "smoke"
    other_balance = "entropy"
    best3 = resolve(f"outputs/{'smoke/' if smoke else ''}task3/optuna/{mode}/best_params.json")
    if best3.exists():
        other_balance = "squared" if json.loads(best3.read_text()).get("balance") == "entropy" else "entropy"
    return {
        "task1": [
            ["scripts/optimize_task1.py", *m],
            ["scripts/train_task1.py", *m, "--use-best"],
            ["scripts/train_task1.py", *m, "--use-best", "--skip-levels", "0", "--tag", "skip0"],
            ["scripts/evaluate_task1.py", *m],
            ["scripts/evaluate_task1.py", *m, "--tag", "skip0"],
        ],
        "task2": [
            ["scripts/optimize_task2_classifier.py", *m],
            ["scripts/train_task2_classifier.py", *m, "--use-best"],
            ["scripts/optimize_task2_specialists.py", *m],
            ["scripts/train_task2_specialists.py", *m, "--use-best"],
            ["scripts/evaluate_task2.py", *m],
        ],
        "task3": [
            ["scripts/optimize_task3.py", *m],
            ["scripts/train_task3.py", *m, "--use-best"],
            ["BALANCE_ABLATION"],  # resolved after the study exists (needs best balance type)
            ["scripts/evaluate_task3.py", *m] + (["--stress-images", "16"] if smoke else []),
            ["scripts/evaluate_task3.py", *m, "--tag", "balance_ablation"] + (["--stress-images", "16"] if smoke else []),
        ],
        "task4": [
            ["scripts/optimize_task4.py", *m],
            ["scripts/train_task4.py", *m, "--use-best"],
            ["scripts/evaluate_task4.py", *m, "--split", "val"],
            ["scripts/evaluate_task4.py", *m, "--split", "test"],
        ],
        "export": [["scripts/export_onnx.py", *m]],
        "report": [["scripts/make_report_artifacts.py", *m]],
    }, other_balance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="final", choices=["smoke", "dev", "final"])
    ap.add_argument("--stages", nargs="*", default=["task1", "task2", "task3", "task4", "export", "report"])
    args = ap.parse_args()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONUNBUFFERED": "1"}
    logdir = resolve(f"outputs/logs/{args.mode}")
    logdir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for stage in args.stages:
        cmds, _ = stages(args.mode)
        for i, cmd in enumerate(cmds[stage]):
            if cmd == ["BALANCE_ABLATION"]:
                _, other = stages(args.mode)  # re-read: the study has finished now
                cmd = ["scripts/train_task3.py", "--mode", args.mode, "--use-best", "--balance", other,
                       "--tag", "balance_ablation"]
            log = logdir / f"{stage}_{i}_{os.path.basename(cmd[0]).replace('.py', '')}.log"
            print(f"[{time.strftime('%H:%M:%S')}] {stage}: {' '.join(cmd)}  (log: {log.name})", flush=True)
            t0 = time.time()
            with open(log, "w", encoding="utf-8") as f:
                rc = subprocess.run([sys.executable, *cmd], cwd=resolve("."), env=env, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            print(f"    -> exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
            if rc != 0:
                print(f"FAILED — see {log}")
                sys.exit(rc)
    print(f"pipeline finished in {(time.time() - t_all) / 3600:.2f} h")


if __name__ == "__main__":
    main()
