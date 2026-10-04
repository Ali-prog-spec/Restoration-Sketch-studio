"""Live progress bar for the Task 4 part of the FINAL pipeline.

    python scripts/training_progress.py            # refreshes every 20 s, Ctrl+C to quit
    python scripts/training_progress.py --once     # print once

Reads the Optuna study (optuna_studies/task4.db), the training logs and the pipeline
driver log. Only reads files; it never interferes with the training.
"""

import _bootstrap  # noqa: F401

import argparse
import re
import time

from src.utils.config import resolve

LOGS = resolve("outputs/logs/final")
DRIVER = resolve("outputs/logs/final_pipeline_driver_part2.log")
N_TRIALS, TRIAL_EPOCHS, FINAL_EPOCHS = 20, 25, 200
STAGES = ["optimize_task4", "train_task4", "evaluate_task4.py --mode final --split val",
          "evaluate_task4.py --mode final --split test", "export_onnx", "make_report_artifacts"]
EPOCH_RE = re.compile(r"\[GAN\] epoch (\d+)/(\d+).*\((\d+(?:\.\d+)?)s\)")


def read(path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def last_epoch(text: str):
    m = EPOCH_RE.findall(text)
    return (int(m[-1][0]), int(m[-1][1]), float(m[-1][2])) if m else (0, 0, 20.0)


def finished_trials() -> int:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        st = optuna.load_study(study_name="task4_pix2pix_final",
                               storage=f"sqlite:///{resolve('optuna_studies/task4.db').as_posix()}")
    except Exception:  # noqa: BLE001
        return 0
    done = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(t.state in done for t in st.trials)


def bar(frac: float, width: int = 40) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {frac * 100:5.1f}%"


def status() -> str:
    driver = read(DRIVER)
    if "pipeline finished" in driver:
        return bar(1.0) + "\n\nTRAINING COMPLETE - all stages finished (Task 4, ONNX export, report)."
    if "FAILED" in driver:
        return "PIPELINE FAILED - see " + str(DRIVER) + "\n" + driver[-600:]
    started = [s for s in STAGES if s in driver]
    stage = started[-1] if started else "starting"

    trials = finished_trials()
    cur_ep, _, sec = last_epoch(read(LOGS / "task4_0_optimize_task4.log"))
    final_ep, _, fsec = last_epoch(read(LOGS / "task4_1_train_task4.log"))

    # work measured in epochs: search (20 x 25) + final training (200); the rest is minutes
    search_total = N_TRIALS * TRIAL_EPOCHS
    if stage == "optimize_task4":
        search_done = min(trials * TRIAL_EPOCHS + cur_ep, search_total)
        done, step = search_done, f"Hyper-parameter search: trial {min(trials + 1, N_TRIALS)}/{N_TRIALS}, epoch {cur_ep}/{TRIAL_EPOCHS}"
        sec_per_epoch = sec
    elif stage == "train_task4":
        done, step = search_total + final_ep, f"Final training: epoch {final_ep}/{FINAL_EPOCHS}"
        sec_per_epoch = fsec
    else:
        done, step = search_total + FINAL_EPOCHS, f"Finishing: {stage.split('.py')[0].split()[0]}"
        sec_per_epoch = 0
    total = search_total + FINAL_EPOCHS
    remaining_min = (total - done) * sec_per_epoch / 60 + 10  # +10 min for evaluation/export/report
    eta = time.strftime("%H:%M", time.localtime(time.time() + remaining_min * 60))
    return (f"{bar(done / total)}\n\n"
            f"Stage : {step}\n"
            f"Note  : pruned (poor) trials stop early, so the search may finish sooner\n"
            f"Left  : ~{remaining_min / 60:.1f} h  (estimated finish around {eta})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()
    while True:
        text = status()
        if not args.once:
            print("\033[2J\033[H", end="")  # clear screen
        print("Task 4 + export + report  (FINAL pipeline)\n")
        print(text)
        if args.once or "COMPLETE" in text or "FAILED" in text:
            break
        print(f"\n(refreshes every {args.every} s - Ctrl+C to quit; training keeps running)")
        time.sleep(args.every)


if __name__ == "__main__":
    main()
