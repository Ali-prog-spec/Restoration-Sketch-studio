"""Final TEST evaluation of the Task 1 universal DAE (run once, after model selection).

    python scripts/evaluate_task1.py --mode final
    python scripts/evaluate_task1.py --mode smoke

Outputs (outputs/task1/):
  per_record.csv, metrics_by_type.csv, metrics_by_level.csv, metrics_by_severity.csv
  figures/representative_*.png (12 examples), figures/failures_*.png, figures/severity_*.png
  report tables in report/tables/task1_*.tex
"""

import _bootstrap  # noqa: F401

import argparse

import torch

from src.evaluation.plots import severity_plot
from src.evaluation.reporting import report_dir, example_figures, latex_table, save_tables
from src.evaluation.restoration import run_restoration_eval, select_failures, select_representative, summarize
from src.models import DenoisingAutoencoder
from src.training.common import autocast, load_checkpoint, test_loader
from src.utils.config import add_common_args, config_from_args, resolve, save_json
from src.utils.env_info import get_device
from src.utils.tracking import Tracker


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task1.yaml")
    device = get_device(cfg.get("device"))
    tag = f"_{args.tag}" if args.tag else ""
    ckpt_path = args.checkpoint or cfg["output"]["checkpoint"].replace(".pt", f"{tag}.pt")
    ck = load_checkpoint(ckpt_path)
    model = DenoisingAutoencoder(**ck["model_config"]).to(device).eval()
    model.load_state_dict(ck["model_state"])
    amp = cfg["train"].get("amp", True)

    def restore(x, batch=None):
        with autocast(device, amp):
            return {"output": model(x).float()}

    loader = test_loader(cfg, batch_size=128)
    df = run_restoration_eval(restore, loader, device, desc="Task1 test")
    out = cfg["output"]["dir"]
    df.to_csv(resolve(f"{out}/per_record{tag}.csv"), index=False)
    by_type, by_level, by_sev = summarize(df)
    save_tables({f"metrics_by_type{tag}": by_type, f"metrics_by_level{tag}": by_level,
                 f"metrics_by_severity{tag}": by_sev}, out)

    cols = ["l1", "psnr", "ssim", "input_psnr", "input_ssim", "n"]
    fmt = {"psnr": "{:.2f}", "input_psnr": "{:.2f}", "n": "{:.0f}"}
    latex_table(by_type, f"{report_dir(cfg)}/task1_by_type{tag}.tex", "Task 1 test results per condition "
                "(input\\_* = corrupted input vs. clean target).", f"tab:task1_type{tag}", cols, fmt)
    latex_table(by_level, f"{report_dir(cfg)}/task1_by_level{tag}.tex", "Task 1 test results per condition "
                "and severity.", f"tab:task1_level{tag}", cols, fmt)

    ds = loader.dataset

    @torch.no_grad()
    def restore_one(x):
        return model(x[None].to(device)).float()[0].cpu()

    reps = select_representative(df)
    example_figures(ds, df, reps, restore_one, f"{out}/figures/representative{tag}",
                    "Task 1: representative test examples (median J per condition/level)")
    fails = select_failures(df)
    fails.to_csv(resolve(f"{out}/failure_cases{tag}.csv"), index=False)
    example_figures(ds, df, [int(r) for r in fails["record"]], restore_one, f"{out}/figures/failures{tag}",
                    "Task 1: failure cases", extra_caption=lambda r: "")
    severity_plot(by_level, f"{out}/figures/severity_ssim{tag}.png", "ssim", "Task 1 SSIM by severity")
    severity_plot(by_level, f"{out}/figures/severity_psnr{tag}.png", "psnr", "Task 1 PSNR by severity")

    save_json({"checkpoint": ckpt_path, "n_records": len(df), "representative_records": reps,
               "failure_records": fails["record"].tolist(), "overall": by_type.loc["overall"].to_dict()},
              f"{out}/test_summary{tag}.json")
    tracker = Tracker(cfg)
    with tracker.run(f"task1-test-eval-{cfg['mode']}{tag}", tags={"task": "task1", "split": "test"}):
        tracker.log_metrics({f"test_{k}": v for k, v in by_type.loc["overall"].items()})
        for t, row in by_type.iterrows():
            tracker.log_metrics({f"test_{t}_ssim": row["ssim"], f"test_{t}_psnr": row["psnr"], f"test_{t}_l1": row["l1"]})
        tracker.log_artifacts(resolve(f"{out}/figures"), "figures")
        for f in ["per_record", "metrics_by_type", "metrics_by_level", "metrics_by_severity"]:
            tracker.log_artifact(resolve(f"{out}/{f}{tag}.csv"), "tables")
    print(by_type.round(4).to_string())
    print(by_level.round(4).to_string())


if __name__ == "__main__":
    main()
