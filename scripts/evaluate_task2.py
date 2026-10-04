"""Final TEST evaluation of Task 2: classifier + hard-routed specialists.

    python scripts/evaluate_task2.py --mode final

1. Classifier: accuracy, macro precision/recall/F1, per-class metrics, normalised
   confusion matrix (overall and per severity).
2. Hard routing in two modes on the SAME test manifest:
     oracle    — route = true corruption label from the manifest
     predicted — route = argmax of the classifier probabilities
   Clean routes use the identity bypass (no expert is run).
3. Routing-failure analysis: records where the classifier was wrong AND the
   predicted-routing output is worse than the oracle output.
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np
import pandas as pd
import torch

from src.corruptions.constants import CLASS_DISPLAY, CLASS_NAMES
from src.evaluation.classification import classification_report
from src.evaluation.plots import compare_systems_plot, confusion_matrix_plot, image_row_grid
from src.evaluation.reporting import report_dir, example_figures, latex_table, save_tables
from src.evaluation.restoration import run_restoration_eval, select_failures, select_representative, summarize
from src.models import hard_route
from src.models.loading import load_classifier, load_experts
from src.training.common import autocast, test_loader
from src.utils.config import add_common_args, config_from_args, load_config, resolve, save_json
from src.utils.env_info import get_device
from src.utils.tracking import Tracker


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--degradation-threshold", type=float, default=0.01,
                    help="J(predicted) - J(oracle) above which a misrouting counts as a restoration failure")
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task2_specialists.yaml")
    ccfg = load_config("configs/task2_classifier.yaml", mode=cfg["mode"])
    device = get_device(cfg.get("device"))
    amp = True
    clf = load_classifier(ccfg["output"]["checkpoint"]).to(device)
    experts = [e.to(device) for e in load_experts(cfg["output"]["checkpoint_dir"])]
    out = "outputs/task2" if cfg["mode"] != "smoke" else "outputs/smoke/task2"
    loader = test_loader(cfg, batch_size=128)
    ds = loader.dataset

    # ---------------- restoration functions ----------------
    @torch.no_grad()
    def predicted_restore(x, batch):
        with autocast(device, amp):
            probs = torch.softmax(clf(x).float(), dim=1)
            pred = probs.argmax(1)
            y = hard_route(x, pred, experts)
        return {"output": y.float(), "pred": pred, "probs": probs}

    @torch.no_grad()
    def oracle_restore(x, batch):
        route = batch["label"].to(device)
        with autocast(device, amp):
            y = hard_route(x, route, experts)
        return {"output": y.float()}

    df_pred = run_restoration_eval(predicted_restore, loader, device, desc="predicted routing")
    df_orc = run_restoration_eval(oracle_restore, loader, device, desc="oracle routing")
    df_pred["pred_name"] = [CLASS_NAMES[int(p)] for p in df_pred["pred"]]
    df_pred.to_csv(resolve(f"{out}/per_record_predicted.csv"), index=False)
    df_orc.to_csv(resolve(f"{out}/per_record_oracle.csv"), index=False)

    # ---------------- 1. classifier metrics ----------------
    y_true = np.array([CLASS_NAMES.index(t) for t in df_pred["type"]])
    y_pred = df_pred["pred"].to_numpy().astype(int)
    rep = classification_report(y_true, y_pred)
    rep["per_class"].to_csv(resolve(f"{out}/classifier_per_class.csv"))
    pd.DataFrame(rep["confusion_normalized"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        resolve(f"{out}/classifier_confusion_normalized.csv"))
    pd.DataFrame(rep["confusion"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(resolve(f"{out}/classifier_confusion_counts.csv"))
    confusion_matrix_plot(rep["confusion_normalized"], f"{out}/figures/confusion_matrix.png",
                          "Task 2 classifier — normalised confusion (test)")
    # accuracy per condition x severity
    acc_level = df_pred.assign(correct=(y_true == y_pred)).groupby(["type", "level"], sort=False)["correct"].mean()
    acc_level.to_csv(resolve(f"{out}/classifier_accuracy_by_level.csv"))
    cls_summary = {k: rep[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    latex_table(rep["per_class"], f"{report_dir(cfg)}/task2_classifier_per_class.tex",
                f"Task 2 classifier on the test manifest (accuracy {cls_summary['accuracy']:.4f}, macro-F1 "
                f"{cls_summary['macro_f1']:.4f}).", "tab:task2_cls", ["precision", "recall", "f1", "support"],
                {"support": "{:.0f}"}, index_names="Class")

    # ---------------- 2. oracle vs predicted ----------------
    t_pred, l_pred, s_pred = summarize(df_pred)
    t_orc, l_orc, s_orc = summarize(df_orc)
    save_tables({"routing_predicted_by_type": t_pred, "routing_predicted_by_level": l_pred,
                 "routing_oracle_by_type": t_orc, "routing_oracle_by_level": l_orc}, out)
    comp = pd.DataFrame({
        "oracle_psnr": l_orc["psnr"], "pred_psnr": l_pred["psnr"],
        "oracle_ssim": l_orc["ssim"], "pred_ssim": l_pred["ssim"],
        "input_ssim": l_orc["input_ssim"],
        "cls_accuracy": acc_level,
    })
    comp.to_csv(resolve(f"{out}/oracle_vs_predicted_by_level.csv"))
    latex_table(comp, f"{report_dir(cfg)}/task2_oracle_vs_predicted.tex",
                "Task 2 hard routing on the test manifest: oracle vs. predicted routing.",
                "tab:task2_routing", list(comp.columns),
                {"oracle_psnr": "{:.2f}", "pred_psnr": "{:.2f}"})
    compare_systems_plot({"oracle routing": l_orc, "predicted routing": l_pred},
                         f"{out}/figures/oracle_vs_predicted_ssim.png", "ssim", "Task 2: oracle vs predicted routing")

    # ---------------- 3. routing failures caused by the classifier ----------------
    merged = df_pred.merge(df_orc[["record", "objective", "ssim", "psnr"]], on="record", suffixes=("", "_oracle"))
    merged["misrouted"] = merged["type"] != merged["pred_name"]
    merged["degradation"] = merged["objective"] - merged["objective_oracle"]
    caused = merged[merged["misrouted"] & (merged["degradation"] > args.degradation_threshold)]
    caused = caused.sort_values("degradation", ascending=False)
    caused.to_csv(resolve(f"{out}/classifier_caused_failures.csv"), index=False)
    mis_summary = merged.groupby(["type", "level"], sort=False).agg(
        misrouted_rate=("misrouted", "mean"),
        mean_degradation_if_misrouted=("degradation", lambda s: s[merged.loc[s.index, "misrouted"]].mean()))
    mis_summary.to_csv(resolve(f"{out}/misrouting_summary.csv"))

    # figure: target | input | oracle | predicted for the worst classifier-caused failures
    rows, caps = [], []
    for _, r in caused.drop_duplicates("image_id").head(8).iterrows():
        item = ds[int(r["record"])]
        x = item["input"][None].to(device)
        with torch.no_grad():
            orc = hard_route(x, torch.tensor([CLASS_NAMES.index(r["type"])], device=device), experts)[0].float().cpu()
            prd = hard_route(x, torch.tensor([int(r["pred"])], device=device), experts)[0].float().cpu()
        probs = ", ".join(f"{p:.2f}" for p in [r[f"probs_{k}"] for k in range(4)])
        rows.append([item["target"], item["input"], orc, prd])
        caps.append(f"true {r['type']} ({r['level']})\npred {r['pred_name']}\np=[{probs}]\n"
                    f"SSIM {r['ssim_oracle']:.3f}→{r['ssim']:.3f}")
    if rows:
        image_row_grid(rows, ["Clean target", "Input", "Oracle route", "Predicted route"],
                       f"{out}/figures/routing_failures.png", caps, title="Task 2: failures caused by classifier errors")

    # representative + general failure grids for the operational (predicted) system
    @torch.no_grad()
    def restore_one(x):
        return predicted_restore(x[None].to(device), None)["output"][0].cpu()

    example_figures(ds, df_pred, select_representative(df_pred), restore_one, f"{out}/figures/representative",
                    "Task 2 (predicted routing): representative test examples",
                    extra_caption=lambda r: f"routed: {CLASS_DISPLAY[r['pred_name']].split()[0]}")
    fails = select_failures(df_pred)
    fails.to_csv(resolve(f"{out}/failure_cases.csv"), index=False)
    example_figures(ds, df_pred, [int(r) for r in fails["record"]], restore_one, f"{out}/figures/failures",
                    "Task 2 (predicted routing): failure cases",
                    extra_caption=lambda r: f"routed: {CLASS_DISPLAY[r['pred_name']].split()[0]}")

    summary = {"classifier": cls_summary,
               "oracle_overall": t_orc.loc["overall"].to_dict(), "predicted_overall": t_pred.loc["overall"].to_dict(),
               "n_misrouted": int(merged["misrouted"].sum()), "n_classifier_caused_failures": int(len(caused)),
               "degradation_threshold": args.degradation_threshold}
    save_json(summary, f"{out}/test_summary.json")

    tracker = Tracker(cfg)
    with tracker.run(f"task2-test-eval-{cfg['mode']}", tags={"task": "task2", "split": "test"}):
        tracker.log_metrics({f"cls_{k}": v for k, v in cls_summary.items()})
        tracker.log_metrics({"oracle_ssim": t_orc.loc["overall", "ssim"], "pred_ssim": t_pred.loc["overall", "ssim"],
                             "oracle_psnr": t_orc.loc["overall", "psnr"], "pred_psnr": t_pred.loc["overall", "psnr"]})
        tracker.log_artifacts(resolve(f"{out}/figures"), "figures")
    print(summary)
    print(comp.round(4).to_string())


if __name__ == "__main__":
    main()
