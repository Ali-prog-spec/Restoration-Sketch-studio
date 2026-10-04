"""Shared restoration evaluation for Tasks 1-3 on the deterministic TEST manifest.

``run_restoration_eval(restore_fn, loader, device)`` returns a per-record DataFrame
with metrics of the restored output AND of the corrupted input (identity baseline),
plus any extra per-sample fields the restore_fn returns (routing decisions,
probabilities, mixture weights...).

Example selection is done from these real results:
  * representative: for every (condition, level) the record closest to that group's
    median objective J  -> clean (3 quantiles) + 9 corruption levels = 12 examples
  * failures: the worst records by J and the records with the largest *negative*
    improvement over the corrupted input (restoration made it worse), distinct images.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.corruptions import CLASS_NAMES
from src.evaluation.metrics import batch_metrics

PSNR_CAP_DB = 100.0  # an exactly lossless output (identity on a clean input) has infinite PSNR


@torch.no_grad()
def run_restoration_eval(restore_fn, loader, device, desc="eval") -> pd.DataFrame:
    rows = []
    for batch in tqdm(loader, desc=desc, leave=False):
        x_in = batch["input"].to(device, non_blocking=True)
        x = batch["target"].to(device, non_blocking=True)
        res = restore_fn(x_in, batch)
        out = res["output"].float()
        m_out = batch_metrics(out, x)
        m_in = batch_metrics(x_in, x)
        b = x.shape[0]
        extras = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                  for k, v in res.items() if k != "output"}
        for i in range(b):
            r = {
                "record": int(batch["record"][i]),
                "image_id": batch["image_id"][i],
                "type": CLASS_NAMES[int(batch["label"][i])],
                "level": batch["level"][i],
                "severity": float(batch["severity"][i]),
                "l1": float(m_out["l1"][i]),
                "mse": float(m_out["mse"][i]),
                "psnr": min(float(m_out["psnr"][i]), PSNR_CAP_DB),
                "ssim": float(m_out["ssim"][i]),
                "input_l1": float(m_in["l1"][i]),
                "input_psnr": min(float(m_in["psnr"][i]), PSNR_CAP_DB),
                "input_ssim": float(m_in["ssim"][i]),
            }
            for k, v in extras.items():
                val = v[i]
                if np.ndim(val) == 0:
                    r[k] = val.item() if hasattr(val, "item") else val
                else:
                    for j, vv in enumerate(np.ravel(val)):
                        r[f"{k}_{j}"] = float(vv)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["objective"] = 0.5 * df["l1"] + 0.5 * (1 - df["ssim"])
    df["input_objective"] = 0.5 * df["input_l1"] + 0.5 * (1 - df["input_ssim"])
    df["ssim_gain"] = df["ssim"] - df["input_ssim"]
    df["psnr_gain"] = df["psnr"] - df["input_psnr"]
    return df


METRIC_COLS = ["l1", "psnr", "ssim", "input_l1", "input_psnr", "input_ssim", "ssim_gain", "psnr_gain"]


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(by condition, by condition x level, by level across corruptions)."""
    by_type = df.groupby("type", sort=False)[METRIC_COLS].mean()
    by_type["n"] = df.groupby("type", sort=False).size()
    by_type = by_type.reindex([c for c in CLASS_NAMES if c in by_type.index])
    overall = df[METRIC_COLS].mean().to_frame("overall").T
    overall["n"] = len(df)
    by_type = pd.concat([by_type, overall])

    by_level = df.groupby(["type", "level"], sort=False)[METRIC_COLS].mean()
    by_level["n"] = df.groupby(["type", "level"], sort=False).size()
    order = [(c, lv) for c in CLASS_NAMES for lv in ["none", "low", "medium", "high"]]
    by_level = by_level.reindex([o for o in order if o in by_level.index])

    corr = df[df["type"] != "clean"]
    by_sev = corr.groupby("level")[METRIC_COLS].mean().reindex(["low", "medium", "high"])
    by_sev["n"] = corr.groupby("level").size()
    return by_type, by_level, by_sev


def select_representative(df: pd.DataFrame) -> list[int]:
    picks = []
    clean = df[df["type"] == "clean"].sort_values("objective")
    if len(clean):
        for q in (0.25, 0.5, 0.75):
            picks.append(int(clean.iloc[int(q * (len(clean) - 1))]["record"]))
    for t in CLASS_NAMES[1:]:
        for lv in ("low", "medium", "high"):
            g = df[(df["type"] == t) & (df["level"] == lv)]
            if not len(g):
                continue
            med = g["objective"].median()
            picks.append(int(g.iloc[(g["objective"] - med).abs().argsort().iloc[0]]["record"]))
    return picks


def select_failures(df: pd.DataFrame, n_worst: int = 4, n_regressions: int = 4) -> pd.DataFrame:
    """The worst output of each corruption type (distinct images) plus the cases where
    restoration most DEGRADED the input (negative SSIM gain) — at least 4 cases."""
    chosen, used = [], set()
    worst = df.sort_values("objective", ascending=False)
    per_type_seen = set()
    for _, r in worst.iterrows():  # first pass: one per corruption type
        if r["image_id"] in used or r["type"] in per_type_seen or r["type"] == "clean":
            continue
        chosen.append({**r.to_dict(), "failure_kind": "worst_objective"})
        used.add(r["image_id"])
        per_type_seen.add(r["type"])
        if len(chosen) >= n_worst:
            break
    regress = df[df["ssim_gain"] < 0].sort_values("ssim_gain")
    k = 0
    for _, r in regress.iterrows():
        if r["image_id"] in used:
            continue
        chosen.append({**r.to_dict(), "failure_kind": "degraded_input"})
        used.add(r["image_id"])
        k += 1
        if k >= n_regressions:
            break
    return pd.DataFrame(chosen)
