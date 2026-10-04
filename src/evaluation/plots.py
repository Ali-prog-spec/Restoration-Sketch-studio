"""Publication-style figures (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.corruptions.constants import CLASS_DISPLAY, CLASS_NAMES  # noqa: E402
from src.utils.config import resolve  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

BRANCH_NAMES = ["Identity", "Salt expert", "Blur expert", "Occlusion expert"]
LEVEL_ORDER = ["none", "low", "medium", "high"]


def _save(fig, path) -> Path:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def to_hwc(t) -> np.ndarray:
    a = t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return np.clip(a, 0, 1)


# ----------------------------------------------------------------------------
def training_curves(history_csv, out, pairs=None, title=""):
    """pairs: list of (train_col, val_col, label)."""
    df = pd.read_csv(resolve(history_csv))
    pairs = pairs or [("train_loss", "val_loss", "Loss"), ("train_l1", "val_l1", "L1"), ("train_ssim", "val_ssim", "SSIM")]
    pairs = [p for p in pairs if p[0] in df or p[1] in df]
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs), 2.6))
    axes = np.atleast_1d(axes)
    for ax, (tr, va, label) in zip(axes, pairs):
        if tr in df:
            ax.plot(df["epoch"], df[tr], label="train")
        if va in df:
            ax.plot(df["epoch"], df[va], label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.legend(frameon=False)
    fig.suptitle(title)
    return _save(fig, out)


def restoration_grid(examples: list[dict], out, title="", error_scale: float = 0.5):
    """examples: dicts with keys target, input, output (CHW tensors in [0,1]) and caption.
    Columns: clean target | corrupted input | reconstruction | |x - x^| error map."""
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(6.4, 1.65 * n))
    axes = np.atleast_2d(axes)
    for i, ex in enumerate(examples):
        tgt, inp, rec = to_hwc(ex["target"]), to_hwc(ex["input"]), to_hwc(ex["output"])
        err = np.abs(tgt - rec).mean(axis=2)
        for j, (img, name) in enumerate([(tgt, "Clean target"), (inp, "Corrupted input"),
                                         (rec, "Reconstruction"), (err, "|x - x̂|")]):
            ax = axes[i, j]
            if j == 3:
                im = ax.imshow(img, cmap="inferno", vmin=0, vmax=error_scale)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=8)
        axes[i, 0].set_ylabel(ex.get("caption", ""), fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.89, 0.35, 0.015, 0.3])
    fig.colorbar(im, cax=cax, label="abs. error (mean over RGB)")
    fig.suptitle(title, fontsize=9)
    return _save(fig, out)


def severity_plot(by_level: pd.DataFrame, out, metric="ssim", title=""):
    """Grouped bars: input vs output metric for each corruption and severity."""
    df = by_level.reset_index()
    df = df[df["type"] != "clean"]
    labels = [f"{CLASS_DISPLAY[t].split()[0]}\n{lv}" for t, lv in zip(df["type"], df["level"])]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.bar(x - 0.2, df[f"input_{metric}"], 0.4, label="corrupted input", color="#bbbbbb")
    ax.bar(x + 0.2, df[metric], 0.4, label="restored", color="#3b6fb6")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel(metric.upper())
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(title)
    return _save(fig, out)


def compare_systems_plot(tables: dict[str, pd.DataFrame], out, metric="ssim", title=""):
    """tables: system name -> by_level DataFrame. Line plot per condition/level."""
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    first = next(iter(tables.values())).reset_index()
    labels = [f"{t[:5]}-{lv}" for t, lv in zip(first["type"], first["level"])]
    for name, t in tables.items():
        d = t.reset_index()
        ax.plot(range(len(d)), d[metric], marker="o", ms=3, label=name)
    ax.plot(range(len(first)), first[f"input_{metric}"], ls="--", color="grey", label="input")
    ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=7)
    ax.set_ylabel(metric.upper())
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(title)
    return _save(fig, out)


def confusion_matrix_plot(cm_norm: np.ndarray, out, title="Normalised confusion matrix"):
    names = [CLASS_DISPLAY[c] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(4), names, rotation=30, ha="right")
    ax.set_yticks(range(4), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, fraction=0.046)
    ax.set_title(title)
    return _save(fig, out)


def routing_heatmap(mean_weights: pd.DataFrame, out, title="Average routing weights"):
    """mean_weights: rows = (type, level) labels, cols = 4 branches."""
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(mean_weights) + 1.2))
    im = ax.imshow(mean_weights.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(mean_weights.shape[0]):
        for j in range(mean_weights.shape[1]):
            v = mean_weights.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.6 else "black", fontsize=7)
    ax.set_xticks(range(4), BRANCH_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(len(mean_weights)), [str(i) for i in mean_weights.index], fontsize=7)
    fig.colorbar(im, fraction=0.046, label="mean weight")
    ax.set_title(title)
    return _save(fig, out)


def weight_distribution_plot(df: pd.DataFrame, out, weight_prefix="weights_"):
    """Box plots of each branch weight, grouped by the true condition."""
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.4), sharey=True)
    for k, ax in enumerate(axes):
        data = [df.loc[df["type"] == c, f"{weight_prefix}{k}"].values for c in CLASS_NAMES]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, 5), ["clean", "salt", "blur", "occ"], fontsize=7)
        ax.set_title(BRANCH_NAMES[k], fontsize=8)
        if k == 0:
            ax.set_ylabel("routing weight")
    fig.suptitle("Routing-weight distribution by true condition", fontsize=9)
    return _save(fig, out)


def utilization_plot(util: pd.DataFrame, out):
    """util: index = branches, columns mean_weight and argmax_share."""
    x = np.arange(len(util))
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.bar(x - 0.2, util["mean_weight"], 0.4, label="mean weight")
    ax.bar(x + 0.2, util["argmax_share"], 0.4, label="share of inputs where dominant")
    ax.axhline(0.25, ls="--", color="grey", lw=0.8)
    ax.set_xticks(x, BRANCH_NAMES, rotation=15, fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Expert utilisation")
    return _save(fig, out)


def image_row_grid(rows: list[list], col_titles: list[str], out, row_captions=None, cmap=None, title=""):
    """Generic grid of images (each a CHW tensor / HxW array in [0,1])."""
    n, m = len(rows), len(col_titles)
    fig, axes = plt.subplots(n, m, figsize=(1.6 * m, 1.6 * n))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            a = to_hwc(img)
            axes[i, j].imshow(a, cmap=cmap if a.ndim == 2 else None, vmin=0, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=7)
        if row_captions:
            axes[i, 0].set_ylabel(row_captions[i], fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
    fig.suptitle(title, fontsize=9)
    return _save(fig, out)
