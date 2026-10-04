"""Turn evaluation DataFrames into CSV / LaTeX tables and example figures."""

from __future__ import annotations

import pandas as pd
import torch

from src.corruptions.constants import CLASS_DISPLAY
from src.evaluation.plots import restoration_grid
from src.utils.config import resolve


def report_dir(cfg: dict, kind: str = "tables") -> str:
    """Only FINAL-mode results may be written into report/ (prevents smoke/dev numbers
    from ending up in the IEEE report by accident)."""
    return f"report/{kind}" if cfg["mode"] == "final" else f"outputs/{cfg['mode']}_report/{kind}"


def save_tables(tables: dict[str, pd.DataFrame], out_dir) -> None:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv")


def latex_table(df: pd.DataFrame, path, caption: str, label: str, cols=None, fmt=None, index_names=None) -> None:
    """Minimal booktabs table generator (numbers come straight from the CSV)."""
    cols = cols or list(df.columns)
    fmt = fmt or {}
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule"]
    head = index_names or "Condition"
    lines.append(head + " & " + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\")
    lines.append("\\midrule")
    for idx, row in df.iterrows():
        name = " / ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        cells = []
        for c in cols:
            v = row[c]
            cells.append(fmt.get(c, "{:.4f}").format(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else str(v))
        lines.append(name.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def example_figures(dataset, df: pd.DataFrame, records: list[int], restore_one, out, title, per_fig: int = 6,
                    extra_caption=None):
    """Render the given manifest records as target/input/output/error-map rows."""
    by_rec = df.set_index("record")
    examples = []
    for rec in records:
        item = dataset[rec]
        out_img = restore_one(item["input"])
        r = by_rec.loc[rec]
        cap = f"{CLASS_DISPLAY[r['type']]}\n{r['level']}\nSSIM {r['input_ssim']:.3f}→{r['ssim']:.3f}\nPSNR {r['psnr']:.1f} dB"
        if extra_caption:
            cap += "\n" + extra_caption(r)
        examples.append({"target": item["target"], "input": item["input"], "output": out_img, "caption": cap})
    paths = []
    for k in range(0, len(examples), per_fig):
        part = examples[k : k + per_fig]
        suffix = f"_{k // per_fig + 1}" if len(examples) > per_fig else ""
        paths.append(restoration_grid(part, f"{out}{suffix}.png", title=title))
    return paths
