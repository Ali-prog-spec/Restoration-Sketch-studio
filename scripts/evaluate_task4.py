"""Final evaluation of the Task 4 generator on the OFFICIAL FS2K TEST split (run once).

    python scripts/evaluate_task4.py --mode final [--lpips]

ASSIGNMENT-REQUIRED measurement: L1 between generated and ground-truth sketch
(the reconstruction term of the objective) — reported on validation during training
and on test here.
ADDITIONAL research-based metrics: SSIM and PSNR (paired, pixel/structure level) and
optionally LPIPS (Zhang et al., 2018; perceptual distance, needs `pip install lpips`).
FID is NOT reported: with ~1k test images at 128 px it is strongly biased
(Heusel et al. 2017 recommend >= 10k samples) and Inception features are not
designed for grayscale line drawings.
"""

import _bootstrap  # noqa: F401

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.fs2k import FS2KPairs, load_split
from src.evaluation.metrics import batch_metrics
from src.evaluation.plots import image_row_grid
from src.evaluation.reporting import report_dir, latex_table
from src.models import StyleUNetGenerator
from src.training.train_gan import style_influence, to01
from src.utils.config import add_common_args, config_from_args, resolve, save_json
from src.utils.env_info import get_device
from src.utils.tracking import Tracker


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()
    cfg = config_from_args(args, "configs/task4.yaml")
    device = get_device(cfg.get("device"))
    ck = torch.load(resolve(cfg["output"]["checkpoint"]), map_location="cpu", weights_only=False)
    G = StyleUNetGenerator(**ck["model_config"]).to(device).eval()
    G.load_state_dict(ck["model_state"])
    split = load_split()
    ds = FS2KPairs(split[args.split], split["root"], ck["model_config"]["out_channels"], train=False,
                   limit=cfg["data"].get("limit_data"))
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    lp = None
    if args.lpips:
        import lpips  # optional

        lp = lpips.LPIPS(net="alex").to(device)

    rows = []
    with torch.no_grad():
        for b in dl:
            x, y, s = b["photo"].to(device), b["sketch"].to(device), b["style"].to(device)
            y_hat = G(x, s)
            m = batch_metrics(to01(y_hat), to01(y))
            lp_v = None
            if lp is not None:
                rep = lambda t: t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t  # noqa: E731
                lp_v = lp(rep(y_hat), rep(y)).flatten().cpu()
            for i in range(len(s)):
                r = {"name": b["name"][i], "style": int(s[i]) + 1, "l1": float(m["l1"][i]), "ssim": float(m["ssim"][i]),
                     "psnr": float(m["psnr"][i])}
                if lp_v is not None:
                    r["lpips"] = float(lp_v[i])
                rows.append(r)
    df = pd.DataFrame(rows)
    out = cfg["output"]["dir"]
    df.to_csv(resolve(f"{out}/{args.split}_per_image.csv"), index=False)
    metric_cols = [c for c in ["l1", "ssim", "psnr", "lpips"] if c in df]
    by_style = df.groupby("style")[metric_cols].mean()
    by_style.loc["overall"] = df[metric_cols].mean()
    by_style["n"] = list(df.groupby("style").size()) + [len(df)]
    by_style.to_csv(resolve(f"{out}/{args.split}_metrics_by_style.csv"))
    latex_table(by_style, f"{report_dir(cfg)}/task4_{args.split}_by_style.tex",
                f"Task 4 generator on the FS2K {args.split} split. L1 is the assignment-required reconstruction "
                "measure; SSIM/PSNR (and LPIPS if shown) are additional.", f"tab:task4_{args.split}",
                metric_cols + ["n"], {"psnr": "{:.2f}", "n": "{:.0f}"}, index_names="Style")

    infl = style_influence(G, dl, device)
    save_json({"split": args.split, "overall": by_style.loc["overall"].to_dict(), "style_influence": infl,
               "checkpoint": cfg["output"]["checkpoint"], "epoch": ck.get("epoch")}, f"{out}/{args.split}_summary.json")

    # photo | target | generated grids: best, median, worst per style + style swap grid
    for st in (1, 2, 3):
        g = df[df["style"] == st].sort_values("ssim")
        if not len(g):
            continue
        picks = [g.iloc[-1], g.iloc[len(g) // 2], g.iloc[0]]
        rws, caps = [], []
        for r, label in zip(picks, ["best", "median", "worst"]):
            it = ds[[rec["image_name"] for rec in ds.records].index(r["name"])]
            with torch.no_grad():
                gen = G(it["photo"][None].to(device), torch.tensor([st - 1], device=device))[0].cpu()
            rws.append([to01(it["photo"]), to01(it["sketch"]), to01(gen)])
            caps.append(f"{label}\nSSIM {r['ssim']:.3f}\nL1 {r['l1']:.3f}")
        image_row_grid(rws, ["Photo", "Target", "Generated"], f"{out}/figures/{args.split}_style{st}_examples.png", caps,
                       cmap="gray", title=f"Task 4 Style {st}: best / median / worst ({args.split})")
    swap_rows, swap_caps = [], []
    for i in range(0, min(len(ds), 6 * max(1, len(ds) // 6)), max(1, len(ds) // 6)):
        it = ds[i]
        with torch.no_grad():
            gens = [G(it["photo"][None].to(device), torch.tensor([k], device=device))[0].cpu() for k in range(3)]
        swap_rows.append([to01(it["photo"]), to01(it["sketch"])] + [to01(g) for g in gens])
        swap_caps.append(f"true style {it['style'] + 1}")
    image_row_grid(swap_rows[:6], ["Photo", "Target", "Style 1", "Style 2", "Style 3"],
                   f"{out}/figures/{args.split}_style_swap.png", swap_caps, cmap="gray",
                   title="Same photo, three style conditions")

    tracker = Tracker(cfg)
    with tracker.run(f"task4-{args.split}-eval-{cfg['mode']}", tags={"task": "task4", "split": args.split}):
        tracker.log_metrics({f"{args.split}_{k}": v for k, v in by_style.loc["overall"].items()})
        tracker.log_metrics({f"style_diff_{k}": v for k, v in infl.items() if isinstance(v, float)})
        tracker.log_artifacts(resolve(f"{out}/figures"), "figures")
    print(by_style.round(4).to_string())
    print(infl)


if __name__ == "__main__":
    main()
