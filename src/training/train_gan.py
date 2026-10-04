"""Training loop for the style-conditioned pix2pix (Task 4).

D step:  L_D = 0.5 * [ BCE(D(x, y, s), 1) + BCE(D(x, G(x,s).detach(), s), 0) ]
G step:  L_G = BCE(D(x, G(x,s), s), 1) + lambda_L1 * L1(y, G(x,s))          (PDF p.7-8)
(BCE = binary cross-entropy with logits; the 0.5 factor on D follows pix2pix.)

Logged separately every epoch: d_real, d_fake, g_adv, g_l1 and validation L1 / SSIM /
PSNR. A FIXED, style-balanced set of validation photos is rendered every
``sample_every`` epochs so that the generator's progress can be inspected over time.

Evaluation uses G.eval() (dropout off) so the deployed ONNX generator is deterministic;
pix2pix kept dropout on at test time as a noise source — see research notes.
"""

from __future__ import annotations

import math

import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.fs2k import FS2KPairs, load_split
from src.evaluation.metrics import batch_metrics
from src.evaluation.plots import image_row_grid
from src.losses import selection_objective
from src.models import StylePatchDiscriminator, StyleUNetGenerator, init_weights
from src.training.common import EarlyStopping, History, Timer, autocast, grad_scaler
from src.utils.config import resolve
from src.utils.env_info import get_device
from src.utils.seed import make_generator, set_seed, worker_init_fn


def to01(t: torch.Tensor) -> torch.Tensor:
    return (t.float() + 1) / 2


def fs2k_loaders(cfg: dict, batch_size: int):
    split = load_split()
    dcfg = cfg["data"]
    limit = dcfg.get("limit_data")
    root = split["root"]
    ch = int(cfg["model"]["out_channels"])
    aug = dcfg.get("augment", {})
    train_ds = FS2KPairs(split["train"], root, ch, train=True, load_size=int(aug.get("load_size", 143)),
                         crop_size=128, max_rotation=float(aug.get("max_rotation", 0.0)), seed=int(cfg["seed"]), limit=limit)
    val_ds = FS2KPairs(split["val"], root, ch, train=False, limit=limit)
    nw = int(dcfg.get("num_workers", 0))
    common = dict(num_workers=nw, worker_init_fn=worker_init_fn, pin_memory=torch.cuda.is_available(),
                  persistent_workers=nw > 0)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=make_generator(int(cfg["seed"])), **common)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, **common)
    return train_dl, val_dl


def fixed_validation_batch(val_ds, per_style: int = 3):
    """Deterministic, style-balanced set of validation pairs for progress logging."""
    chosen = {s: [] for s in range(3)}
    for i, r in enumerate(val_ds.records):
        s = int(r["style"])
        if len(chosen[s]) < per_style:
            chosen[s].append(i)
    idx = [i for s in range(3) for i in chosen[s]]
    items = [val_ds[i] for i in idx]
    return {"photo": torch.stack([it["photo"] for it in items]), "sketch": torch.stack([it["sketch"] for it in items]),
            "style": torch.tensor([it["style"] for it in items]), "name": [it["name"] for it in items]}


@torch.no_grad()
def evaluate_generator(G, loader, device) -> dict:
    G.eval()
    tot, n = {"l1": 0.0, "ssim": 0.0, "psnr": 0.0}, 0
    per_style = {s: {"l1": 0.0, "ssim": 0.0, "n": 0} for s in range(3)}
    for b in loader:
        x, y, s = b["photo"].to(device), b["sketch"].to(device), b["style"].to(device)
        y_hat = G(x, s)
        m = batch_metrics(to01(y_hat), to01(y))
        for k in tot:
            tot[k] += float(m[k].sum())
        for st in range(3):
            sel = s == st
            per_style[st]["l1"] += float(m["l1"][sel].sum())
            per_style[st]["ssim"] += float(m["ssim"][sel].sum())
            per_style[st]["n"] += int(sel.sum())
        n += len(s)
    res = {k: v / max(n, 1) for k, v in tot.items()}
    res["objective"] = selection_objective(res["l1"], res["ssim"])
    for st, d in per_style.items():
        res[f"style{st + 1}_l1"] = d["l1"] / max(d["n"], 1)
        res[f"style{st + 1}_ssim"] = d["ssim"] / max(d["n"], 1)
    return res


@torch.no_grad()
def render_fixed(G, fixed, device, path, title):
    """Rows: photo | ground truth (its own style) | G(x, Style 1) | G(x, Style 2) | G(x, Style 3)."""
    G.eval()
    x = fixed["photo"].to(device)
    outs = [G(x, torch.full((len(x),), s, device=device, dtype=torch.long)) for s in range(3)]
    rows = []
    for i in range(len(x)):
        rows.append([to01(fixed["photo"][i]), to01(fixed["sketch"][i])] + [to01(o[i]).cpu() for o in outs])
    caps = [f"{n.split('/')[-1]}\ntrue style {int(s) + 1}" for n, s in zip(fixed["name"], fixed["style"])]
    image_row_grid(rows, ["Photo", "Target sketch", "G(x, Style 1)", "G(x, Style 2)", "G(x, Style 3)"], path, caps,
                   cmap="gray", title=title)
    own = torch.stack([outs[int(s)][i] for i, s in enumerate(fixed["style"])]).cpu()
    return own


def train_gan(cfg: dict, trial: optuna.Trial | None = None, tracker=None, checkpoint_path=None,
              history_path=None, progress_dir=None) -> dict:
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    tcfg, mcfg = cfg["train"], cfg["model"]
    epochs = int(tcfg["epochs"])
    amp = bool(tcfg.get("amp", False))
    train_dl, val_dl = fs2k_loaders(cfg, int(tcfg["batch_size"]))
    G = StyleUNetGenerator(in_channels=3, out_channels=int(mcfg["out_channels"]), base_channels=int(mcfg["g_channels"]),
                           style_dim=int(mcfg["style_dim"]), dropout=float(mcfg["dropout"]), norm=mcfg["norm"]).to(device)
    D = StylePatchDiscriminator(photo_channels=3, sketch_channels=int(mcfg["out_channels"]),
                                base_channels=int(mcfg["d_channels"]), style_dim=int(mcfg["style_dim"]),
                                norm=mcfg["norm"]).to(device)
    G.apply(init_weights)
    D.apply(init_weights)
    betas = (float(tcfg.get("beta1", 0.5)), 0.999)
    opt_g = torch.optim.Adam(G.parameters(), lr=float(tcfg["lr_g"]), betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=float(tcfg["lr_d"]), betas=betas)
    # pix2pix schedule: constant lr, then linear decay to 0 over the second half
    decay_start = int(epochs * float(tcfg.get("decay_start_fraction", 0.5)))
    lam = lambda e: 1.0 if e < decay_start else max(0.0, 1 - (e - decay_start) / max(epochs - decay_start, 1))  # noqa: E731
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lam)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lam)
    scaler_g, scaler_d = grad_scaler(device, amp), grad_scaler(device, amp)
    lambda_l1 = float(cfg["loss"]["lambda_l1"])
    fixed = fixed_validation_batch(val_dl.dataset, int(cfg.get("fixed_per_style", 3)))
    sample_every = int(tcfg.get("sample_every", 5))
    hist = History()
    stopper = EarlyStopping(tcfg.get("early_stopping_patience"))
    best = {"objective": math.inf}
    progression = []

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        agg = {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0, "d_real_prob": 0.0, "d_fake_prob": 0.0}
        nb = 0
        with Timer() as t:
            for b in tqdm(train_dl, desc=f"GAN ep{epoch}", leave=False, disable=not cfg.get("progress", True)):
                x, y, s = b["photo"].to(device), b["sketch"].to(device), b["style"].to(device)
                # ---- D step ----
                with autocast(device, amp):
                    y_fake = G(x, s)
                    logit_real = D(x, y, s)
                    logit_fake = D(x, y_fake.detach(), s)
                d_real = F.binary_cross_entropy_with_logits(logit_real.float(), torch.ones_like(logit_real, dtype=torch.float32))
                d_fake = F.binary_cross_entropy_with_logits(logit_fake.float(), torch.zeros_like(logit_fake, dtype=torch.float32))
                loss_d = 0.5 * (d_real + d_fake)
                opt_d.zero_grad(set_to_none=True)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
                # ---- G step ----
                with autocast(device, amp):
                    logit_gen = D(x, y_fake, s)
                g_adv = F.binary_cross_entropy_with_logits(logit_gen.float(), torch.ones_like(logit_gen, dtype=torch.float32))
                g_l1 = F.l1_loss(y_fake.float(), y)
                loss_g = g_adv + lambda_l1 * g_l1
                opt_g.zero_grad(set_to_none=True)
                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g)
                scaler_g.update()
                agg["d_real"] += float(d_real.detach())
                agg["d_fake"] += float(d_fake.detach())
                agg["g_adv"] += float(g_adv.detach())
                agg["g_l1"] += float(g_l1.detach())
                agg["d_real_prob"] += float(torch.sigmoid(logit_real.detach().float()).mean())
                agg["d_fake_prob"] += float(torch.sigmoid(logit_fake.detach().float()).mean())
                nb += 1
        sch_g.step()
        sch_d.step()
        val = evaluate_generator(G, val_dl, device)
        row = {"epoch": epoch, **{f"train_{k}": v / max(nb, 1) for k, v in agg.items()},
               **{f"val_{k}": v for k, v in val.items()}, "lr_g": opt_g.param_groups[0]["lr"], "epoch_time_s": t.elapsed}
        hist.add(row)
        if tracker:
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        print(f"[GAN] epoch {epoch}/{epochs} D_real={row['train_d_real']:.3f} D_fake={row['train_d_fake']:.3f} "
              f"G_adv={row['train_g_adv']:.3f} G_L1={row['train_g_l1']:.4f} | val L1={val['l1']:.4f} "
              f"SSIM={val['ssim']:.4f} PSNR={val['psnr']:.2f} ({t.elapsed:.1f}s)")
        if progress_dir and (epoch == 1 or epoch % sample_every == 0 or epoch == epochs):
            p = f"{progress_dir}/epoch_{epoch:03d}.png"
            own = render_fixed(G, fixed, device, p, f"Fixed validation photos — epoch {epoch}")
            progression.append((epoch, own))
            if tracker:
                tracker.log_artifact(resolve(p), "progress")
        if not math.isfinite(val["objective"]):
            if trial:
                raise optuna.TrialPruned("diverged")
            raise RuntimeError("GAN diverged")
        improved, stop = stopper.step(val["objective"])
        if improved:
            best = {**val, "epoch": epoch}
            if checkpoint_path:
                path = resolve(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": G.state_dict(), "model_config": G.config, "d_state": D.state_dict(),
                            "d_config": D.config, "epoch": epoch, "val": val, "cfg": cfg}, path)
        if trial is not None:
            trial.report(val["objective"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stop:
            print(f"[GAN] early stopping at epoch {epoch}")
            break

    if history_path:
        hist.save(history_path)
    if progress_dir and progression:
        rows = [[to01(fixed["photo"][i]), to01(fixed["sketch"][i])] + [to01(o[i]) for _, o in progression]
                for i in range(0, len(fixed["photo"]), max(1, len(fixed["photo"]) // 3))][:3]
        image_row_grid(rows, ["Photo", "Target"] + [f"ep {e}" for e, _ in progression],
                       f"{progress_dir}/progression_summary.png", cmap="gray",
                       title="Generator progression on fixed validation photos")
    return {"best": best, "history": hist.rows, "g_params": sum(p.numel() for p in G.parameters()),
            "d_params": sum(p.numel() for p in D.parameters())}


def style_influence(G, loader, device) -> dict:
    """Mean absolute difference between generations of the same photo under different
    styles — quantitative evidence that the style embedding changes the output."""
    G.eval()
    diffs = {"1v2": [], "1v3": [], "2v3": []}
    with torch.no_grad():
        for b in loader:
            x = b["photo"].to(device)
            o = [to01(G(x, torch.full((len(x),), s, device=device, dtype=torch.long))) for s in range(3)]
            diffs["1v2"].append((o[0] - o[1]).abs().mean(dim=(1, 2, 3)).cpu())
            diffs["1v3"].append((o[0] - o[2]).abs().mean(dim=(1, 2, 3)).cpu())
            diffs["2v3"].append((o[1] - o[2]).abs().mean(dim=(1, 2, 3)).cpu())
    return {k: float(torch.cat(v).mean()) for k, v in diffs.items()} | {"note": "mean |G(x,a)-G(x,b)| in [0,1] units"}

