"""Training loop for denoising autoencoders (Task 1 universal model, Task 2 specialists).

Loss:  alpha * L1 + (1 - alpha) * (1 - SSIM)                    (PDF p.4)
Model selection / Optuna objective:  J = 0.5 * L1 + 0.5 * (1 - SSIM) on the
deterministic validation manifest (see src/losses/reconstruction.py for why J is
independent of alpha).
"""

from __future__ import annotations

import math

import optuna
import torch
from tqdm import tqdm

from src.evaluation.metrics import batch_metrics
from src.losses import l1_ssim_loss, selection_objective
from src.models import DenoisingAutoencoder
from src.training.common import (
    EarlyStopping, History, Timer, autocast, grad_scaler, make_scheduler, oxford_loaders,
    save_checkpoint, step_scheduler,
)
from src.utils.env_info import get_device
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate_autoencoder(model, loader, device, amp=True) -> dict:
    model.eval()
    tot = {"l1": 0.0, "ssim": 0.0, "psnr": 0.0}
    n = 0
    for batch in loader:
        x_in = batch["input"].to(device, non_blocking=True)
        x = batch["target"].to(device, non_blocking=True)
        with autocast(device, amp):
            x_hat = model(x_in)
        m = batch_metrics(x_hat.float(), x)
        b = x.shape[0]
        for k in tot:
            tot[k] += float(m[k].sum())
        n += b
    out = {k: v / max(n, 1) for k, v in tot.items()}
    out["objective"] = selection_objective(out["l1"], out["ssim"])
    return out


def train_autoencoder(cfg: dict, allowed_types=None, trial: optuna.Trial | None = None,
                      tracker=None, checkpoint_path: str | None = None, desc: str = "AE",
                      history_path: str | None = None, step_offset: int = 0) -> dict:
    """Train one autoencoder. ``allowed_types=None`` -> universal (all 4 conditions,
    balanced batches); ``["salt_pepper"]`` etc. -> specialist on that corruption only.
    ``step_offset`` shifts the Optuna report step (a shared specialist trial trains
    three models in sequence and reports them as one continuous curve)."""
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    tcfg = cfg["train"]
    epochs = int(tcfg["epochs"])
    amp = bool(tcfg.get("amp", True))
    alpha = float(cfg["loss"]["alpha"])

    train_dl, val_dl = oxford_loaders(
        cfg, int(tcfg["batch_size"]), allowed_types=allowed_types,
        balanced=cfg.get("data", {}).get("balanced_batches", True),
        val_types=allowed_types,
    )
    model = DenoisingAutoencoder(**cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.0)))
    sched = make_scheduler(opt, tcfg, epochs)
    scaler = grad_scaler(device, amp)
    stopper = EarlyStopping(tcfg.get("early_stopping_patience"))
    hist = History()
    best = {"objective": math.inf}

    for epoch in range(1, epochs + 1):
        model.train()
        run = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
        nb = 0
        with Timer() as t:
            for batch in tqdm(train_dl, desc=f"{desc} ep{epoch}", leave=False, disable=not cfg.get("progress", True)):
                x_in = batch["input"].to(device, non_blocking=True)
                x = batch["target"].to(device, non_blocking=True)
                with autocast(device, amp):
                    x_hat = model(x_in)
                loss, parts = l1_ssim_loss(x_hat, x, alpha)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run["loss"] += float(loss.detach())
                run["l1"] += float(parts["l1"])
                run["ssim"] += float(parts["ssim"])
                nb += 1
        train_m = {f"train_{k}": v / max(nb, 1) for k, v in run.items()}
        val_m = evaluate_autoencoder(model, val_dl, device, amp)
        # validation loss with the SAME alpha as training (for train/val curves)
        val_m["loss"] = alpha * val_m["l1"] + (1 - alpha) * (1 - val_m["ssim"])
        step_scheduler(sched, val_m["objective"])
        row = {"epoch": epoch, **train_m, **{f"val_{k}": v for k, v in val_m.items()},
               "lr": opt.param_groups[0]["lr"], "epoch_time_s": t.elapsed}
        hist.add(row)
        if tracker:
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        print(f"[{desc}] epoch {epoch}/{epochs} train_loss={train_m['train_loss']:.4f} "
              f"val_L1={val_m['l1']:.4f} val_SSIM={val_m['ssim']:.4f} val_PSNR={val_m['psnr']:.2f} "
              f"J={val_m['objective']:.4f} ({t.elapsed:.1f}s)")

        if not math.isfinite(val_m["objective"]):
            raise optuna.TrialPruned("non-finite objective") if trial else RuntimeError("training diverged")
        improved, stop = stopper.step(val_m["objective"])
        if improved:
            best = {**val_m, "epoch": epoch}
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, {"epoch": epoch, "val": val_m, "cfg": cfg,
                                                         "allowed_types": allowed_types})
        if trial is not None:
            trial.report(val_m["objective"], epoch + step_offset)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stop:
            print(f"[{desc}] early stopping at epoch {epoch}")
            break

    if history_path:
        hist.save(history_path)
    return {"best": best, "history": hist.rows, "bottleneck_dim": model.bottleneck_dim}
