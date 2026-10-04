"""Training loop for the Task 2 corruption classifier.

* labels come from the runtime corruption pipeline (the sampled condition)
* every batch is exactly balanced (BalancedBatchSampler: B/4 per class)
* loss: multiclass cross-entropy; optimiser AdamW (decoupled weight decay,
  Loshchilov & Hutter 2019) so that the tuned weight decay is a true L2-style decay
* model selection / Optuna objective: 1 - macro-F1 on the balanced validation
  manifest, with validation cross-entropy logged too
"""

from __future__ import annotations

import math

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.models import CorruptionClassifier
from src.training.common import (
    EarlyStopping, History, Timer, autocast, grad_scaler, make_scheduler, oxford_loaders,
    save_checkpoint, step_scheduler,
)
from src.utils.env_info import get_device
from src.utils.seed import set_seed


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        with autocast(device, amp):
            lg = model(x)
        logits.append(lg.float().cpu())
        labels.append(batch["label"])
    return torch.cat(logits), torch.cat(labels)


def evaluate_classifier(model, loader, device, amp=True) -> dict:
    logits, y = predict(model, loader, device, amp)
    pred = logits.argmax(1)
    return {
        "ce": float(F.cross_entropy(logits, y)),
        "accuracy": float((pred == y).float().mean()),
        "macro_f1": float(f1_score(y.numpy(), pred.numpy(), average="macro")),
    }


def train_classifier(cfg: dict, trial: optuna.Trial | None = None, tracker=None,
                     checkpoint_path: str | None = None, history_path: str | None = None) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    tcfg = cfg["train"]
    epochs = int(tcfg["epochs"])
    amp = bool(tcfg.get("amp", True))
    train_dl, val_dl = oxford_loaders(cfg, int(tcfg["batch_size"]), balanced=True)

    model = CorruptionClassifier(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    sched = make_scheduler(opt, tcfg, epochs)
    scaler = grad_scaler(device, amp)
    stopper = EarlyStopping(tcfg.get("early_stopping_patience"))
    hist = History()
    best = {"objective": math.inf}

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, correct, n = 0.0, 0, 0
        class_counts = np.zeros(4, dtype=np.int64)
        with Timer() as t:
            for batch in tqdm(train_dl, desc=f"CLS ep{epoch}", leave=False, disable=not cfg.get("progress", True)):
                x = batch["input"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                with autocast(device, amp):
                    logits = model(x)
                loss = F.cross_entropy(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot_loss += float(loss.detach()) * len(y)
                correct += int((logits.argmax(1) == y).sum())
                n += len(y)
                class_counts += np.bincount(batch["label"].numpy(), minlength=4)
        val = evaluate_classifier(model, val_dl, device, amp)
        objective = 1.0 - val["macro_f1"]
        step_scheduler(sched, val["ce"])
        row = {"epoch": epoch, "train_loss": tot_loss / max(n, 1), "train_accuracy": correct / max(n, 1),
               "val_loss": val["ce"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
               "objective": objective, "lr": opt.param_groups[0]["lr"], "epoch_time_s": t.elapsed,
               # evidence that batches are balanced (should be exactly equal)
               "train_class_min": int(class_counts.min()), "train_class_max": int(class_counts.max())}
        hist.add(row)
        if tracker:
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        print(f"[CLS] epoch {epoch}/{epochs} loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} "
              f"val_acc={val['accuracy']:.4f} val_F1={val['macro_f1']:.4f} ({t.elapsed:.1f}s)")
        improved, stop = stopper.step(objective)
        if improved:
            best = {**val, "objective": objective, "epoch": epoch}
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, {"epoch": epoch, "val": val, "cfg": cfg})
        if trial is not None:
            trial.report(objective, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stop:
            break
    if history_path:
        hist.save(history_path)
    return {"best": best, "history": hist.rows}
