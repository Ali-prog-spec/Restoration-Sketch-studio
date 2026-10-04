"""Joint training of the soft mixture-of-experts (Task 3).

Initialisation (PDF p.6): gate <- trained Task 2 classifier, experts <- trained Task 2
specialists. Nothing is randomly initialised.

Stage 1 (warm-up, ``warmup_epochs``): experts FROZEN, only the gate is trained (lr_gate).
Stage 2 (joint, ``joint_epochs``):    experts unfrozen, the whole system is fine-tuned
                                      with the smaller lr_joint (< lr_gate, and far below
                                      the specialists' training lr).

    L_MoE = l1 * L1 + ls * (1 - SSIM) + lc * CE(G(x~), y) + lb * L_balance(w)

CE is applied to the untempered gate logits G(x~) (the classifier's own output), so
tau only controls how sharply the gate's belief is turned into mixture weights.
Balance is computed over BALANCED batches (B/4 samples of each true condition).

Expert BatchNorm statistics are kept fixed (``freeze_expert_bn``, default true); see
the comment in the training loop.

Routing collapse: after each epoch the mean validation weight of every branch and the
mean weight on the correct branch are computed. A configuration is "collapsed" if some
branch receives < ``inactive_threshold`` mean weight everywhere (inactive expert) or one
branch receives > ``dominance_threshold`` mean weight on inputs of an unrelated class.
Collapsed Optuna trials are pruned.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.corruptions import CLASS_NAMES
from src.evaluation.metrics import batch_metrics
from src.losses import BALANCE_LOSSES, routing_entropy, selection_objective, ssim
from src.models import SoftMoE
from src.models.loading import load_classifier, load_experts
from src.training.common import EarlyStopping, History, Timer, autocast, grad_scaler, oxford_loaders
from src.utils.config import resolve
from src.utils.env_info import get_device
from src.utils.seed import set_seed


def build_initialised_moe(cfg: dict) -> SoftMoE:
    init = cfg["init"]
    gate = load_classifier(init["classifier_checkpoint"])
    experts = load_experts(init["specialist_dir"])
    return SoftMoE(gate, *experts, tau=float(cfg["model"]["tau"]))


def moe_config_dict(moe: SoftMoE) -> dict:
    return {"gate": moe.gate.config, "experts": [e.config for e in moe.experts], "tau": float(moe.tau)}


def moe_loss(out, weights, logits, x, y, lcfg):
    l1 = F.l1_loss(out.float(), x)
    s = ssim(out.float(), x)
    ce = F.cross_entropy(logits.float(), y)
    bal = BALANCE_LOSSES[lcfg.get("balance", "squared")](weights)
    loss = lcfg["l1"] * l1 + lcfg["ssim"] * (1 - s) + lcfg["ce"] * ce + lcfg["balance_weight"] * bal
    return loss, {"l1": l1.detach(), "ssim": s.detach(), "ce": ce.detach(), "balance": bal.detach()}


@torch.no_grad()
def evaluate_moe(moe, loader, device, amp=True) -> dict:
    moe.eval()
    n, tot = 0, {"l1": 0.0, "ssim": 0.0, "psnr": 0.0, "ce": 0.0, "acc": 0.0, "entropy": 0.0}
    w_sum = np.zeros((4, 4))  # [true class, branch]
    counts = np.zeros(4)
    for batch in loader:
        x_in = batch["input"].to(device, non_blocking=True)
        x = batch["target"].to(device, non_blocking=True)
        y = batch["label"].to(device)
        with autocast(device, amp):
            out, w, logits = moe(x_in)
        m = batch_metrics(out.float(), x)
        b = len(y)
        tot["l1"] += float(m["l1"].sum())
        tot["ssim"] += float(m["ssim"].sum())
        tot["psnr"] += float(m["psnr"].clamp(max=100).sum())
        tot["ce"] += float(F.cross_entropy(logits.float(), y, reduction="sum"))
        tot["acc"] += float((w.argmax(1) == y).sum())
        tot["entropy"] += float(routing_entropy(w).sum())
        yy = y.cpu().numpy()
        ww = w.float().cpu().numpy()
        for c in range(4):
            sel = yy == c
            w_sum[c] += ww[sel].sum(0)
            counts[c] += sel.sum()
        n += b
    res = {k: v / max(n, 1) for k, v in tot.items()}
    res["objective"] = selection_objective(res["l1"], res["ssim"])
    mean_w = w_sum / np.maximum(counts[:, None], 1)
    res["mean_weights"] = mean_w  # rows true class, cols branch
    res["branch_usage"] = (w_sum.sum(0) / max(counts.sum(), 1))
    return res


def collapse_status(mean_w: np.ndarray, inactive_thr: float, dominance_thr: float) -> dict:
    usage = mean_w.mean(axis=0)  # balanced val set -> average over classes = overall usage
    inactive = [CLASS_NAMES[k] for k in range(4) if mean_w[:, k].max() < inactive_thr]
    dominating = [(CLASS_NAMES[k], CLASS_NAMES[c]) for c in range(4) for k in range(4)
                  if k != c and mean_w[c, k] > dominance_thr]
    return {"collapsed": bool(inactive or dominating), "inactive": inactive, "dominating": dominating,
            "usage": usage.tolist()}


def train_moe(cfg: dict, trial: optuna.Trial | None = None, tracker=None, checkpoint_path=None,
              history_path=None) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    tcfg, lcfg = cfg["train"], cfg["loss"]
    amp = bool(tcfg.get("amp", True))
    moe = build_initialised_moe(cfg).to(device)
    train_dl, val_dl = oxford_loaders(cfg, int(tcfg["batch_size"]), balanced=True)
    scaler = grad_scaler(device, amp)
    hist = History()
    stopper = EarlyStopping(tcfg.get("early_stopping_patience"))
    thr_in, thr_dom = float(tcfg.get("inactive_threshold", 0.05)), float(tcfg.get("dominance_threshold", 0.6))

    # evaluation of the initialised (untrained) system — the Task 2 starting point
    init_val = evaluate_moe(moe, val_dl, device, amp)
    print(f"[MoE] initial val J={init_val['objective']:.4f} SSIM={init_val['ssim']:.4f} acc={init_val['acc']:.3f}")
    best = {"objective": math.inf}
    best_state = None
    warm, joint = int(tcfg["warmup_epochs"]), int(tcfg["joint_epochs"])
    total = warm + joint
    opt = None
    for epoch in range(1, total + 1):
        stage = "warmup" if epoch <= warm else "joint"
        if epoch == 1 or epoch == warm + 1:
            if stage == "warmup":
                moe.set_experts_trainable(False)
                opt = torch.optim.Adam(moe.gate.parameters(), lr=float(tcfg["lr_gate"]))
            else:
                moe.set_experts_trainable(True)
                opt = torch.optim.Adam(moe.parameters(), lr=float(tcfg["lr_joint"]))
        moe.train()
        if stage == "warmup" or tcfg.get("freeze_expert_bn", True):
            # Warm-up: experts are frozen, so their BatchNorm statistics must not move either.
            # Joint stage (freeze_expert_bn): every expert now sees ALL four conditions; updating
            # its BN running statistics with mixed inputs would shift the specialist's normalisation
            # away from what it was trained with, so we keep the statistics fixed and fine-tune
            # the weights only (standard practice when fine-tuning with small batches / small lr).
            moe.experts.eval()
        agg = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "ce": 0.0, "balance": 0.0}
        nb = 0
        with Timer() as t:
            for batch in tqdm(train_dl, desc=f"MoE {stage} ep{epoch}", leave=False, disable=not cfg.get("progress", True)):
                x_in = batch["input"].to(device, non_blocking=True)
                x = batch["target"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                with autocast(device, amp):
                    out, w, logits = moe(x_in)
                loss, parts = moe_loss(out, w, logits, x, y, lcfg)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                agg["loss"] += float(loss.detach())
                for k in ("l1", "ssim", "ce", "balance"):
                    agg[k] += float(parts[k])
                nb += 1
        val = evaluate_moe(moe, val_dl, device, amp)
        col = collapse_status(val["mean_weights"], thr_in, thr_dom)
        row = {"epoch": epoch, "stage": stage, **{f"train_{k}": v / max(nb, 1) for k, v in agg.items()},
               "val_l1": val["l1"], "val_ssim": val["ssim"], "val_psnr": val["psnr"], "val_ce": val["ce"],
               "val_route_acc": val["acc"], "val_entropy": val["entropy"], "val_objective": val["objective"],
               "collapsed": col["collapsed"], "epoch_time_s": t.elapsed,
               **{f"val_usage_{CLASS_NAMES[k]}": val["branch_usage"][k] for k in range(4)},
               **{f"val_w_{CLASS_NAMES[c]}_to_{CLASS_NAMES[k]}": val["mean_weights"][c, k] for c in range(4) for k in range(4)}}
        # validation loss with the training weights (for train/val curves)
        row["val_loss"] = (lcfg["l1"] * val["l1"] + lcfg["ssim"] * (1 - val["ssim"]) + lcfg["ce"] * val["ce"])
        hist.add(row)
        if tracker:
            tracker.log_metrics({k: v for k, v in row.items() if k not in ("epoch", "stage")}, step=epoch)
        print(f"[MoE] {stage} epoch {epoch}/{total} loss={row['train_loss']:.4f} val_SSIM={val['ssim']:.4f} "
              f"val_PSNR={val['psnr']:.2f} J={val['objective']:.4f} route_acc={val['acc']:.3f} "
              f"usage={np.round(val['branch_usage'], 3).tolist()} collapsed={col['collapsed']} ({t.elapsed:.1f}s)")
        improved, stop = stopper.step(val["objective"])
        if improved:
            best = {k: v for k, v in val.items() if k not in ("mean_weights", "branch_usage")}
            best.update({"epoch": epoch, "stage": stage, "collapse": col, "mean_weights": val["mean_weights"].tolist()})
            best_state = copy.deepcopy(moe.state_dict())
        if trial is not None:
            if col["collapsed"] and stage == "joint":
                trial.set_user_attr("pruned_reason", f"routing collapse: {col}")
                raise optuna.TrialPruned(f"routing collapse {col}")
            trial.report(val["objective"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stop and stage == "joint":
            break
    if history_path:
        hist.save(history_path)
    initial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in init_val.items()}
    if checkpoint_path and best_state is not None:
        path = resolve(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "model_config": moe_config_dict(moe), "val": best, "cfg": cfg,
                    "initial_val": initial}, path)
    return {"best": best, "history": hist.rows, "initial": initial}
