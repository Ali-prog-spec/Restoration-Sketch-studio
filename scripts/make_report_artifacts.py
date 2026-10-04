"""Collect experiment outputs into report/ (figures + LaTeX tables) and draw diagrams.

    python scripts/make_report_artifacts.py --mode final

Every number that ends up in a generated table is read from a machine-readable
artifact (CSV/JSON written by the training/evaluation scripts). Missing artifacts are
reported and skipped — nothing is invented.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.utils.config import resolve  # noqa: E402

FIG = resolve("report/figures")
TAB = resolve("report/tables")


def copy(src: str, name: str, missing: list):
    p = resolve(src)
    if p.exists():
        FIG.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, FIG / name)
    else:
        missing.append(src)


def tex_escape(s) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def optuna_table(study_dir: str, task: str, label: str, missing: list):
    p = resolve(study_dir) / "study_summary.json"
    if not p.exists():
        missing.append(str(p))
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for name, spec in s["search_space"].items():
        if spec["type"] == "categorical":
            rng = "\\{" + ", ".join(tex_escape(c) for c in spec["choices"]) + "\\}"
        else:
            rng = f"[{spec['low']:g}, {spec['high']:g}]" + (" (log)" if spec.get("log") else "")
        best = s.get("best_params", {}).get(name, "--")
        best = f"{best:.4g}" if isinstance(best, float) else tex_escape(best)
        rows.append(f"{tex_escape(name)} & {spec['type']} & {rng} & {best} \\\\")
    states = ", ".join(f"{k.lower()} {v}" for k, v in s.get("trial_states", {}).items())
    body = "\n".join(rows)
    cap = (f"{task} Optuna study \\texttt{{{tex_escape(s['study_name'])}}}: {s['n_trials_total']} trials ({states}); "
           f"{s.get('trial_epochs', '?')} epochs per trial; best objective "
           f"{s.get('best_value', float('nan')):.4f} (trial {s.get('best_trial_number', '--')}).")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{" + label + "}\n\\small\n"
           "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{llll}\n\\toprule\nParameter & Type & Search range & Best \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{label.replace('tab:', '')}.tex").write_text(tex, encoding="utf-8")


# ----------------------------------------------------------------------------- diagrams
def _box(ax, x, y, w, h, text, color="#e0e7ff", fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=color, ec="#334155", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#334155"))


def diagram_autoencoder():
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 3)
    ax.axis("off")
    enc = [("x̃\n3×128²", 0.2, 2.2), ("E1\nC×64²", 1.5, 2.0), ("E2\n2C×32²", 2.8, 1.7), ("E3\n4C×16²", 4.1, 1.4),
           ("E4\n8C×8²", 5.4, 1.1)]
    for t, x, h in enc:
        _box(ax, x, 1.5 - h / 2, 1.0, h, t, "#dbeafe")
    _box(ax, 6.7, 1.1, 1.1, 0.8, "z\nL×8×8", "#fde68a")
    dec = [("D1\n4C×16²", 8.1, 1.4), ("D2\n2C×32²", 9.4, 1.7), ("D3\nC×64²", 10.7, 2.0), ("D4\nC×128²", 12.0, 2.2),
           ("x̂\n3×128²", 13.3, 2.2)]
    for t, x, h in dec:
        _box(ax, x, 1.5 - h / 2, 1.0, h, t, "#dcfce7")
    xs = [x for _, x, _ in enc] + [6.7] + [x for _, x, _ in dec]
    ws = [1.0] * 5 + [1.1] + [1.0] * 5
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + ws[i], 1.5), (xs[i + 1], 1.5))
    ax.text(3.2, 0.05, "encoder: stride-2 conv + conv (BN, LeakyReLU)", fontsize=6.5, ha="center")
    ax.text(10.7, 0.05, "decoder: bilinear ×2 + conv; sigmoid output", fontsize=6.5, ha="center")
    ax.text(7.25, 2.35, "bottleneck\n(no skip connections)", fontsize=6.5, ha="center")
    fig.savefig(FIG / "arch_autoencoder.pdf", bbox_inches="tight")
    fig.savefig(FIG / "arch_autoencoder.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def diagram_moe():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    _box(ax, 0.2, 1.8, 1.2, 0.8, "x̃", "#dbeafe")
    _box(ax, 2.2, 3.4, 2.4, 0.8, "Gate G (Task 2 classifier)", "#fde68a")
    _box(ax, 5.2, 3.4, 2.2, 0.8, "w = softmax(G/τ)", "#fde68a")
    names = ["identity", "A_salt", "A_blur", "A_occ"]
    for i, n in enumerate(names):
        _box(ax, 2.6, 2.4 - i * 0.75, 1.8, 0.6, n, "#dcfce7" if i else "#f1f5f9")
        _arrow(ax, (1.4, 2.2), (2.6, 2.7 - i * 0.75))
    _arrow(ax, (1.4, 2.4), (2.2, 3.8))
    _arrow(ax, (4.6, 3.8), (5.2, 3.8))
    _box(ax, 8.0, 1.2, 1.6, 1.4, "Σ w_k · branch_k", "#e9d5ff")
    for i in range(4):
        _arrow(ax, (4.4, 2.7 - i * 0.75), (8.0, 1.9))
    _arrow(ax, (6.3, 3.4), (8.8, 2.6))
    _box(ax, 10.2, 1.5, 1.5, 0.8, "x̂", "#dbeafe")
    _arrow(ax, (9.6, 1.9), (10.2, 1.9))
    ax.text(6.0, 0.1, "Loss: λ1·L1 + λs·(1−SSIM) + λc·CE(G(x̃), y) + λb·L_balance(w)", ha="center", fontsize=7)
    fig.savefig(FIG / "arch_moe.pdf", bbox_inches="tight")
    fig.savefig(FIG / "arch_moe.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def diagram_gan():
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    _box(ax, 0.1, 2.6, 1.4, 0.8, "photo x", "#dbeafe")
    _box(ax, 0.1, 1.2, 1.4, 0.8, "style s\n(1,2,3)", "#fde68a")
    _box(ax, 2.1, 1.2, 1.7, 0.8, "Embedding_G\n(3 × d)", "#fde68a")
    _box(ax, 4.4, 2.2, 2.3, 1.4, "U-Net G\ninput: [x ; tile(e_s)]\nbottleneck: [h ; e_s]", "#dcfce7")
    _box(ax, 7.3, 2.6, 1.3, 0.8, "ŷ = G(x,s)", "#dbeafe")
    _box(ax, 9.2, 1.6, 2.6, 1.6, "PatchGAN D\n[x ; y or ŷ ; tile(e'_s)]\n→ 14×14 logits", "#fecaca")
    _box(ax, 7.3, 0.6, 1.3, 0.8, "real y", "#f1f5f9")
    _arrow(ax, (1.5, 3.0), (4.4, 3.0))
    _arrow(ax, (1.5, 1.6), (2.1, 1.6))
    _arrow(ax, (3.8, 1.6), (4.4, 2.5))
    _arrow(ax, (6.7, 3.0), (7.3, 3.0))
    _arrow(ax, (8.6, 3.0), (9.2, 2.7))
    _arrow(ax, (8.6, 1.0), (9.2, 2.0))
    ax.text(6.0, 0.05, "L_G = BCE(D(x,ŷ,s),1) + λ_L1·|y − ŷ|₁ ;  L_D = ½[BCE(D(x,y,s),1) + BCE(D(x,ŷ,s),0)]",
            ha="center", fontsize=6.8)
    fig.savefig(FIG / "arch_cgan.pdf", bbox_inches="tight")
    fig.savefig(FIG / "arch_cgan.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def diagram_app():
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    _box(ax, 0.1, 1.1, 1.8, 1.2, "Browser\n(evaluator)", "#f1f5f9")
    _box(ax, 2.5, 0.6, 3.0, 2.2, "frontend container\nnginx + React/Tailwind\n4 workspaces\n/api → backend", "#dbeafe")
    _box(ax, 6.2, 0.6, 3.0, 2.2, "backend container\nFastAPI + ONNX Runtime\nvalidate · preprocess ·\ncorrupt · infer · encode", "#dcfce7")
    _box(ax, 9.9, 0.6, 2.0, 2.2, "models/ volume\n7 × .onnx\nmodel_card.json", "#fde68a")
    _arrow(ax, (1.9, 1.7), (2.5, 1.7))
    _arrow(ax, (5.5, 1.7), (6.2, 1.7))
    _arrow(ax, (9.2, 1.7), (9.9, 1.7))
    ax.text(6.0, 3.1, "docker compose up --build   (http://localhost:8080)", ha="center", fontsize=7.5)
    fig.savefig(FIG / "arch_application.pdf", bbox_inches="tight")
    fig.savefig(FIG / "arch_application.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="final", choices=["smoke", "dev", "final"])
    args = ap.parse_args()
    base = "outputs/smoke" if args.mode == "smoke" else "outputs"
    global FIG, TAB
    if args.mode != "final":  # only FINAL results go into report/
        FIG, TAB = resolve(f"outputs/{args.mode}_report/figures"), resolve(f"outputs/{args.mode}_report/tables")
    FIG.mkdir(parents=True, exist_ok=True)
    missing: list = []
    diagram_autoencoder()
    diagram_moe()
    diagram_gan()
    diagram_app()

    files = {
        # Task 1
        f"{base}/task1/figures/training_curves.png": "task1_training_curves.png",
        f"{base}/task1/figures/representative_1.png": "task1_representative_1.png",
        f"{base}/task1/figures/representative_2.png": "task1_representative_2.png",
        f"{base}/task1/figures/failures_1.png": "task1_failures_1.png",
        f"{base}/task1/figures/failures_2.png": "task1_failures_2.png",
        f"{base}/task1/figures/severity_ssim.png": "task1_severity_ssim.png",
        f"{base}/task1/figures/severity_psnr.png": "task1_severity_psnr.png",
        f"{base}/task1/optuna/{args.mode}/optimization_history.png": "task1_optuna_history.png",
        f"{base}/task1/optuna/{args.mode}/param_importances.png": "task1_optuna_importance.png",
        # Task 2
        f"{base}/task2/classifier/figures/training_curves.png": "task2_classifier_curves.png",
        f"{base}/task2/figures/confusion_matrix.png": "task2_confusion_matrix.png",
        f"{base}/task2/figures/oracle_vs_predicted_ssim.png": "task2_oracle_vs_predicted.png",
        f"{base}/task2/figures/routing_failures.png": "task2_routing_failures.png",
        f"{base}/task2/figures/representative_1.png": "task2_representative_1.png",
        f"{base}/task2/specialists/figures/training_curves_salt_pepper.png": "task2_curves_salt.png",
        f"{base}/task2/specialists/figures/training_curves_blur.png": "task2_curves_blur.png",
        f"{base}/task2/specialists/figures/training_curves_occlusion.png": "task2_curves_occlusion.png",
        f"{base}/task2/classifier/optuna/{args.mode}/optimization_history.png": "task2_cls_optuna_history.png",
        f"{base}/task2/specialists/optuna/{args.mode}/optimization_history.png": "task2_spec_optuna_history.png",
        # Task 3
        f"{base}/task3/figures/training_curves.png": "task3_training_curves.png",
        f"{base}/task3/figures/routing_heatmap.png": "task3_routing_heatmap.png",
        f"{base}/task3/figures/weight_distribution.png": "task3_weight_distribution.png",
        f"{base}/task3/figures/expert_utilisation.png": "task3_expert_utilisation.png",
        f"{base}/task3/figures/dominant_examples.png": "task3_dominant_examples.png",
        f"{base}/task3/figures/mixed_examples.png": "task3_mixed_examples.png",
        f"{base}/task3/figures/failures_1.png": "task3_failures.png",
        f"{base}/task3/figures/system_comparison_ssim.png": "task3_system_comparison.png",
        f"{base}/task3/optuna/{args.mode}/optimization_history.png": "task3_optuna_history.png",
        f"{base}/task3/optuna/{args.mode}/param_importances.png": "task3_optuna_importance.png",
        # Task 4
        f"{base}/task4/figures/gan_losses.png": "task4_gan_losses.png",
        f"{base}/task4/figures/val_metrics.png": "task4_val_metrics.png",
        f"{base}/task4/progress/progression_summary.png": "task4_progression.png",
        f"{base}/task4/figures/test_style_swap.png": "task4_style_swap.png",
        f"{base}/task4/figures/test_style1_examples.png": "task4_style1.png",
        f"{base}/task4/figures/test_style2_examples.png": "task4_style2.png",
        f"{base}/task4/figures/test_style3_examples.png": "task4_style3.png",
        f"{base}/task4/optuna/{args.mode}/optimization_history.png": "task4_optuna_history.png",
        f"{base}/task4/optuna/{args.mode}/param_importances.png": "task4_optuna_importance.png",
    }
    for src, name in files.items():
        copy(src, name, missing)

    optuna_table(f"{base}/task1/optuna/{args.mode}", "Task 1", "tab:optuna_task1", missing)
    optuna_table(f"{base}/task2/classifier/optuna/{args.mode}", "Task 2 classifier", "tab:optuna_task2_cls", missing)
    optuna_table(f"{base}/task2/specialists/optuna/{args.mode}", "Task 2 specialists (shared)", "tab:optuna_task2_spec", missing)
    optuna_table(f"{base}/task3/optuna/{args.mode}", "Task 3", "tab:optuna_task3", missing)
    optuna_table(f"{base}/task4/optuna/{args.mode}", "Task 4", "tab:optuna_task4", missing)

    print(f"figures -> {FIG}\ntables  -> {TAB}")
    if missing:
        print(f"{len(missing)} artifacts not found (not generated yet):")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
