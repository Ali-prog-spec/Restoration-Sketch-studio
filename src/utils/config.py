"""YAML configuration with SMOKE / DEV / FINAL modes and CLI overrides.

A config file has a ``base`` section plus optional ``modes: {smoke: ..., dev: ..., final: ...}``
sections that are deep-merged on top of ``base``. Command-line flags are merged last.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, mode: str = "dev", overrides: dict | None = None) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = copy.deepcopy(raw.get("base", {}))
    modes = raw.get("modes", {})
    if mode not in modes and mode != "base":
        raise ValueError(f"mode '{mode}' not defined in {path} (have {list(modes)})")
    cfg = deep_update(cfg, modes.get(mode, {}))
    cfg = deep_update(cfg, overrides or {})
    cfg["mode"] = mode
    cfg["config_path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags shared by every training / optimisation / evaluation script."""
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--mode", choices=["smoke", "dev", "final"], default="dev")
    parser.add_argument("--smoke-test", action="store_true", help="alias for --mode smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | auto")
    parser.add_argument("--limit-data", type=int, default=None, help="use only N images per split")
    parser.add_argument("--num-trials", type=int, default=None, help="Optuna trials")
    parser.add_argument("--no-tracking", action="store_true", help="disable MLflow logging")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def config_from_args(args: argparse.Namespace, default_config: str) -> dict:
    mode = "smoke" if getattr(args, "smoke_test", False) else args.mode
    ov: dict = {}
    train: dict = {}
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if train:
        ov["train"] = train
    data: dict = {}
    if args.num_workers is not None:
        data["num_workers"] = args.num_workers
    if args.limit_data is not None:
        data["limit_data"] = args.limit_data
    if data:
        ov["data"] = data
    if args.device is not None:
        ov["device"] = args.device
    if args.num_trials is not None:
        ov["optuna"] = {"n_trials": args.num_trials}
    if args.no_tracking:
        ov["tracking"] = {"enabled": False}
    if args.seed is not None:
        ov["seed"] = args.seed
    return load_config(args.config or default_config, mode=mode, overrides=ov)


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def save_json(obj, path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path):
    with open(resolve(path), "r", encoding="utf-8") as f:
        return json.load(f)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
