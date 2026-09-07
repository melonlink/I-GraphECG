"""YAML config loading with lightweight CLI overrides."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config into a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write the (runtime) config back to YAML for experiment provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dump, f, allow_unicode=True, sort_keys=False)


def parse_args_with_config(description: str = "") -> tuple[argparse.Namespace, dict[str, Any]]:
    """Standard entry point: --config gives the YAML, --override k=v overrides top-level keys."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="path to the YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="overrides of the form key=value")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for item in args.override:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        val = yaml.safe_load(v)
        if "." in k:  # nested keys are supported, e.g. train.epochs=3
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
        else:
            cfg[k] = val
    return args, cfg
