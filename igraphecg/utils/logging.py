"""Unified logging: to the console and to outputs/logs/<name>_<timestamp>.log."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from .paths import LOGS_DIR


def get_logger(name: str = "ecg", to_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid adding duplicate handlers
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(LOGS_DIR / f"{name}_{ts}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")
