"""Single-place logger configuration for Phase 1."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .paths import LOG_FILE, ensure_dirs


def get_logger(name: str = "phase1") -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def log_header(logger: logging.Logger, stage: str, **kv: object) -> None:
    logger.info("=" * 80)
    logger.info("STAGE: %s", stage)
    for k, v in kv.items():
        logger.info("  %s = %s", k, v)
    logger.info("=" * 80)
