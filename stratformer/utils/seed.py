"""Seeding utilities for reproducibility.

This module sets seeds for ``random``, ``numpy``, and optionally ``torch`` if
installed. Use this once at program start.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

__all__ = ["set_global_seed", "seed_all"]

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch for reproducibility.

    Parameters
    ----------
    seed
        Seed value to use across libraries.
    deterministic_torch
        If ``True`` and PyTorch is available, set deterministic flags for CUDA.

    Notes
    -----
    PyTorch is optional; if not installed, seeding silently skips it.
    """

    logger.info("Setting global seed: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception as exc:  # noqa: BLE001 - narrow context
                logger.debug("Could not set cuDNN deterministic flags: %s", exc)
    except Exception:
        logger.debug("PyTorch not available; skipping torch seeding.")

def seed_all(seed: int) -> None:
    """Seed Python ``random``, NumPy, and OpenSpiel (if available).

    This utility is intentionally lightweight and avoids optional heavy deps.

    Parameters
    ----------
    seed
        Seed value to use across libraries.
    """

    logger.info("Seeding random, numpy, and OpenSpiel (if available) with: %d", seed)
    random.seed(seed)
    np.random.seed(seed)

    # Seed OpenSpiel if the Python API exposes a seeding hook.
    try:
        import pyspiel

        if hasattr(pyspiel, "set_random_seed"):
            # Newer OpenSpiel versions
            pyspiel.set_random_seed(int(seed))
        elif hasattr(pyspiel, "set_global_random_seed"):
            # Fallback name seen in some builds
            pyspiel.set_global_random_seed(int(seed))
        # If no seeding hook exists, we still achieve determinism by sampling
        # chance nodes via our own NumPy RNG in environment wrappers.
    except ImportError:
        logger.debug("pyspiel not available; skipping OpenSpiel seeding.")


