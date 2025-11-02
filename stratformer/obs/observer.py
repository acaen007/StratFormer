"""Observer interface for constructing observations from environment state."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Observer"]


class Observer:
    """Builds model-ready observations from environment state.

    Implementations should be game-aware and convert the environment's native
    state representation into a fixed-size numeric array.
    """

    def build_observation(self, env_state: Any) -> np.ndarray:  # pragma: no cover - interface only
        """Construct an observation array from ``env_state``.

        # IMPLEMENT: define the observation schema for each supported game.
        """

        raise NotImplementedError


