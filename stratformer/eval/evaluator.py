"""Evaluation utilities for policies (stub)."""

from __future__ import annotations

from stratformer.env.open_spiel_wrapper import OpenSpielEnv
from stratformer.pool.strategy_pool import Policy

__all__ = ["Evaluator"]


class Evaluator:
    """Run evaluation episodes for a single policy.

    Notes
    -----
    # IMPLEMENT: roll out episodes and aggregate metrics (win rate, EV, etc.).
    """

    def evaluate(self, policy: Policy, env: OpenSpielEnv, *, num_episodes: int) -> dict[str, float]:
        """Evaluate ``policy`` in ``env`` for ``num_episodes`` and return metrics.

        Returns a minimal placeholder to keep the interface stable.
        """

        return {"episodes": float(num_episodes)}


