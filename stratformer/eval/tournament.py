"""Tournament utilities (stub)."""

from __future__ import annotations

from stratformer.env.open_spiel_wrapper import OpenSpielEnv
from stratformer.pool.strategy_pool import StrategyPool

__all__ = ["Tournament"]


class Tournament:
    """Run round-robin tournaments among a set of policies.

    Notes
    -----
    # IMPLEMENT: orchestrate matches, compute standings, and export metrics.
    """

    def run(
        self,
        participants: list[str],
        pool: StrategyPool,
        env: OpenSpielEnv,
    ) -> dict[tuple[str, str], float]:
        """Return placeholder results keyed by (player, opponent)."""

        results: dict[tuple[str, str], float] = {}
        for i, a in enumerate(participants):
            for b in participants[i + 1 :]:
                results[(a, b)] = 0.0
                results[(b, a)] = 0.0
        return results


