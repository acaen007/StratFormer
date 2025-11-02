"""Policy selector with KL-regularized objective (stub)."""

from __future__ import annotations

__all__ = ["Selector"]


class Selector:
    """Selects a policy given a posterior over opponent strategies.

    Notes
    -----
    - Objective should incorporate a KL regularizer towards a baseline policy.
    - Optionally enforce an exploitability constraint.
    - This stub returns the policy that maximizes posterior mass as a placeholder.
    """

    def select_policy(
        self,
        candidates: list[str],
        posteriors: dict[str, float],
        *,
        kl_reg: float = 0.0,
        exploitability_limit: float | None = None,
    ) -> str:
        """Return the selected policy name.

        # IMPLEMENT: optimize KL-regularized objective under constraints.
        """

        if not candidates:
            raise ValueError("No candidate policies provided.")
        # Placeholder: choose the candidate with the highest posterior if present,
        # otherwise fall back to the first candidate deterministically.
        best = None
        best_p = -1.0
        for name in candidates:
            p = float(posteriors.get(name, -1.0))
            if p > best_p:
                best = name
                best_p = p
        return best or candidates[0]


