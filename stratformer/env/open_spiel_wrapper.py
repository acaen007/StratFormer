"""Thin wrapper around OpenSpiel poker environments.

The wrapper abstracts environment step/reset and provides typed interfaces that
are stable across different poker games (Kuhn, Leduc, Hold'em with abstraction).

Advanced logic should not live here; this module defines the interface and
expected shapes. Concrete implementations may delegate to OpenSpiel at runtime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Mapping, Protocol, Tuple

import numpy as np

from stratformer.utils.seed import seed_all

if TYPE_CHECKING:  # pragma: no cover - only for static type checkers
    import pyspiel

__all__ = ["OpenSpielEnv", "Policy", "RandomPolicy", "KuhnEnv"]

logger = logging.getLogger(__name__)


class OpenSpielEnv:
    """Environment wrapper interface.

    Parameters
    ----------
    game_name
        OpenSpiel game name (e.g., ``kuhn_poker``).
    num_players
        Number of players in the environment.

    Notes
    -----
    - Observations are returned as NumPy arrays to be featurized downstream.
    - Rewards are per-step scalars from the perspective of the acting player.
    - This class should encapsulate OpenSpiel-specific details and expose a
      minimal interface used elsewhere in the codebase.
    """

    def __init__(self, game_name: str, num_players: int = 2) -> None:
        self.game_name = game_name
        self.num_players = num_players

    def reset(self) -> np.ndarray:  # noqa: D401 - short docstrings for interface
        """Reset the environment and return the initial observation.

        # IMPLEMENT: delegate to OpenSpiel; return observation for current player.
        """

        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply ``action`` and return ``(obs, reward, done, info)``.

        # IMPLEMENT: step through OpenSpiel; return next observation for the new
        # current player, with reward/terminal flags aligned to RL conventions.
        """

        raise NotImplementedError

    def legal_actions(self) -> np.ndarray:
        """Return the legal actions for the current player as an array.

        # IMPLEMENT: surface OpenSpiel legal actions for the current state.
        """

        raise NotImplementedError

    def current_player(self) -> int:
        """Return the current player index.

        # IMPLEMENT: return current player from underlying OpenSpiel state.
        """

        raise NotImplementedError

    def observation_shape(self) -> Tuple[int, ...]:
        """Return the shape of a single observation array.

        # IMPLEMENT: expose observation tensor shape for allocation.
        """

        raise NotImplementedError



class Policy(Protocol):
    """Policy protocol compatible with OpenSpiel states.

    Implementations return a probability distribution over legal actions.
    """

    def action_probabilities(self, state: "pyspiel.State") -> Mapping[int, float]:  # noqa: D401 - short
        """Return mapping from action id to probability for ``state``."""


class RandomPolicy:
    """Uniform random policy over legal actions."""

    def action_probabilities(self, state: Any) -> Mapping[int, float]:  # noqa: D401 - short
        legal_actions = state.legal_actions()
        if not legal_actions:
            return {}
        prob = 1.0 / float(len(legal_actions))
        return {int(a): prob for a in legal_actions}


class KuhnEnv:
    """Minimal Kuhn Poker wrapper using OpenSpiel.

    Provides deterministic rollouts between two policies under a given seed.
    """

    def __init__(self) -> None:
        # Lazy import to avoid hard dependency at package import time
        import pyspiel

        self._game = pyspiel.load_game("kuhn_poker")

    def rollout(
        self,
        policy_a: Policy,
        policy_b: Policy,
        n_hands: int,
        seed: int,
    ) -> Dict[str, float | int]:
        """Play ``n_hands`` hands and report average returns.

        Parameters
        ----------
        policy_a
            Policy for player 0.
        policy_b
            Policy for player 1.
        n_hands
            Number of independent hands to simulate.
        seed
            Random seed for deterministic sampling.

        Returns
        -------
        dict
            Keys: ``{"p0_avg_return", "p1_avg_return", "hands"}``
        """

        seed_all(seed)
        rng = np.random.default_rng(int(seed))

        total_p0 = 0.0
        total_p1 = 0.0

        for _ in range(int(n_hands)):
            state = self._game.new_initial_state()

            while not state.is_terminal():
                if state.is_chance_node():
                    action = _sample_chance_action(state, rng)
                    state.apply_action(int(action))
                    continue

                player = state.current_player()
                policy = policy_a if player == 0 else policy_b
                action = _sample_policy_action(state, policy, rng)
                state.apply_action(int(action))

            returns = state.returns()
            total_p0 += float(returns[0])
            total_p1 += float(returns[1])

        p0_avg = total_p0 / float(n_hands)
        p1_avg = total_p1 / float(n_hands)

        return {"p0_avg_return": p0_avg, "p1_avg_return": p1_avg, "hands": int(n_hands)}


def _sample_chance_action(state: Any, rng: np.random.Generator) -> int:
    outcomes = state.chance_outcomes()  # list[(action, prob)]
    actions = np.fromiter((int(a) for a, _ in outcomes), dtype=int)
    probs = np.fromiter((float(p) for _, p in outcomes), dtype=float)
    # Normalize for defensive robustness against float drift from upstream
    probs = probs / probs.sum()
    idx = int(rng.choice(len(actions), p=probs))
    return int(actions[idx])


def _sample_policy_action(
    state: Any, policy: Policy, rng: np.random.Generator
) -> int:
    dist = policy.action_probabilities(state)
    if not dist:
        # Fall back to uniform over legal actions if policy returns empty.
        legal = state.legal_actions()
        if not legal:
            raise RuntimeError("No legal actions available.")
        prob = 1.0 / float(len(legal))
        dist = {int(a): prob for a in legal}

    actions = np.fromiter((int(a) for a in dist.keys()), dtype=int)
    probs = np.fromiter((float(p) for p in dist.values()), dtype=float)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(actions), p=probs))
    return int(actions[idx])

