"""Bayesian opponent model and posterior tracking.

This module maintains a distribution over opponent strategies and updates it as
evidence arrives. All internal computations are performed in log-space for
numerical stability; public accessors return normalized probabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from stratformer.utils.math_utils import logsumexp

__all__ = ["PosteriorTracker"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PosteriorTracker:
    """Track and update posteriors over a discrete set of strategies.

    Parameters
    ----------
    prior
        Mapping from strategy name to prior probability. Must sum to 1.

    Notes
    -----
    - Internally stores log-probabilities for stability.
    - ``update`` expects evidence as log-likelihoods per strategy.
    - ``get_posteriors`` returns normalized probabilities that sum to 1.
    """

    prior: Mapping[str, float]
    _keys: list[str] = field(init=False, repr=False)
    _log_post: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        probs = np.asarray([float(self.prior[k]) for k in self.prior.keys()], dtype=np.float64)
        if np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0):
            raise ValueError("Prior must be non-negative and sum to 1.")
        # Keep deterministic key order
        self._keys = list(self.prior.keys())
        self._log_post = np.log(probs)

    def update(self, log_likelihoods: Mapping[str, float]) -> None:
        """Update log-posteriors with new evidence.

        Parameters
        ----------
        log_likelihoods
            Mapping of strategy name to log-likelihood of observed data under
            that strategy. Strategies not present are assumed to have
            ``-inf`` (i.e., impossible evidence).

        Notes
        -----
        # IMPLEMENT: extend to time-weighted updates and hierarchical priors.
        """

        ll_vec = np.full_like(self._log_post, fill_value=-np.inf, dtype=np.float64)
        for i, k in enumerate(self._keys):
            if k in log_likelihoods:
                ll_vec[i] = float(log_likelihoods[k])
        # Bayes rule in log-space: log p(h|D) ∝ log p(h) + log p(D|h)
        self._log_post = self._log_post + ll_vec
        # Normalize in log-space
        norm = logsumexp(self._log_post, axis=None, keepdims=True)
        self._log_post = self._log_post - norm

    def get_posteriors(self) -> dict[str, float]:
        """Return normalized posteriors as probabilities that sum to 1."""

        probs = np.exp(self._log_post)
        # Renormalize defensively against tiny numerical drift
        probs = probs / np.sum(probs)
        return {k: float(p) for k, p in zip(self._keys, probs)}

    def reset(self) -> None:
        """Reset to the initial prior distribution.

        # IMPLEMENT: allow custom resets or tempering schedules if needed.
        """

        probs = np.asarray([float(self.prior[k]) for k in self._keys], dtype=np.float64)
        self._log_post = np.log(probs)


