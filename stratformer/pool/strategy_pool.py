"""Strategy pool for storing and retrieving policies.

Supports both tabular and neural policies via a minimal "Policy" protocol and
stores per-policy metadata.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

__all__ = ["Policy", "StrategyPool"]

logger = logging.getLogger(__name__)


class Policy(Protocol):
    """Minimal policy protocol.

    Policies should be pure functions of the observation (plus optional state) and
    return an integer action index consistent with the environment's action set.
    """

    def act(self, observation: np.ndarray) -> int:  # pragma: no cover - interface only
        ...


class StrategyPool:
    """Container for named policies and associated metadata.

    Notes
    -----
    - Policies are stored under unique string keys.
    - Metadata may include training details, evaluation metrics, etc.
    """

    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add_policy(
        self,
        name: str,
        policy: Policy,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Register a policy with optional metadata.

        Raises
        ------
        ValueError
            If a policy with the same name already exists.
        """

        if name in self._policies:
            raise ValueError(f"Policy already exists: {name}")
        self._policies[name] = policy
        self._metadata[name] = dict(metadata or {})

    def get_policy(self, name: str) -> Policy:
        """Retrieve a policy by name."""

        return self._policies[name]

    def list_policies(self) -> list[str]:
        """Return a list of policy names in insertion order."""

        return list(self._policies.keys())

    def get_metadata(self, name: str) -> Mapping[str, Any]:
        """Return read-only metadata for a policy."""

        return dict(self._metadata.get(name, {}))


