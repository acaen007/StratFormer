"""Feature featurizer for observations.

Transforms raw observations into model features (e.g., normalization,
concatenation of context, etc.).
"""

from __future__ import annotations

import numpy as np

__all__ = ["Featurizer"]


class Featurizer:
    """Transforms observation arrays into feature arrays.

    Keep this minimal and typed; task-specific feature engineering can live in
    experimental code.
    """

    def featurize(self, observation: np.ndarray) -> np.ndarray:  # pragma: no cover - interface only
        """Return a feature array derived from ``observation``.

        # IMPLEMENT: add feature transformations appropriate for the model family.
        """

        raise NotImplementedError


