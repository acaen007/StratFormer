"""Novelty detection based on sustained low maximum posterior (stub)."""

from __future__ import annotations

from collections import deque

__all__ = ["NoveltyDetector"]


class NoveltyDetector:
    """Detects novel opponent behavior from posterior time series.

    Parameters
    ----------
    window_size
        Number of recent steps to consider.
    threshold
        Trigger when the maximum posterior across known strategies stays below
        this threshold for the entire window.
    """

    def __init__(self, *, window_size: int = 10, threshold: float = 0.5) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, max_posterior: float) -> bool:
        """Ingest a new maximum posterior value and return whether novelty triggers.

        Notes
        -----
        # IMPLEMENT: support more robust statistics (e.g., EMA, quantiles).
        """

        self._buffer.append(float(max_posterior))
        if len(self._buffer) < self.window_size:
            return False
        # Trigger when the average max-posterior across the window stays below
        # the threshold. This is robust to occasional spikes but still resets
        # once a high value enters the window.
        return (sum(self._buffer) / float(self.window_size)) < self.threshold


