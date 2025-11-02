"""Longformer-based opponent model (optional dependency, stub).

This module should only be used when the optional "longformer" extra is
installed. Import guards prevent hard dependency errors at import-time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["LongformerOpponentModel"]


class LongformerOpponentModel:
    """Sequence model for opponent behavior using Longformer (stub).

    Notes
    -----
    # IMPLEMENT: wire up transformers Longformer with task-specific heads.
    """

    def __init__(self, *, model_name: str = "allenai/longformer-base-4096") -> None:
        try:
            import transformers  # noqa: F401
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "LongformerOpponentModel requires the 'longformer' extra. "
                "Install via: pip install .[longformer]"
            ) from exc
        self.model_name = model_name
        # IMPLEMENT: load model/tokenizer lazily when first used.

    def forward(self, features: Any) -> Any:  # pragma: no cover - stub
        """Forward pass placeholder."""

        raise NotImplementedError


