"""Configuration utilities for experiments.

Provides a typed dataclass for experiment configuration and a YAML loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ExperimentConfig", "load_experiment_config"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExperimentConfig:
    """Typed configuration for an experiment.

    Attributes
    ----------
    game_name
        OpenSpiel game name (e.g., ``kuhn_poker``).
    seed
        Global random seed.
    num_episodes
        Number of episodes to run during evaluation.
    exp_name
        Name of the experiment; used for artifact paths.
    output_dir
        Base directory for artifacts.
    """

    game_name: str
    seed: int
    num_episodes: int
    exp_name: str
    output_dir: str = "artifacts"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from a YAML file.

    Parameters
    ----------
    path
        Path to a YAML file on disk.

    Returns
    -------
    ExperimentConfig
        Parsed and validated configuration.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        try:
            # Lazy import to keep PyYAML optional for library users
            import yaml  # type: ignore

            data: dict[str, Any] = yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001 - third-party import/parse can raise various
            raise RuntimeError(f"Failed to parse YAML config at {path}: {exc}") from exc

    required = {"game_name", "seed", "num_episodes", "exp_name"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    return ExperimentConfig(
        game_name=str(data["game_name"]),
        seed=int(data["seed"]),
        num_episodes=int(data["num_episodes"]),
        exp_name=str(data["exp_name"]),
        output_dir=str(data.get("output_dir", "artifacts")),
    )


