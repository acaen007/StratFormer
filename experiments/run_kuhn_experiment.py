"""Run a minimal Kuhn Poker experiment.

This script demonstrates wiring together config loading, seeding, environment
construction, a strategy pool, and evaluation. Advanced logic intentionally
omitted.
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from stratformer.env.open_spiel_wrapper import OpenSpielEnv
from stratformer.eval.evaluator import Evaluator
from stratformer.pool.strategy_pool import Policy, StrategyPool
from stratformer.utils.config import ExperimentConfig, load_experiment_config
from stratformer.utils.seed import set_global_seed


def _create_output_dir(cfg: ExperimentConfig) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out = Path(cfg.output_dir) / cfg.exp_name / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kuhn/baseline.yaml",
        help="Path to experiment YAML config.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = load_experiment_config(args.config)
    set_global_seed(cfg.seed, deterministic_torch=False)
    out_dir = _create_output_dir(cfg)

    env = OpenSpielEnv(cfg.game_name, num_players=2)

    class DummyPolicy(Policy):  # minimal placeholder policy
        def act(self, observation: np.ndarray) -> int:  # noqa: D401 - trivial
            # IMPLEMENT: replace with an actual tabular/NN policy.
            return 0

    pool = StrategyPool()
    pool.add_policy("baseline", DummyPolicy(), metadata={"game": cfg.game_name})

    evaluator = Evaluator()
    metrics = evaluator.evaluate(pool.get_policy("baseline"), env, num_episodes=cfg.num_episodes)

    # Persist minimal CSV metrics
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    logging.info("Saved metrics to %s", csv_path)


if __name__ == "__main__":
    main()


