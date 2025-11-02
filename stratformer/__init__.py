"""StratFormer core package.

This package provides research scaffolding for OpenSpiel-based poker experiments
scaling from Kuhn → Leduc → Hold'em (abstracted). Public interfaces are kept
minimal and stable; experimental implementations should live under
`experiments/`.

Design invariants:
- Deterministic seeding via `stratformer.utils.seed`.
- Numerical stability via log-space utilities in `stratformer.utils.math_utils`.
- Policies, models, and evaluators expose small, typed interfaces.

Note: This module exports the public API surface via `__all__`.
"""

from .counter.selector import Selector
from .env.open_spiel_wrapper import OpenSpielEnv
from .eval.evaluator import Evaluator
from .eval.tournament import Tournament
from .features.featurizer import Featurizer
from .oppmod.bayes_model import PosteriorTracker
from .pool.strategy_pool import StrategyPool
from .utils.math_utils import logsumexp, stable_log_softmax
from .utils.seed import set_global_seed

__all__ = [
    "logsumexp",
    "stable_log_softmax",
    "set_global_seed",
    "OpenSpielEnv",
    "StrategyPool",
    "PosteriorTracker",
    "Selector",
    "NoveltyDetector",
    "Evaluator",
    "Tournament",
    "Featurizer",
]


