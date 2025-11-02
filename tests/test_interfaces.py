from stratformer.counter.selector import Selector
from stratformer.env.open_spiel_wrapper import OpenSpielEnv
from stratformer.eval.evaluator import Evaluator
from stratformer.eval.tournament import Tournament
from stratformer.features.featurizer import Featurizer
from stratformer.novelty.novelty_detector import NoveltyDetector
from stratformer.oppmod.bayes_model import PosteriorTracker
from stratformer.pool.strategy_pool import StrategyPool


def test_classes_exist_and_basic_signatures():
    _ = OpenSpielEnv("kuhn_poker")
    _ = Evaluator()
    _ = Tournament()
    _ = Featurizer()
    _ = NoveltyDetector()
    _ = Selector()
    _ = PosteriorTracker(prior={"s1": 0.5, "s2": 0.5})
    _ = StrategyPool()


