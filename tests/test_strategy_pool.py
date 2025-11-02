import numpy as np

from stratformer.pool.strategy_pool import Policy, StrategyPool


class _DummyPolicy(Policy):
    def act(self, observation: np.ndarray) -> int:  # noqa: D401 - trivial
        return 0


def test_strategy_pool_add_get_list():
    pool = StrategyPool()
    pool.add_policy("p1", _DummyPolicy(), metadata={"foo": 1})
    assert pool.list_policies() == ["p1"]
    assert isinstance(pool.get_policy("p1"), _DummyPolicy)
    md = pool.get_metadata("p1")
    assert md["foo"] == 1


