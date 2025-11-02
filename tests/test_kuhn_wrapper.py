from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyspiel")

from stratformer.env.open_spiel_wrapper import KuhnEnv, RandomPolicy


def test_determinism_with_seed():
    env = KuhnEnv()
    p0 = RandomPolicy()
    p1 = RandomPolicy()

    res1 = env.rollout(p0, p1, n_hands=200, seed=123)
    res2 = env.rollout(p0, p1, n_hands=200, seed=123)

    assert res1["p0_avg_return"] == res2["p0_avg_return"]
    assert res1["p1_avg_return"] == res2["p1_avg_return"]


def test_zero_sum_average():
    env = KuhnEnv()
    p0 = RandomPolicy()
    p1 = RandomPolicy()

    res = env.rollout(p0, p1, n_hands=1000, seed=999)
    total = float(res["p0_avg_return"]) + float(res["p1_avg_return"]) 
    assert np.isclose(total, 0.0, rtol=1e-9, atol=1e-12)


