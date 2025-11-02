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



def test_random_policy_reasonable_returns():
    env = KuhnEnv()
    p0 = RandomPolicy()
    p1 = RandomPolicy()

    res = env.rollout(p0, p1, n_hands=1000, seed=42)
    p0_ret = float(res["p0_avg_return"])
    p1_ret = float(res["p1_avg_return"])

    assert np.isfinite(p0_ret)
    assert np.isfinite(p1_ret)
    assert -1.0 <= p0_ret <= 1.0
    assert -1.0 <= p1_ret <= 1.0


def test_different_seeds_produce_different_outcomes():
    env = KuhnEnv()
    p0 = RandomPolicy()
    p1 = RandomPolicy()

    res1 = env.rollout(p0, p1, n_hands=2001, seed=1)
    res2 = env.rollout(p0, p1, n_hands=2001, seed=2)

    # Determinism with different seeds should lead to different aggregate outcomes
    same_p0 = float(res1["p0_avg_return"]) == float(res2["p0_avg_return"])
    same_p1 = float(res1["p1_avg_return"]) == float(res2["p1_avg_return"])
    assert not (same_p0 and same_p1)

