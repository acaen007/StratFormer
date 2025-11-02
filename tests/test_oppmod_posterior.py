import numpy as np

from stratformer.oppmod.bayes_model import PosteriorTracker


def test_posterior_normalizes_and_updates_in_right_direction():
    prior = {"a": 0.6, "b": 0.4}
    tracker = PosteriorTracker(prior=prior)

    # Evidence favors b by factor 2
    tracker.update({"a": 0.0, "b": np.log(2.0)})
    post = tracker.get_posteriors()
    total = sum(post.values())
    assert np.isclose(total, 1.0)
    assert post["b"] > prior["b"]


