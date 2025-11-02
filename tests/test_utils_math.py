import numpy as np

from stratformer.utils.math_utils import logsumexp, stable_log_softmax


def test_logsumexp_matches_reference():
    x = np.array([[1.0, 2.0, 3.0], [-10.0, -5.0, 0.0]])
    ref = np.log(np.sum(np.exp(x), axis=1))
    got = logsumexp(x, axis=1)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_stable_log_softmax_normalizes():
    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    log_probs = stable_log_softmax(logits, axis=1)
    probs = np.exp(log_probs)
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, np.array([1.0, 1.0]), rtol=1e-6, atol=1e-9)


