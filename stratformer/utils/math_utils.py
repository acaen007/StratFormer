"""Numerically stable math utilities.

These helpers should be used for all likelihood/probability computations to
ensure stable behavior in log-space.
"""

from __future__ import annotations

import logging

import numpy as np

__all__ = ["logsumexp", "stable_log_softmax"]

logger = logging.getLogger(__name__)


def logsumexp(
    a: np.ndarray, *, axis: int | None = None, keepdims: bool = False
) -> np.ndarray:
    """Compute log(sum(exp(a))) in a numerically stable way.

    Parameters
    ----------
    a
        Input array of log-values.
    axis
        Axis over which to reduce. If ``None``, reduce over the entire array.
    keepdims
        If ``True``, retains reduced dimensions with length 1.

    Returns
    -------
    np.ndarray
        The log-sum-exp of ``a`` along ``axis``.

    Notes
    -----
    Stability is achieved by subtracting the maximum value before exponentiating.
    """

    if not isinstance(a, np.ndarray):
        a = np.asarray(a)

    # Subtract the maximum for numerical stability
    max_a = np.max(a, axis=axis, keepdims=True)
    # Handle all -inf case: max(-inf) = -inf; subtracting yields nan if not guarded
    max_a = np.where(np.isfinite(max_a), max_a, 0.0)

    shifted = a - max_a
    sum_exp = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = np.log(sum_exp) + max_a

    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def stable_log_softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Compute log-softmax in a numerically stable way.

    Parameters
    ----------
    logits
        Real-valued scores. Can be any shape.
    axis
        Axis along which to apply softmax.

    Returns
    -------
    np.ndarray
        Log-probabilities with the same shape as ``logits``.

    Notes
    -----
    Computes ``log_softmax(x) = x - logsumexp(x)`` with stability via max-shift.
    """

    if not isinstance(logits, np.ndarray):
        logits = np.asarray(logits)

    max_logits = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_logits
    lse = logsumexp(shifted, axis=axis, keepdims=True)
    return shifted - lse


