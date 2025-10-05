import numpy as np
from sklearn.linear_model import RidgeClassifier

class RidgeOpponentModel:
    """
    Trains a classifier: x(history) -> opponent action distribution (or type).
    Output can be: (a) action id; or (b) opponent type id.
    Keep `predict_proba` for reweighting σ.
    """
    def __init__(self, **ridge_kwargs):
        self.clf = RidgeClassifier(**ridge_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.clf.predict(X)

    def predict_proba_like(self, X: np.ndarray):
        # RidgeClassifier doesn't give proba; convert decision function via softmax.
        z = self.clf.decision_function(X)  # shape [N, C]
        e = np.exp(z - z.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return p
