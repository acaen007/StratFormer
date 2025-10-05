import math
from typing import Dict

def softmax(d: Dict[str, float], tau: float) -> Dict[str, float]:
    exps = {k: math.exp(v/tau) for k, v in d.items()}
    s = sum(exps.values())
    return {k: v/s for k, v in exps.items()}

class SigmaMixer:
    """
    σ(t): mixture over our candidate policies for our agent (player 0).
    Update rule uses opponent-model score per candidate type.
    """
    def __init__(self, pool: Dict[str, object], tau: float = 0.5):
        self.keys = list(pool.keys())
        self.weights = {k: 1.0/len(self.keys) for k in self.keys}
        self.tau = tau

    def update_from_scores(self, scores: Dict[str, float]):
        # e.g., scores are log-likelihoods or accuracy margins from the model
        new = softmax(scores, self.tau)
        self.weights = new

    def current(self) -> Dict[str, float]:
        return dict(self.weights)
z