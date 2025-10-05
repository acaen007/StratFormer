from typing import List, Tuple
from core.interfaces import DenseEncoder, Transition

# --- Shared primitives

def pad_or_trim(vec: List[float], size: int) -> List[float]:
    return (vec + [0.0] * (size - len(vec)))[:size]

# --- Kuhn

class KuhnDenseEncoder(DenseEncoder):
    env_name = "kuhn"
    MAX_LEN = 16  # history window
    def encode(self, traj: Transition) -> Tuple[List[float], List[int]]:
        hist = traj.history  # action ids
        # simple features: one-hot of last K actions + player id bias + pot proxy (optional)
        feats = []
        for a in hist[-self.MAX_LEN:]:
            feats.extend([1.0 if i == a else 0.0 for i in range(max(traj.legal_actions)+1)])
        feats.append(float(traj.player))
        x = pad_or_trim(feats, 1 + self.MAX_LEN * (max(traj.legal_actions)+1))
        return x, traj.legal_actions

# --- Leduc (start with same idea, extend later with street markers and bet sizing bins)

class LeducDenseEncoder(DenseEncoder):
    env_name = "leduc"
    MAX_LEN = 32
    def encode(self, traj: Transition) -> Tuple[List[float], List[int]]:
        feats = []
        for a in traj.history[-self.MAX_LEN:]:
            feats.extend([1.0 if i == a else 0.0 for i in range(max(traj.legal_actions)+1)])
        feats.append(float(traj.player))
        return pad_or_trim(feats, 1 + self.MAX_LEN * (max(traj.legal_actions)+1)), traj.legal_actions
