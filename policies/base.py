import random
from typing import Dict
from core.interfaces import Policy, Env, PlayerID, Action

class TabularPolicy(Policy):
    def __init__(self, name: str, env_name: str, table: Dict[str, Dict[Action, float]]):
        self.name = name
        self.env_name = env_name
        self.table = table  # key: info_state_string(player) → {action: prob}

    def action_probs(self, env: Env, player: PlayerID):
        key = env.observation(player)
        probs = self.table.get(key, None)
        if probs is None:
            # uniform over legal actions as fallback
            legal = env.legal_actions()
            return {a: 1.0/len(legal) for a in legal}
        # mask illegal if needed
        legal = set(env.legal_actions())
        return {a: p for a, p in probs.items() if a in legal}

    def act(self, env: Env, player: PlayerID, rng=None):
        rng = rng or random
        items = list(self.action_probs(env, player).items())
        actions, probs = zip(*items)
        r = rng.random()
        cum = 0.0
        for a, p in items:
            cum += p
            if r <= cum: return a
        return actions[-1]
