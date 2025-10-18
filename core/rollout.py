from typing import Dict, List, Tuple
from core.interfaces import Env, Policy, Transition

def play_episode(env: Env, policies: Dict[int, Policy]) -> Tuple[List[Transition], List[float]]:
    env.reset()
    traj: List[Transition] = []
    while not env.is_terminal():
        p = env.current_player()
        if p < 0:  # Chance node, sample random action
            legal_actions = env.legal_actions()
            if legal_actions:
                import random
                a = random.choice(legal_actions)
                env.step(a)
            else:
                break
        else:  # Player node
            t = Transition(
                obs=env.observation(p),
                info=env.info(),
                legal_actions=env.legal_actions(),
                history=env.history().copy(),
                player=p,
            )
            a = policies[p].act(env, p)
            t.action = a
            env.step(a)
            traj.append(t)
    rets = env.returns()
    for t in traj:
        t.terminal = True
    return traj, rets
