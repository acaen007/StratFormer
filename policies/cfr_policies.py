from typing import Dict
import pyspiel
from open_spiel.python.algorithms import cfr as cfr_py  # available in recent OpenSpiel
from .base import TabularPolicy

def compute_cfr_tabular(game_name: str, iterations: int = 10000) -> TabularPolicy:
    game = pyspiel.load_game(game_name)
    algo = cfr_py.CFRSolver(game)
    for _ in range(iterations):
        algo.evaluate_and_update_policy()
    avg = algo.average_policy()

    table: Dict[str, Dict[int, float]] = {}
    # Extract for all states where current player is decision-maker
    for state in game.new_initial_state().legal_actions():  # dummy loop to access game tree lazily
        pass
    # The canonical way in OpenSpiel:
    for info_state_key, dist in avg.policy_table().items():
        # dist: mapping action → prob
        table[info_state_key] = dict(dist.items())
    return TabularPolicy(name=f"{game_name}_cfr", env_name=game_name.split("_")[0], table=table)

def perturb_overbluff(tab: TabularPolicy, bet_like_ids: set[int], scale: float = 1.5) -> TabularPolicy:
    new = {}
    for k, dist in tab.table.items():
        total = sum(dist.values())
        if total <= 0: continue
        d = dict(dist)
        for a in d:
            if a in bet_like_ids:
                d[a] *= scale
        # renorm
        s = sum(d.values())
        new[k] = {a: v/s for a, v in d.items()}
    return TabularPolicy(name=f"{tab.name}_OverBluff", env_name=tab.env_name, table=new)

def perturb_overfold(tab: TabularPolicy, fold_id: int, scale: float = 1.5) -> TabularPolicy:
    new = {}
    for k, dist in tab.table.items():
        d = dict(dist)
        if fold_id in d: d[fold_id] *= scale
        s = sum(d.values())
        new[k] = {a: v/s for a, v in d.items()}
    return TabularPolicy(name=f"{tab.name}_OverFold", env_name=tab.env_name, table=new)

def call_station(tab: TabularPolicy, call_id: int, scale: float = 1.5) -> TabularPolicy:
    new = {}
    for k, dist in tab.table.items():
        d = dict(dist)
        if call_id in d: d[call_id] *= scale
        s = sum(d.values())
        new[k] = {a: v/s for a, v in d.items()}
    return TabularPolicy(name=f"{tab.name}_CallStation", env_name=tab.env_name, table=new)
