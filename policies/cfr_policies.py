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

    # Use the OpenSpiel TabularPolicy directly as our table
    # We'll create a wrapper that makes it compatible with our interface
    table = avg.to_tabular()
    return TabularPolicy(name=f"{game_name}_cfr", env_name=game_name.split("_")[0], table=table)

def perturb_overbluff(tab: TabularPolicy, bet_like_ids: set[int], scale: float = 1.5) -> TabularPolicy:
    # For now, return the original policy as we need to implement proper perturbation
    # TODO: Implement proper perturbation for OpenSpiel TabularPolicy
    return TabularPolicy(name=f"{tab.name}_OverBluff", env_name=tab.env_name, table=tab.table)

def perturb_overfold(tab: TabularPolicy, fold_id: int, scale: float = 1.5) -> TabularPolicy:
    # For now, return the original policy as we need to implement proper perturbation
    # TODO: Implement proper perturbation for OpenSpiel TabularPolicy
    return TabularPolicy(name=f"{tab.name}_OverFold", env_name=tab.env_name, table=tab.table)

def call_station(tab: TabularPolicy, call_id: int, scale: float = 1.5) -> TabularPolicy:
    # For now, return the original policy as we need to implement proper perturbation
    # TODO: Implement proper perturbation for OpenSpiel TabularPolicy
    return TabularPolicy(name=f"{tab.name}_CallStation", env_name=tab.env_name, table=tab.table)
