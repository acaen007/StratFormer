from .cfr_policies import compute_cfr_tabular, perturb_overbluff, perturb_overfold, call_station

def load_baselines(env_name: str):
    game_name = {"kuhn":"kuhn_poker", "leduc":"leduc_poker"}[env_name]
    base = compute_cfr_tabular(game_name, iterations=5000)
    # NOTE: action ids vary by game; map them appropriately.
    if env_name == "kuhn":
        bet_like = {1}     # example
        fold_id = 3        # example
        call_id = 2
    else:
        bet_like = {1,4}   # bet/raise ids
        fold_id = 3
        call_id = 2
    return {
        "GTO": base,
        "OverBluff": perturb_overbluff(base, bet_like),
        "OverFold":  perturb_overfold(base, fold_id),
        "CallStation": call_station(base, call_id),
    }
