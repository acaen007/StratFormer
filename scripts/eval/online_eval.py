import argparse, pickle, numpy as np, random
from collections import defaultdict, deque
from core.factory import load_env, load_policy_pool, load_encoders
from core.rollout import play_episode
from core.reweight import SigmaMixer

def choose_policy_from_sigma(pool, sigma_weights, rng):
    keys, w = zip(*sigma_weights.items())
    r = rng.random()
    cum = 0.0
    for k, p in zip(keys, w):
        cum += p
        if r <= cum: return k
    return keys[-1]

def make_drift_list(env_name, names_csv):
    pool = load_policy_pool(env_name)
    order = [n.strip() for n in names_csv.split(",")]
    return [pool[n] for n in order], order

def main(args):
    rng = random.Random(args.seed)
    env = load_env(args.env)
    dense_enc, tok_enc = load_encoders(args.env)
    pool = load_policy_pool(args.env)  # {"GTO":..., "OverBluff":...}

    # Load models
    ridge = pickle.load(open(args.ridge_path, "rb")) if args.ridge_path else None
    longf = pickle.load(open(args.longformer_path, "rb")) if args.longformer_path else None

    opp_list, opp_names = make_drift_list(args.env, args.drift_list)
    opp_idx, phase_len = 0, args.phase_episodes
    opp = opp_list[opp_idx]

    sigma = SigmaMixer(pool={k:v for k,v in pool.items() if k!="GTO"}, tau=args.tau)  # mix among archetypes for *our* policy
    logs = {"ev":[], "sigma":[], "top1":[], "phase":[]}

    sigma_window_scores = defaultdict(lambda: deque(maxlen=args.window))

    for ep in range(args.episodes):
        # Choose our current policy by σ(t)
        sigma_choice = choose_policy_from_sigma({k:pool[k] for k in sigma.weights.keys()}, sigma.current(), rng)
        our_policy = pool[sigma_choice]
        traj, rets = play_episode(env, {0: our_policy, 1: opp})
        logs["ev"].append(rets[0])
        logs["phase"].append(opp_names[opp_idx])

        # Update opponent-model scores from traj (only steps where player==1)
        seqs, X = [], []
        acts = []
        for t in traj:
            if t.player == 1:
                x,_ = dense_enc.encode(t)
                X.append(x)
                seqs.append(tok_enc.encode(t))
                acts.append(t.action)

        scores = {}
        if ridge and X:
            proba_like = ridge.predict_proba_like(np.array(X))  # [N,C]
            # Use the prob assigned to true action as "evidence"
            score = float(np.mean([p[a] for p,a in zip(proba_like, acts)]))
            scores["ridge"] = np.log(score + 1e-9)

        if longf and seqs:
            P = longf.predict_proba(seqs)  # [N,C]
            score = float(np.mean([p[a] for p,a in zip(P, acts)]))
            scores["longformer"] = np.log(score + 1e-9)

        # Combine model evidences into per-archetype scores (simple: same for all, or map via separate per-type models)
        # Here: use same aggregate evidence to upweight the currently most-likely archetype by action-likelihood heuristic.
        # Placeholder: uniform per type boosted by evidence sum.
        agg = {k: sum(scores.values()) for k in sigma.weights.keys()}
        # Smooth with rolling window
        for k,v in agg.items():
            sigma_window_scores[k].append(v)
            agg[k] = sum(sigma_window_scores[k]) / len(sigma_window_scores[k])

        sigma.update_from_scores(agg)
        logs["sigma"].append(sigma.current())
        logs["top1"].append(max(sigma.current(), key=sigma.current().get))

        # Handle drift phases
        if (ep+1) % phase_len == 0:
            opp_idx = (opp_idx + 1) % len(opp_list)
            opp = opp_list[opp_idx]

    # TODO: save logs for plotting
    print("Final σ:", sigma.current())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["kuhn","leduc"])
    ap.add_argument("--drift_list", required=True, help="e.g. OverBluff,OverFold,GTO")
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--phase_episodes", type=int, default=1000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--ridge_path", type=str, default=None)
    ap.add_argument("--longformer_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    main(args)
