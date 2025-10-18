import argparse, numpy as np
from core.factory import load_env, load_encoders, load_policy_pool
from core.rollout import play_episode
from core.interfaces import Transition
from models.ridge import RidgeOpponentModel

def collect(env_name: str, num_eps: int, opp_name: str):
    env = load_env(env_name)
    dense_enc, _ = load_encoders(env_name)
    pool = load_policy_pool(env_name)
    opp = pool[opp_name]
    # Our agent plays fixed GTO in data collection; opponent = opp_name
    our = pool["GTO"]
    X, y = [], []
    for i in range(num_eps):
        traj, _ = play_episode(env, {0: our, 1: opp})
        print(f"Episode {i}: {len(traj)} transitions")
        for t in traj:
            print(f"  Player {t.player}, Action {t.action}")
            if t.player == 1:  # learn opponent model
                x, legal = dense_enc.encode(t)
                X.append(x)
                y.append(t.action)  # predict opponent action; alternative: type id
    print(f"Total samples: {len(X)}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["kuhn","leduc"])
    ap.add_argument("--opp", default="OverBluff")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--out", default="ridge_model.pkl")
    args = ap.parse_args()

    X, y = collect(args.env, args.episodes, args.opp)
    model = RidgeOpponentModel()
    model.fit(X, y)

    import pickle, os
    os.makedirs("artifacts", exist_ok=True)
    with open(f"artifacts/{args.out}", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved to artifacts/{args.out} with {len(X)} samples.")
