import argparse, pickle
from core.factory import load_env, load_encoders, load_policy_pool
from core.rollout import play_episode
from models.longformer import LongformerOpponentModel

def collect_text(env_name: str, opp_name: str, episodes: int):
    env = load_env(env_name)
    _, tok_enc = load_encoders(env_name)
    pool = load_policy_pool(env_name)
    our = pool["GTO"]; opp = pool[opp_name]
    texts, y = [], []
    for _ in range(episodes):
        traj, _ = play_episode(env, {0: our, 1: opp})
        for t in traj:
            if t.player == 1:
                texts.append(tok_enc.encode(t))
                y.append(t.action)
    return texts, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["kuhn","leduc"])
    ap.add_argument("--opp", default="OverBluff")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--out", default="longformer.bin")
    args = ap.parse_args()

    texts, y = collect_text(args.env, args.opp, args.episodes)
    model = LongformerOpponentModel(num_labels=max(y)+1)
    model.fit(texts, y, epochs=1)
    with open(f"artifacts/{args.out}", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved Longformer to artifacts/{args.out} on {len(texts)} sequences.")
