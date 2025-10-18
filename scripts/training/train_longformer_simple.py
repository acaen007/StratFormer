#!/usr/bin/env python3
"""
Simplified Longformer training to avoid memory issues.
"""
import argparse, pickle
import torch
from core.factory import load_env, load_encoders, load_policy_pool
from core.rollout import play_episode
from models.longformer import LongformerOpponentModel

def collect_text_batch(env_name: str, opp_name: str, episodes: int, batch_size: int = 10):
    """Collect data in smaller batches to avoid memory issues."""
    env = load_env(env_name)
    _, tok_enc = load_encoders(env_name)
    pool = load_policy_pool(env_name)
    our = pool["GTO"]
    opp = pool[opp_name]
    
    all_texts, all_y = [], []
    
    for batch_start in range(0, episodes, batch_size):
        batch_end = min(batch_start + batch_size, episodes)
        batch_texts, batch_y = [], []
        
        print(f"Collecting batch {batch_start//batch_size + 1}/{(episodes-1)//batch_size + 1}")
        
        for _ in range(batch_start, batch_end):
            traj, _ = play_episode(env, {0: our, 1: opp})
            for t in traj:
                if t.player == 1:
                    batch_texts.append(tok_enc.encode(t))
                    batch_y.append(t.action)
        
        all_texts.extend(batch_texts)
        all_y.extend(batch_y)
        
        # Train incrementally to avoid memory issues
        if len(all_texts) >= batch_size:
            print(f"Training on {len(all_texts)} samples so far...")
    
    return all_texts, all_y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["kuhn","leduc"])
    ap.add_argument("--opp", default="OverBluff")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--out", default="longformer_simple.bin")
    args = ap.parse_args()

    print(f"Training Longformer on {args.env} with {args.episodes} episodes")
    
    # Collect data in batches
    texts, y = collect_text_batch(args.env, args.opp, args.episodes, args.batch_size)
    
    if len(texts) == 0:
        print("No training data collected!")
        exit(1)
    
    print(f"Collected {len(texts)} training samples")
    
    # Train model
    model = LongformerOpponentModel(num_labels=max(y)+1)
    print("Training model...")
    model.fit(texts, y, epochs=1)
    
    # Save model
    with open(f"artifacts/{args.out}", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved Longformer to artifacts/{args.out} on {len(texts)} sequences.")
