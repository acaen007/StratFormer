#!/usr/bin/env python3
"""
Compare Ridge vs Longformer models for opponent modeling in StratFormer.
"""
import argparse
import pickle
import numpy as np
from core.factory import load_env, load_encoders, load_policy_pool
from core.rollout import play_episode
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, env, encoders, episodes=100):
    """Evaluate a model on test episodes."""
    dense_enc, tok_enc = encoders
    pool = load_policy_pool(env.name)
    our = pool["GTO"]
    opp = pool["OverBluff"]
    
    predictions = []
    true_labels = []
    
    for _ in range(episodes):
        traj, _ = play_episode(env, {0: our, 1: opp})
        for t in traj:
            if t.player == 1:  # opponent actions
                if hasattr(model, 'predict_proba_like'):
                    # Ridge model
                    x, _ = dense_enc.encode(t)
                    pred = model.predict([x])[0]
                else:
                    # Longformer model
                    text = tok_enc.encode(t)
                    probs = model.predict_proba([text])
                    pred = np.argmax(probs[0])
                
                predictions.append(pred)
                true_labels.append(t.action)
    
    return predictions, true_labels

def main():
    parser = argparse.ArgumentParser(description='Compare Ridge vs Longformer models')
    parser.add_argument('--env', default='kuhn', choices=['kuhn', 'leduc'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--ridge_model', default='artifacts/kuhn_ridge_overbluff.pkl')
    parser.add_argument('--longformer_model', default='artifacts/kuhn_longformer_overbluff.bin')
    args = parser.parse_args()
    
    # Load environment and encoders
    env = load_env(args.env)
    encoders = load_encoders(args.env)
    
    print(f"Evaluating models on {args.env} with {args.episodes} episodes")
    print("=" * 50)
    
    # Load and evaluate Ridge model
    try:
        with open(args.ridge_model, 'rb') as f:
            ridge_model = pickle.load(f)
        
        print("🔵 Ridge Model Results:")
        ridge_preds, true_labels = evaluate_model(ridge_model, env, encoders, args.episodes)
        ridge_acc = accuracy_score(true_labels, ridge_preds)
        print(f"Accuracy: {ridge_acc:.3f}")
        print("Classification Report:")
        print(classification_report(true_labels, ridge_preds))
        print()
        
    except FileNotFoundError:
        print(f"❌ Ridge model not found: {args.ridge_model}")
        print()
    
    # Load and evaluate Longformer model
    try:
        with open(args.longformer_model, 'rb') as f:
            longformer_model = pickle.load(f)
        
        print("🟢 Longformer Model Results:")
        longformer_preds, true_labels = evaluate_model(longformer_model, env, encoders, args.episodes)
        longformer_acc = accuracy_score(true_labels, longformer_preds)
        print(f"Accuracy: {longformer_acc:.3f}")
        print("Classification Report:")
        print(classification_report(true_labels, longformer_preds))
        print()
        
    except FileNotFoundError:
        print(f"❌ Longformer model not found: {args.longformer_model}")
        print()
    
    # Summary comparison
    if 'ridge_acc' in locals() and 'longformer_acc' in locals():
        print("📊 Model Comparison Summary:")
        print(f"Ridge Accuracy:      {ridge_acc:.3f}")
        print(f"Longformer Accuracy: {longformer_acc:.3f}")
        print(f"Difference:          {abs(ridge_acc - longformer_acc):.3f}")
        
        if ridge_acc > longformer_acc:
            print("🏆 Ridge model performs better!")
        elif longformer_acc > ridge_acc:
            print("🏆 Longformer model performs better!")
        else:
            print("🤝 Both models perform equally!")

if __name__ == "__main__":
    main()
