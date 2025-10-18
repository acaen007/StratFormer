# 🧠 multi_env_opponent_modeling_with_openspiel

**Adaptive Opponent Modeling across Multiple Poker Environments (Kuhn & Leduc) using OpenSpiel, Ridge Regression, and Longformer Transformers**

---

## 📘 Overview

This repository generalizes your original **Kuhn-Poker opponent-modeling** setup into a **multi-environment** framework using **DeepMind OpenSpiel** as the backend.

You can now:
- Run experiments across **Kuhn** and **Leduc Hold’em** (and easily extend to others)
- Use **Ridge regression** and **Longformer** models to predict opponent behavior
- Adaptively **reweight strategy mixtures σ(t)** over a pool of archetype policies (OverBluff, OverFold, CallStation, GTO)
- Track **rolling EV, drift phases, and σ(t)** evolution in online adaptive play

The design abstracts away game-specific logic using unified interfaces for:
- `Env` — Environment wrapper (OpenSpiel game)
- `Policy` — Tabular policies (CFR + perturbations)
- `Encoder` — Converts trajectories to features/tokens for models

---

## 🧩 Architecture Summary

```
multi_env_opponent_modeling_with_openspiel/
├── core/
│   ├── interfaces.py
│   ├── factory.py
│   ├── rollout.py
│   ├── reweight.py
│   ├── drift.py
│   └── utils.py
├── envs/
│   ├── openspiel_base.py
│   ├── kuhn.py
│   └── leduc.py
├── encoders/
│   ├── dense_features.py
│   ├── longformer_tokens.py
│   └── registry.py
├── policies/
│   ├── base.py
│   ├── cfr_policies.py
│   └── registry.py
├── models/
│   ├── ridge.py
│   └── longformer.py
├── scripts/
│   ├── train_ridge.py
│   ├── train_longformer.py
│   ├── online_eval.py
│   ├── plotters.py
│   └── notebooks/
│       └── openspiel_env_sanity.ipynb
├── configs/
│   ├── kuhn.yaml
│   └── leduc.yaml
├── data/
│   ├── policies/
│   └── datasets/
├── artifacts/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### Recommended Python version

**Python 3.12.x** is the most stable (OpenSpiel 1.5+).  
Python 3.13 works with OpenSpiel 1.6+, but may require a source build.

### Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Install dependencies

```bash
pip install -r requirements.txt
```

If OpenSpiel fails to install:
```bash
pip install cmake ninja pybind11
pip install open_spiel
```

Validate:
```bash
python - <<'PY'
import pyspiel
print("Games:", len(pyspiel.registered_games()))
print("Has Kuhn?", any("kuhn" in g.short_name for g in pyspiel.registered_games()))
print("Has Leduc?", any("leduc" in g.short_name for g in pyspiel.registered_games()))
PY
```

---

## 🧪 Sanity Check (Notebook)

Run `scripts/notebooks/openspiel_env_sanity.ipynb` to verify both environments.
It will print step-by-step states and plot returns.

---

## 🧮 Training Workflows

### Ridge model

```bash
python scripts/train_ridge.py --env kuhn --opp OverBluff --episodes 8000 --out kuhn_ridge_overbluff.pkl
```

### Longformer model

```bash
python scripts/train_longformer.py --env leduc --opp OverFold --episodes 12000 --out leduc_longformer_overfold.bin
```

---

## 🧠 Online Adaptive Evaluation

```bash
python scripts/online_eval.py   --env kuhn   --drift_list OverBluff,OverFold,GTO   --episodes 12000   --phase_episodes 3000   --ridge_path artifacts/kuhn_ridge_overbluff.pkl   --longformer_path artifacts/kuhn_longformer_overbluff.bin
```

Outputs rolling EV, σ(t), top-1 archetype timeline, and drift phases.

---

## 🧱 Dependencies Summary

| Component | Library |
|------------|----------|
| Environment | open_spiel |
| ML / Models | numpy, scikit-learn, torch, transformers |
| Visualization | matplotlib, seaborn |
| IO & Config | pandas, yaml, tqdm |
| Dev utilities | jupyter, ipykernel |
| Optional build tools | cmake, ninja, pybind11 |

---

## 🔧 Troubleshooting

| Symptom | Fix |
|----------|-----|
| `player = -1` error | Don’t call `information_state_string()` at chance nodes. Use `state.action_to_string(a)` instead. |
| `CMake must be installed` | `pip install cmake ninja pybind11` |
| `ImportError: pyspiel` | venv not activated or OpenSpiel build incomplete |
| Memory errors | Reduce Longformer sequence length or batch size |

---

## 🧩 Work Remaining

- [ ] Improve encoders (pot size, street, bet sizing bins)
- [ ] Cache CFR policies to `data/policies/`
- [ ] Integrate multi-model weighting (Ridge + Longformer jointly)
- [ ] Restore EV/σ plotting scripts
- [ ] Add YAML experiment configs
- [ ] Add pytest tests for envs and encoders

---

## 🚀 Next Steps

| Goal | How |
|------|------|
| Add new OpenSpiel games | Wrap `pyspiel.load_game("<game>")` in a new `Env` subclass |
| Scale datasets | Use OpenSpiel rollouts for 10k+ episodes across envs |
| Extend Longformer | Add multi-task head for action + opponent type prediction |
| Evaluation | Compare adaptive agent vs static CFR opponents |

---

## 🧩 License & Citation

Uses OpenSpiel (Apache 2.0) — cite DeepMind’s paper if used in academic work.

---

© 2025 — Opponent Modeling Framework (StratFormer Multi-Env Edition)
