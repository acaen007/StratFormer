# Stratformer: Adaptive Multi-Strategy Poker Agent

**Goal:** Build a poker agent that can **detect, model, and exploit suboptimal opponents** while **remaining near-unexploitable** (close to GTO).  
The system learns to reason over a *pool of known strategies*, estimate a **posterior distribution** over likely opponent types, and **select safe counter-strategies** that maximize exploitative EV subject to a GTO deviation budget.

---

## 🔍 Motivation
Traditional Game-Theory-Optimal (GTO) poker agents are unexploitable but fail to capitalize on weak opponents.  
Conversely, purely exploitative agents win more short-term but can be counter-exploited.  
Stratformer unifies both: an *adaptive* agent that balances **robustness** and **adaptivity** through online opponent modeling.

---

## 🧩 Architecture Overview

| Module | Role |
|---------|------|
| `env/` | Wrappers for OpenSpiel games (Kuhn, Leduc, Hold’em). |
| `pool/` | Stores GTO and learned policies, supports sampling and evaluation. |
| `oppmod/` | Opponent modeling via Bayesian, Ridge, and Transformer methods. |
| `novelty/` | Detects when an opponent deviates from all known types. |
| `counter/` | Chooses counter-strategies under KL or exploitability constraints. |
| `online_adapt/` | Lightweight adaptation (LoRA, AutoStep) of neural policies. |
| `eval/` | Computes exploitability, EV, and runs tournaments. |
| `experiments/` | Scripts for Kuhn → Leduc → Hold’em experiments. |

---

## 🧠 Core Algorithm (posterior + counter selection)

1. **Track posterior** over pool strategies based on observed actions.  
2. **Select counter** maximizing expected EV – λ × KL(π‖π_GTO).  
3. **Detect novelty** if posterior confidence drops below α.  
4. **Add new strategy** to the pool and continue learning.  

Mathematically:
\[
\pi^* = \arg\max_{\pi} \mathbb{E}_{\theta \sim P(\theta)} [EV(\pi,\theta)] - \lambda D_{KL}(\pi || \pi_{GTO})
\]

---

## 🧪 Planned Experiments

1. **Kuhn Poker** (tabular policies)  
   - Validate posterior updates & detection.  
   - Compare to pure GTO and naive exploiter.

2. **Leduc Poker** (tabular + small NN policies)  
   - Test scalability and online adaptation.

3. **Texas Hold’em (abstracted)**  
   - Introduce sequence models (Longformer).  
   - Evaluate detection latency and exploitability trade-offs.

---

## 📊 Evaluation Metrics
- Exploitability (OpenSpiel best-response metric)
- Average EV vs opponent families
- Posterior identification accuracy
- Novelty detection latency / false-positive rate
- Regret and robustness under drifting mixtures

---

## ⚙️ Implementation Order
1. **Kuhn Environment & Strategy Pool**
2. **PosteriorTracker (Bayesian)**
3. **KL-Regularized Selector**
4. **Novelty Detector**
5. **Evaluation & Plotting**
6. **Ridge Regression Opponent Model**
7. **LoRA / AutoStep online adaptation**
8. **Leduc integration**
9. **Longformer Opponent Model**
10. **Hold’em abstractions & full experiments**

Each stage will have tests & minimal examples before moving forward.

---

## 📚 References (to be expanded via literature review)
- Brown & Sandholm, *“Safe and Nested Subgame Solving for Imperfect-Information Games”* (AAAI 2017)
- Moravčík et al., *“DeepStack: Expert-Level AI in No-Limit Poker”* (Science 2017)
- Brown & Sandholm, *“Superhuman AI for multiplayer poker”* (Science 2019)
- Heinrich & Silver, *“Deep Reinforcement Learning from Self-Play in Imperfect-Information Games”* (AAAI 2016)
- Bowling et al., *“Heads-Up Limit Hold’em Poker is Solved”* (Science 2015)
- Lanctot et al., *OpenSpiel: A Framework for Reinforcement Learning in Games* (NeurIPS 2019)

---

## 🔬 Citation
TBD — will include paper link once published.

---

## 🧰 Environment setup
```bash
conda create -n stratformer python=3.11
conda activate stratformer
pip install -r requirements.txt
