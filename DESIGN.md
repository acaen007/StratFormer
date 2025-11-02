## StratFormer Design Overview

### Goals
- Scale OpenSpiel-based poker research code from Kuhn → Leduc → Hold'em (abstracted).
- Stable, minimal interfaces to let contributors implement algorithms safely.
- Deterministic, numerically stable, and CI-enforced quality gates.

### Invariants
- Reproducibility via `stratformer.utils.seed.set_global_seed`.
- Numerics in log-space via `stratformer.utils.math_utils` (`logsumexp`, `stable_log_softmax`).
- All user-facing policies/models expose small typed interfaces.
- Artifacts and configs are versioned under `artifacts/` and `configs/`.

### Package Layout (public API)
- `stratformer.env.open_spiel_wrapper.OpenSpielEnv`: thin wrapper over OpenSpiel.
- `stratformer.pool.strategy_pool.StrategyPool`: stores named policies and metadata.
- `stratformer.oppmod.bayes_model.PosteriorTracker`: normalized posteriors over opponent strategies.
- `stratformer.counter.selector.Selector`: KL-regularized selector (stubbed).
- `stratformer.novelty.novelty_detector.NoveltyDetector`: triggers on sustained low max posterior.
- `stratformer.eval.{Evaluator,Tournament}`: evaluation and round-robin scaffolding.
- `stratformer.features.Featurizer`, `stratformer.obs.Observer`: observation/feature interfaces.
- `stratformer.utils.{math_utils,seed,config}`: utilities.

`stratformer.__init__` re-exports the public surface for ergonomics.

### Data Flow (typical experiment)
1. Load YAML with `ExperimentConfig`.
2. Set seeds.
3. Create `OpenSpielEnv` and `StrategyPool` with baseline policies.
4. During play, update `PosteriorTracker` with log-likelihood evidence.
5. Use `Selector` to pick a counter policy (KL-regularized objective; stubbed).
6. Use `NoveltyDetector` to detect unknown strategies; trigger discovery when needed.
7. Run `Evaluator`/`Tournament`, save metrics to `artifacts/<exp>/<timestamp>/`.

### Extensibility
- Policies: implement `Policy.act(observation: np.ndarray) -> int` and register in `StrategyPool`.
- Opponent models: extend `PosteriorTracker` or add new modules under `oppmod/`.
- Features/Observations: implement `Observer` and `Featurizer` for each game.
- Experiments: place scripts under `experiments/` and config under `configs/`.

### Testing & CI
- `pytest -q`, `ruff`, `black --check`, `mypy` enforced in GitHub Actions.
- Initial tests validate interfaces and numerical stability only.

### Implementation Notes
- Methods marked with `# IMPLEMENT` are intentional stubs.
- Keep functions small, typed (PEP 484), with NumPy-style docstrings.
- Use `logging` (INFO for runs, DEBUG internally). No prints in library code.


