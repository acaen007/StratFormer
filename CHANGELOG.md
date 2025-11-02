# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-11-02
### Added
- T-001: Minimal OpenSpiel Kuhn wrapper (`stratformer.env.open_spiel_wrapper.KuhnEnv`) with deterministic `rollout(policy_a, policy_b, n_hands, seed)` returning `{"p0_avg_return", "p1_avg_return", "hands"}`.
- `Policy` protocol and `RandomPolicy` for smoke tests.
- `seed_all(seed:int)` in `stratformer.utils.seed` to seed `random`, `numpy`, and OpenSpiel when available.
- Tests `tests/test_kuhn_wrapper.py`: determinism with seed and zero-sum average; additional sanity tests.

### Changed
- Updated `pyproject.toml` to valid TOML and modern ruff config keys under `tool.ruff.lint`.
- Modernized typing across modules (built-in generics, `X | None`, `collections.abc`).
- Sorted imports and enforced style (ruff/isort).
- Test bootstrap `tests/conftest.py` to add repo root to `sys.path` for CI.
- Added `__init__.py` to all `stratformer/*` subpackages to ensure reliable imports in CI.

### Fixed
- `PosteriorTracker` dataclass now declares slot fields, fixing initialization under `slots=True`.
- `NoveltyDetector` trigger uses window average; aligned with test expectations.
- mypy warnings by removing unused `type: ignore`s and tightening annotations.
- `.gitignore` environment patterns anchored to top-level to avoid ignoring `stratformer/env/`.

## [0.1.0] - 2025-11-01
### Added
- Initial repository scaffolding for `stratformer` package.
- Public interfaces and stubs across env, pool, oppmod, counter, novelty, eval, features.
- Utilities for seeding, configs, and numerically stable math.
- Example Kuhn config and minimal experiment script.
- Tests asserting interfaces and numerical stability.
- CI (ruff, black, mypy, pytest) and optional pre-commit hooks.
- DESIGN.md documenting key interfaces and data flow.


