# Training-Data Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native, autopilot as instructed) + superpowers:test-driven-development.

**Goal:** Compare deep hedgers trained on six kinds of training data (and a linear benchmark) on one fixed real Hong Kong test set, walk-forward 2016–2026.

**Architecture:** Generators produce 84-day log-return paths (63 warm-up + 21 hedge) that go through the same trailing-vol window builder as real data; a walk-forward runner builds per-year inputs from pre-year data only, trains all arms in parallel jobs, pools out-of-sample P&L, and computes seed-averaged CVaR with month-clustered bootstrap tests.

**Tech Stack:** NumPy, SciPy (GARCH MLE), autograd, pandas, matplotlib, concurrent.futures.

**Spec:** `docs/design/specs/2026-11-15-training-data-study-design.md`

## Global Constraints
- Window: 63 warm-up returns + 21 hedge days; T = 21/252; S₀ = K = 100; trailing vol = std(ddof=1) of the 63 returns × √252.
- Test windows: non-overlapping, start in year Y; training data for fold Y: only prices dated before Y-01-01; real training windows must end before Y-01-01.
- Costs (0, 0.0005, 0.001, 0.0015, 0.0025, 0.005); focus 0.0015; seeds (0, 1, 2).
- Fixed: net (4, 32, 32, 1), lr 0.01, 200 epochs, batch 256, 6,000 synthetic paths; fine-tune 50 epochs at lr 0.002; bootstrap mean block 10; GARCH only for stocks with ≥ 500 pre-Y returns.
- Hygiene: drop returns with |log return| > 0.5 or NaN (same rule as real windows).
- All 54 existing tests keep passing.

## Review Focus
1. A fold where a stock has < 500 pre-Y returns → stock skipped for GARCH, others used; fold never crashes.
2. A fold where no stock qualifies for GARCH (or bootstrap/real has no data) → that arm is marked unavailable for the fold, not silently trained on nothing.
3. Leakage: every input to fold Y (returns, real windows, vol draws) is dated before Y-01-01 — pinned by test.
4. Synthetic trailing vol must be computed exactly as for real windows (same function) — pinned by test.
5. Seed-averaged bootstrap with a 2-D P&L array gives the same `diff` as averaging per-seed CVaRs.

---

### Task 1: Real-data helpers
**Files:** Modify `src/deephedge/data/realdata.py`; test `tests/test_realdata_training.py`.
**Interfaces (produces):** `build_windows(..., step: int | None = None, end_before: str | None = None)` (default step = horizon); `Windows.end: np.ndarray` (datetime64[D] of last price); `returns_before(closes, date: str, max_abs_logret=0.5) -> dict[str, np.ndarray]` (clean 1-D log returns, both prices dated < date).
- [ ] Tests: `test_step1_overlapping_count`, `test_end_before_filters_by_last_price`, `test_end_field_is_last_price_date`, `test_returns_before_excludes_date_and_errors`.
- [ ] Implement; full suite green; commit.

### Task 2: Generators
**Files:** Create `src/deephedge/sim/generators.py`; test `tests/test_generators.py`.
**Interfaces (produces):** `windows_from_returns(lr: (n, 84)) -> (paths (n, 22), trailing_vol (n,))`; `gbm_returns(sigmas, n_days, rng)`; `stationary_bootstrap(r, n, n_days, mean_block, rng)`; `GarchT` dataclass (mu, omega, alpha, beta, nu) + `fit_garch_t(r) -> GarchT` + `simulate_garch_t(model, n, n_days, rng, h0=None)`; `sample_mixture(per_stock: dict[str, X], n, draw: Callable[[X, int, rng], ndarray], rng)` weighting stocks by their number of returns.
- [ ] Tests: `test_windows_from_returns_matches_build_windows` (same series → same paths & trailing vol), `test_gbm_returns_vol`, `test_bootstrap_values_from_source_and_block_length` (mean run ≈ 10), `test_garch_fit_recovers_simulated_params` (loose), `test_garch_unconditional_variance`, `test_mixture_weights`.
- [ ] Implement (GARCH MLE: scipy L-BFGS-B on transformed params, variance-1 Student-t); commit.

### Task 3: Linear benchmark, fine-tune, multi-seed bootstrap
**Files:** Create `src/deephedge/models/linear.py`; modify `models/train.py` (`init: dict | None = None`), `hedging/metrics.py` (2-D `pnl_a` = seeds × windows); tests `tests/test_linear_finetune.py`.
**Interfaces:** `fit_linear_correction(paths, sigma, K=100, T=21/252) -> np.ndarray (3,)` minimising Var(P&L) of holding δ + c₀ + c₁δ + c₂δ² at zero cost (closed form); `linear_policy(coef)` using features.
- [ ] Tests: `test_linear_zero_coef_equals_delta`, `test_linear_fit_reduces_variance_in_sample`, `test_finetune_zero_epochs_returns_init`, `test_bootstrap_2d_diff_is_mean_of_seed_cvars`.
- [ ] Implement; commit.

### Task 4: Walk-forward study runner
**Files:** Create `src/deephedge/experiments/training_data_study.py`; test `tests/test_training_data_study.py`.
**Interfaces:** `ARMS = ("A0_gbm_true_vol", "A1_gbm", "A2_garch_t", "A3_bootstrap", "A4_real", "A5_finetune")`; `fold_inputs(closes, year) -> dict` (test windows, real train windows, pre-year returns); `run_training_data_study(closes, years, cost_grid, focus_cost, seeds, n_train, epochs, finetune_epochs, finetune_lr, mean_block, layer_sizes, n_jobs) -> dict` with `summary` rows (arm, cost, cvar_mean, cvar_sd, mean), `significance` rows (arm, cost, vs, diff, ci_low, ci_high, p), `by_year`, `realism`, `fold_sizes`.
- [ ] Tests (small synthetic closes, quick settings, n_jobs=1): `test_fold_inputs_no_leakage`, `test_runner_structure`, `test_unavailable_arm_reported`, `test_pooled_test_set_equals_non_overlapping_windows`.
- [ ] Implement (ProcessPoolExecutor over fold×arm×cost×seed jobs; deterministic per-job seeds); commit.

### Task 5: CLI, charts, run, docs, push
**Files:** Create `experiments/run_training_data.py`, `plots/training_data_charts.py`, `configs/training_data.json`; modify `scripts/run_all.sh`, `README.md`, `paper/outline.md`.
- [ ] CLI test (`--quick`, synthetic data) writes tables + charts; full run on `data/raw/hk`; look at every chart (dataviz rules); write results into README + paper outline; push; message the user.
