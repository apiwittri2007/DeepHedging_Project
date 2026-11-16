# Real HK Data Out-of-Sample Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline, autopilot) + superpowers:test-driven-development.

**Goal:** Test the GBM-trained deep hedger against delta hedging on real Hang Seng Tech stock paths (2016–2026 out-of-sample), with charts and docs.

**Architecture:** Generalise `gbm`/`hedge`/`train` to per-path σ (scalar unchanged); add `realdata.py` (load, trailing vol, windows), `real_experiment.py` (calibrate → train → evaluate → breakdowns) and a `real` CLI subcommand writing 6 charts.

**Tech Stack:** NumPy, autograd, pandas (data only), matplotlib, pytest.

**Spec:** `docs/design/specs/2026-11-03-real-hk-data-design.md`

## Global Constraints

- Window: 22 prices (21 steps), T = 21/252, S₀ = K = 100; trailing vol window 63 returns, strictly before window start.
- Split date 2016-01-01; costs (0, 5, 10, 15, 25, 50) bps; focus 15 bps.
- Drop windows with NaN or any |daily log return| > 0.5.
- All existing 30 tests keep passing (scalar σ path unchanged).

## Review Focus

1. A stock with < 63 + 22 prices → yields zero windows, no crash.
2. A NaN inside a window → that window dropped, others kept.
3. Trailing vol must not include the window's own first return (look-ahead) — pinned by test.
4. Per-path σ array of wrong length → clear ValueError.
5. A calibration period with zero windows → clear ValueError, not a silent empty distribution.

---

### Task 1: Per-path σ in GBM, engine and training

**Files:** Modify `deephedge/gbm.py`, `deephedge/hedge.py`, `deephedge/train.py`; add `deephedge/delta.py:feature_delta_policy, feature_banded_policy`; tests `tests/test_per_path_sigma.py`.

**Interfaces:** `simulate_gbm_paths(..., sigma: float | np.ndarray (n_paths,), ...)`; `simulate_hedged_pnl(..., sigma: float | np.ndarray, premium: float | np.ndarray | None)` (premium default = BS price per path); `train_deep_hedger(..., sigma: float | np.ndarray)` slices σ with the batch indices; `feature_delta_policy()` returns `feats[:,3]`; `feature_banded_policy(band)`.

- [ ] Tests: `test_gbm_per_path_sigma_dispersion` (σ=[0.1]*N+[0.6]*N → terminal log-return std ≈ 0.1/0.6), `test_engine_sigma_array_equals_scalar_when_constant`, `test_wrong_length_sigma_raises`, `test_feature_delta_policy_matches_bs_delta`, `test_train_accepts_sigma_array`.
- [ ] Implement, run full suite (30 old + new) green, commit.

### Task 2: Real-data loading and windows

**Files:** Create `deephedge/realdata.py`, `scripts/fetch_hk_data.py`, `data/hk/*.csv`, `data/hk/NOTICE.md`; tests `tests/test_realdata.py`.

**Interfaces:** `load_closes(data_dir) -> dict[str, pd.Series]`; `build_windows(closes: dict, horizon=21, vol_lookback=63, max_abs_logret=0.5) -> Windows` dataclass with `paths (n,22)`, `trailing_vol (n,)`, `realised_vol (n,)`, `start (n,) datetime64`, `ticker (n,)`, `n_dropped`.

- [ ] Tests (synthetic series): `test_window_shape_and_normalisation`, `test_trailing_vol_excludes_window` (shock inside window doesn't change its trailing vol), `test_short_series_gives_no_windows`, `test_nan_window_dropped`, `test_extreme_return_window_dropped`, `test_non_overlapping`.
- [ ] Implement; generate `data/hk` from the cloned source with `python -I scripts/fetch_hk_data.py --src <clone>`; commit.

### Task 3: Real experiment

**Files:** Create `deephedge/real_experiment.py`; tests `tests/test_real_experiment.py`.

**Interfaces:** `run_real_experiment(windows, split="2016-01-01", cost_grid=(0,5e-4,1e-3,1.5e-3,2.5e-3,5e-3), focus_cost=1.5e-3, n_train=6000, epochs=200, seed=0) -> dict` with `rows` (cost, strategy, metrics; test windows), `periods` (period, strategy, cvar95, n at focus cost), `focus_pnl` (strategy→array), `calib_vols`, `stylized` (kurtosis etc.), `misspec` (vol gap, delta pnl). Raises ValueError if calibration has no windows.

- [ ] Tests on synthetic GBM-like "real" windows: `test_rows_structure`, `test_periods_present`, `test_empty_calibration_raises`, `test_deep_beats_no_hedge`.
- [ ] Implement, commit.

### Task 4: CLI, charts, run on real data

**Files:** Modify `deephedge/__main__.py` (add `real` subcommand: `python -m deephedge real --data data/hk --out results/real`); test `tests/test_real_cli.py` (quick mode on synthetic dir writes 6 PNGs + summary).

- [ ] Implement charts 1–6 per spec; run on real data; copy PNGs to `docs/img/real/`; commit.

### Task 5: Docs + CI

- [ ] README "Real Hong Kong data" section with real numbers + charts; `docs/PROJECT_SUMMARY.md`; update `docs/INTERVIEW.md`; `.github/workflows/ci.yml`; requirements add pandas.
- [ ] Full suite green; self-review; merge; push when repo exists.
