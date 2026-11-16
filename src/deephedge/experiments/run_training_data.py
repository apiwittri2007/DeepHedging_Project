"""CLI: python -m deephedge.experiments.run_training_data --config configs/training_data.json

Runs the walk-forward training-data study on real Hong Kong stocks and writes
tables (CSV + Markdown), the raw results (``results.pkl``, so charts can be
redrawn with ``--from-results``) and the charts.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

from deephedge.data.realdata import load_closes
from deephedge.experiments.run_sim import load_overrides
from deephedge.experiments.training_data_study import run_training_data_study

NAMES = {
    "A0_gbm_true_vol": "A0 GBM, known vol", "A1_gbm": "A1 GBM, estimated vol",
    "A2_garch_t": "A2 GARCH-t", "A2b_garch_nu4": "A2b GARCH-t, ν ≥ 4 (sensitivity)", "A3_bootstrap": "A3 Block bootstrap", "A4_real": "A4 Real windows",
    "A5_finetune": "A5 GBM → real fine-tune", "L_linear": "L Linear correction", "HW_mv_delta": "HW Min-variance delta (Hull–White form)",
    "delta": "BS delta", "banded_delta": "Delta + band", "no_hedge": "No hedge",
}
QUICK = dict(years=(2016, 2017), cost_grid=(0.0, 0.0015), seeds=(0,), n_train=400, epochs=5,
             finetune_epochs=3, layer_sizes=(4, 8, 1), n_boot=100)


def _csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_pnl_csv(res, path: Path) -> None:
    """Per-window hedging P&L (one column per seed) in a portable format, for replication."""
    import gzip
    import numpy as np
    months, years = res["months"], res["test_years"]
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        n_seeds = max(v["pnl"].shape[0] for v in res["raw"].values())
        w.writerow(["arm", "cost", "window", "test_year", "month"] + [f"pnl_seed{i}" for i in range(n_seeds)])
        for key, v in sorted(res["raw"].items()):
            arm, cost = key.split("|")
            idx = np.flatnonzero(v["mask"])
            for j, i in enumerate(idx):
                w.writerow([arm, cost, int(i), int(years[i]), months[i]] + [f"{x:.5f}" for x in v["pnl"][:, j]])


def write_tables(res, out: Path) -> str:
    for key in ("summary", "significance", "by_year", "realism", "fold_sizes"):
        _csv(res[key], out / f"{key}.csv")
    if "raw" in res:
        write_pnl_csv(res, out / "pnl_by_window.csv.gz")
    fc = res["focus_cost"]
    lines = [f"## CVaR₉₅ of hedging P&L on the real test set (n = {res['n_test']}, mean ± sd over seeds)", "",
             "| Arm | " + " | ".join(f"{c*1e4:.0f} bps" for c in sorted({r['cost'] for r in res['summary']})) + " |",
             "|---|" + "---:|" * len({r['cost'] for r in res['summary']})]
    costs = sorted({r["cost"] for r in res["summary"]})
    for arm in NAMES:
        cells = []
        for c in costs:
            r = next((x for x in res["summary"] if x["arm"] == arm and x["cost"] == c), None)
            cells.append("—" if r is None else f"{r['cvar_mean']:.3f}" + (f" ± {r['cvar_sd']:.3f}" if r["n_seeds"] > 1 else ""))
        if any(c != "—" for c in cells):
            lines.append(f"| {NAMES[arm]} | " + " | ".join(cells) + " |")
    for vs in ("delta", "A1_gbm"):
        lines += ["", f"## CVaR difference vs {NAMES[vs]} (negative = less tail risk), month-clustered 95% CI", "",
                  "| Arm | Cost (bps) | Diff | 95% CI | p(not better) |", "|---|---:|---:|---|---:|"]
        for g in res["significance"]:
            if g["vs"] == vs:
                lines.append(f"| {NAMES[g['arm']]} | {g['cost']*1e4:.0f} | {g['diff']:+.3f} | "
                             f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] | {g['p']:.3f} |")
    lines += ["", "## Realism of training data (mean over folds)", "",
              "| Source | Excess kurtosis | ACF of abs returns (lags 1–10) | Hill tail index |", "|---|---:|---:|---:|"]
    for src in ("real", "A1_gbm", "A2_garch_t", "A2b_garch_nu4", "A3_bootstrap"):
        rows = [r for r in res["realism"] if r["source"] == src]
        if rows:
            m = lambda k: sum(r[k] for r in rows) / len(rows)
            lines.append(f"| {NAMES.get(src, 'Real (pre-year)')} | {m('excess_kurtosis'):.2f} | "
                         f"{m('acf_abs_1_10'):.3f} | {m('hill_index'):.2f} |")
    lines += ["", "## Folds", "", "| Test year | Test windows | Real training windows | Stocks | GARCH stocks | Unavailable |",
              "|---:|---:|---:|---:|---:|---|"]
    for f in res["fold_sizes"]:
        lines.append(f"| {f['year']} | {f['n_test']} | {f['n_real_train']} | {f['n_stocks_returns']} | "
                     f"{f['n_stocks_garch']} | {', '.join(f['unavailable']) or '—'} |")
    if "raw" in res:
        from deephedge.experiments.inference import economic_size, multiple_testing
        family = [k.split("|")[0] for k in res["raw"]]
        family = sorted({a for a in family if a not in ("delta", "no_hedge")})   # no-hedge is not a contender
        mt = res["multiple_testing"] = multiple_testing(res, vs="delta", arms=family)
        eco = res["economic"] = economic_size(res, vs="delta")
        _csv(mt, out / "multiple_testing.csv")
        _csv(eco, out / "economic_size.csv")
        lines += ["", f"## Multiple testing: every arm × cost vs BS delta as one family (m = {len(mt)})", "",
                  "Month-clustered bootstrap shared by all tests; one-sided (arm has lower CVaR).", "",
                  "| Arm | Cost (bps) | Diff | 95% CI | p raw | p Holm | p Romano–Wolf |",
                  "|---|---:|---:|---|---:|---:|---:|"]
        for g in mt:
            star = " **✓**" if g["p_romano_wolf"] < 0.05 else ""
            lines.append(f"| {NAMES.get(g['arm'], g['arm'])} | {g['cost']*1e4:.0f} | {g['diff']:+.3f} | "
                         f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] | {g['p_raw']:.3f} | {g['p_holm']:.3f} | "
                         f"{g['p_romano_wolf']:.3f}{star} |")
        prem = eco[0]["avg_premium_per_100"] if eco else float("nan")
        lines += ["", f"## Economic size at {fc*1e4:.0f} and 50 bps (vs BS delta; average premium HK${prem:.2f} per HK$100)", "",
                  "| Arm | Cost (bps) | CVaR change, HK$ per HK$1m notional | % of premium | Mean P&L change, HK$ per HK$1m |",
                  "|---|---:|---:|---:|---:|"]
        for e in eco:
            if e["cost"] in (fc, 0.005):
                lines.append(f"| {NAMES.get(e['arm'], e['arm'])} | {e['cost']*1e4:.0f} | {e['cvar_diff_hkd_per_1m']:+,.0f} | "
                             f"{e['cvar_diff_pct_premium']:+.1f}% | {e['mean_pnl_diff_hkd_per_1m']:+,.0f} |")
    lines.append(f"\nPrimary cost: {fc*1e4:.0f} bps.")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    return text


def merge_results(base: dict, new: dict) -> dict:
    """Add the rows of arms (and realism sources) that ``base`` does not have yet.

    Rows already in ``base`` are kept as they are, so re-running a subset of arms
    (e.g. A1 alongside a new sensitivity arm) never overwrites published numbers.
    """
    out = dict(base)
    for key, field in (("summary", "arm"), ("significance", "arm"), ("by_year", "arm"), ("realism", "source")):
        have = {r[field] for r in base[key]}
        out[key] = base[key] + [r for r in new[key] if r[field] not in have]
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Which training data makes a better deep hedger?")
    p.add_argument("--data", default="data/raw/hk")
    p.add_argument("--out", default="results/training_data")
    p.add_argument("--figures", default="figures/training_data")
    p.add_argument("--config", default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--from-results", action="store_true", help="redraw tables/charts from results.pkl")
    p.add_argument("--arms", default=None,
                   help="comma-separated arms to run; new arms are merged into an existing results.pkl")
    args = p.parse_args(argv)
    out, figs = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    if args.from_results:
        res = pickle.loads((out / "results.pkl").read_bytes())
    else:
        kw = load_overrides(args.config)
        if args.quick:
            kw.update(QUICK)
        if "years" in kw:
            kw["years"] = tuple(kw["years"])
        if args.arms:
            kw["arms"] = tuple(args.arms.split(","))
        t0 = time.time()
        res = run_training_data_study(load_closes(args.data), n_jobs=args.n_jobs,
                                      log=lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True), **kw)
        res["runtime_s"] = time.time() - t0
        old = out / "results.pkl"
        if args.arms and old.exists():
            res = merge_results(pickle.loads(old.read_bytes()), res)
        old.write_bytes(pickle.dumps(res))
    print(write_tables(res, out))
    (out / "results.pkl").write_bytes(pickle.dumps(res))      # now includes the joint-inference tables
    from deephedge.plots.training_data_charts import CHARTS
    for name, fn in CHARTS.items():
        fn(res, figs / name)
    return res


if __name__ == "__main__":
    main()
