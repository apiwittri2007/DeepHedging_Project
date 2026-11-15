"""CLI: python -m deephedge.experiments.run_real --data data/raw/hk

Runs the real Hong Kong out-of-sample experiment and writes a results table
plus six charts. Charts follow one palette (validated for colour-blind safety);
the lower-contrast aqua series is always legend- and direct-labelled.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deephedge.experiments.real_study import run_real_experiment  # noqa: E402
from deephedge.plots.real_charts import CHARTS, LABEL  # noqa: E402
from deephedge.data.realdata import build_windows, load_closes  # noqa: E402
from deephedge.experiments.run_sim import load_overrides  # noqa: E402

def write_tables(res, out: Path) -> str:
    with (out / "real_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
        w.writeheader()
        w.writerows(res["rows"])
    lines = ["| Data | Cost (bps) | Strategy | Mean | Std | CVaR₉₅ | VaR₉₅ |", "|---|---:|---|---:|---:|---:|---:|"]
    for r in res["rows"]:
        lines.append(f"| {r['dataset']} | {r['cost']*1e4:.0f} | {LABEL[r['strategy']]} | {r['mean']:.3f} | "
                     f"{r['std']:.3f} | {r['cvar95']:.3f} | {r['var95']:.3f} |")
    lines += ["", "| Period | Strategy | n | CVaR₉₅ | Mean |", "|---|---|---:|---:|---:|"]
    for p in res["periods"]:
        lines.append(f"| {p['period']} | {LABEL[p['strategy']]} | {p['n']} | {p['cvar95']:.3f} | {p['mean']:.3f} |")
    lines += ["", "| Cost (bps) | CVaR deep − delta | 95% CI | p(deep not better) |", "|---:|---:|---|---:|"]
    for g in res["significance"]:
        lines.append(f"| {g['cost']*1e4:.0f} | {g['diff']:+.3f} | [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] | "
                     f"{g['p_a_not_better']:.3f} |")
    st, n = res["stylized"], res["n"]
    lines += ["", f"Windows: calibration {n['calib']}, test {n['test']}, dropped {n['dropped']}, stocks {n['stocks']}.",
              f"Excess kurtosis (raw / vol-standardised): {st['excess_kurtosis_raw']:.2f} / {st['excess_kurtosis_std']:.2f}."]
    text = "\n".join(lines) + "\n"
    (out / "real_summary.md").write_text(text)
    return text


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Deep hedging on real Hong Kong stock paths")
    p.add_argument("--data", default="data/raw/hk")
    p.add_argument("--out", default="results/real", help="where tables go")
    p.add_argument("--figures", default="figures/real", help="where charts go")
    p.add_argument("--config", default=None, help="JSON settings file, e.g. configs/real.json")
    p.add_argument("--split", default="2016-01-01")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    windows = build_windows(load_closes(args.data))
    kw = dict(split=args.split, epochs=args.epochs, n_train=args.n_train, seed=args.seed)
    if args.quick:
        kw.update(epochs=15, n_train=600, n_sim_eval=600, cost_grid=(0.0, 0.0015), layer_sizes=(4, 8, 1))
    kw.update(load_overrides(args.config))
    res = run_real_experiment(windows, **kw)

    out, figs = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    print(write_tables(res, out))
    for name, fn in CHARTS.items():
        fn(res, figs / name)
    return res


if __name__ == "__main__":
    main()
