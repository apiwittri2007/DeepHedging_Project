"""CLI: python -m deephedge.experiments.run_sim  (simulation study -> results/sim + figures/sim)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deephedge.experiments.sim_study import run_experiment  # noqa: E402
from deephedge.plots.sim_charts import (  # noqa: E402
    STYLE, plot_cvar_vs_cost, plot_example, plot_pnl_hist, plot_training,
)

def load_overrides(path: str | None) -> dict:
    """Keyword overrides from a JSON settings file (lists become tuples)."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}


def write_summary(rows, path: Path) -> None:
    hdr = "| Cost (bps) | Strategy | Mean | Std | CVaR₉₅ | VaR₉₅ |\n|---:|---|---:|---:|---:|---:|"
    lines = [hdr]
    for r in rows:
        lines.append(
            f"| {r['cost']*1e4:.0f} | {STYLE[r['strategy']][1]} | {r['mean']:.3f} | "
            f"{r['std']:.3f} | {r['cvar95']:.3f} | {r['var95']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_csv(rows, path: Path) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Deep hedging vs delta hedging experiment")
    p.add_argument("--out", default="results/sim", help="where tables go")
    p.add_argument("--figures", default="figures/sim", help="where charts go")
    p.add_argument("--config", default=None, help="JSON settings file, e.g. configs/sim.json")
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="small/fast run for smoke-testing")
    args = p.parse_args(argv)

    kw = dict(n_steps=args.n_steps, epochs=args.epochs, n_train=args.n_train,
              n_eval=args.n_eval, seed=args.seed)
    if args.quick:
        kw.update(n_steps=8, epochs=20, n_train=800, n_eval=800, cost_grid=(0.0, 0.002))
    kw.update(load_overrides(args.config))

    result = run_experiment(**kw)
    rows, extras = result["rows"], result["extras"]

    out, figs = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "summary.csv")
    write_summary(rows, out / "summary.md")
    plot_cvar_vs_cost(rows, figs / "cvar_vs_cost.png")
    if extras:
        plot_pnl_hist(extras["pnl"], extras["focus_cost"], figs / "pnl_hist.png")
        plot_training(extras["history"], figs / "training.png")
        plot_example(extras["example"], figs / "example_hedge.png")
    print((out / "summary.md").read_text())
    return result


if __name__ == "__main__":
    main()
