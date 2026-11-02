"""CLI: python -m deephedge --out results/  (trains and writes results + plots)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiment import run_experiment  # noqa: E402

STYLE = {"deep": ("#d1495b", "Deep hedger"), "delta": ("#2f6db5", "BS delta"),
         "banded_delta": ("#3fa34d", "Delta + band"), "no_hedge": ("#8a8a8a", "No hedge")}


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


def plot_cvar_vs_cost(rows, path: Path) -> None:
    costs = sorted({r["cost"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat, (color, label) in STYLE.items():
        y = [next(r["cvar95"] for r in rows if r["cost"] == c and r["strategy"] == strat) for c in costs]
        ax.plot([c * 1e4 for c in costs], y, "o-", color=color, label=label, linewidth=1.8)
    ax.set_xlabel("Transaction cost (bps per side)")
    ax.set_ylabel("95% CVaR of hedging P&L (lower = better)")
    ax.set_title("Tail risk vs trading cost: deep hedging vs delta hedging")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pnl_hist(pnl, cost, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat in ("deep", "delta", "no_hedge"):
        color, label = STYLE[strat]
        ax.hist(pnl[strat], bins=80, alpha=0.5, color=color, label=label, density=True)
    ax.set_title(f"Hedging P&L distribution at {cost*1e4:.0f} bps cost")
    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training(history, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color="#d1495b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CVaR training loss")
    ax.set_title("Deep hedger training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(example, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(example["delta"], color="#2f6db5", label="BS delta", linewidth=1.8)
    ax.plot(example["deep"], color="#d1495b", label="Deep hedger", linewidth=1.8)
    ax.set_xlabel("Rebalancing step")
    ax.set_ylabel("Holding (shares)")
    ax.set_title("Example hedge path: deep hedger trades less than delta")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Deep hedging vs delta hedging experiment")
    p.add_argument("--out", default="results")
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

    result = run_experiment(**kw)
    rows, extras = result["rows"], result["extras"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "summary.csv")
    write_summary(rows, out / "summary.md")
    plot_cvar_vs_cost(rows, out / "cvar_vs_cost.png")
    if extras:
        plot_pnl_hist(extras["pnl"], extras["focus_cost"], out / "pnl_hist.png")
        plot_training(extras["history"], out / "training.png")
        plot_example(extras["example"], out / "example_hedge.png")
    print((out / "summary.md").read_text())
    return result


if __name__ == "__main__":
    main()
