"""Charts for the simulation study (deep hedging vs delta on GBM paths)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402,F401

STYLE = {"deep": ("#2a78d6", "Deep hedger"), "delta": ("#eb6834", "BS delta"),
         "banded_delta": ("#1baf7a", "Delta + band"), "no_hedge": ("#8f8e89", "No hedge")}


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
    ax.plot(history, color="#2a78d6")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CVaR training loss")
    ax.set_title("Deep hedger training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(example, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(example["delta"], color="#eb6834", label="BS delta", linewidth=1.8)
    ax.plot(example["deep"], color="#2a78d6", label="Deep hedger", linewidth=1.8)
    ax.set_xlabel("Rebalancing step")
    ax.set_ylabel("Holding (shares)")
    ax.set_title("Example hedge path: deep hedger trades less than delta")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
