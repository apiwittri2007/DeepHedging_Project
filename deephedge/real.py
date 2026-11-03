"""CLI: python -m deephedge.real --data data/hk --out results/real

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

from .real_experiment import run_real_experiment  # noqa: E402
from .realdata import build_windows, load_closes  # noqa: E402

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8f8e89"
COLOR = {"deep": "#2a78d6", "delta": "#eb6834", "banded_delta": "#1baf7a", "no_hedge": NEUTRAL}
LABEL = {"deep": "Deep hedger", "delta": "BS delta", "banded_delta": "Delta + band", "no_hedge": "No hedge"}
HEDGERS = ("deep", "delta", "banded_delta")


def _axes(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _fig(w=9, h=5, ncols=1, sharey=False):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), sharey=sharey)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _end_labels(ax, x, ends: dict):
    """Direct labels at line ends, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap, dx = 0.045 * (hi - lo), 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    last = None
    for name, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y_lab = y if last is None else max(y, last + gap)
        ax.text(x + dx, y_lab, LABEL[name], va="center", fontsize=8, color=INK2)
        last = y_lab


def chart_stylized(res, path):
    st = res["stylized"]
    x = st["std_returns"]
    x = x[np.abs(x) < 10]
    fig, ax = _fig()
    bins = np.linspace(-8, 8, 121)
    ax.hist(x, bins=bins, density=True, color=COLOR["deep"], alpha=0.85, label="HK tech daily returns", rwidth=0.9)
    g = np.linspace(-8, 8, 400)
    ax.plot(g, np.exp(-g**2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=2, ls="--", label="Normal (what GBM assumes)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("Daily return ÷ trader's trailing daily vol")
    ax.set_ylabel("Density (log scale)")
    _axes(ax, "Real Hong Kong returns have fat tails",
          f"Excess kurtosis {st['excess_kurtosis_std']:.1f} (normal = 0); "
          f"{st['share_beyond_4sd']*100:.2f}% of days beyond 4σ vs 0.006% under a normal")
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, path)


def chart_samples(res, path):
    fig, axes = _fig(11, 4.5, ncols=2, sharey=True)
    for ax, key, title, color in ((axes[0], "sim", "Simulated GBM (training world)", COLOR["delta"]),
                                  (axes[1], "real", "Real HK tech 1-month windows (test)", COLOR["deep"])):
        for p in res["sample"][key]:
            ax.plot(p, color=color, alpha=0.35, lw=1)
        ax.axhline(100, color=INK2, lw=1, ls=":")
        ax.set_xlabel("Trading day")
        _axes(ax, title)
    axes[0].set_ylabel("Price (start = 100, strike = 100)")
    _save(fig, path)


def chart_cvar_vs_cost(res, path):
    fig, axes = _fig(11, 4.8, ncols=2)
    for ax, ds, title in ((axes[0], "sim", "Simulated paths (same vol mix)"),
                          (axes[1], "real", "Real HK paths, 2016–2026")):
        rows = [r for r in res["rows"] if r["dataset"] == ds]
        costs = sorted({r["cost"] for r in rows})
        xb = [c * 1e4 for c in costs]
        ends = {}
        for s in HEDGERS:
            y = [next(r["cvar95"] for r in rows if r["cost"] == c and r["strategy"] == s) for c in costs]
            ax.plot(xb, y, "o-", color=COLOR[s], lw=2, ms=7, label=LABEL[s],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ends[s] = y[-1]
        _end_labels(ax, xb[-1], ends)
        nh = next(r["cvar95"] for r in rows if r["strategy"] == "no_hedge")
        ax.set_xlabel("Transaction cost (bps per side)")
        ax.set_ylabel("95% CVaR of P&L per 100 notional")
        ax.axvline(15, color=INK2, lw=1, ls=":")
        ax.text(15.5, ax.get_ylim()[0], " realistic HK\n stock cost", fontsize=8, color=INK2, va="bottom")
        _axes(ax, title, f"lower = less tail risk · no hedge = {nh:.1f} (off scale)")
        ax.set_xlim(right=max(c * 1e4 for c in costs) * 1.28)
    axes[0].legend(frameon=False, labelcolor=INK, loc="upper left")
    _save(fig, path)


def chart_pnl_hist(res, path):
    fig, ax = _fig()
    pnl = res["focus_pnl"]
    lo, hi = np.percentile(np.r_[pnl["deep"], pnl["delta"]], [0.5, 99.5])
    bins = np.linspace(lo, hi, 70)
    for s in ("delta", "deep"):
        ax.hist(pnl[s], bins=bins, histtype="step", lw=2, color=COLOR[s], label=LABEL[s], density=True)
    ax.axvline(0, color=INK2, lw=1, ls=":")
    ax.set_xlabel("Terminal hedging P&L per 100 notional (short 1-month ATM call)")
    ax.set_ylabel("Density")
    _axes(ax, f"Real-data P&L distribution at {res['focus_cost']*1e4:.0f} bps",
          f"{res['n']['test']} out-of-sample windows, 29 HS Tech stocks, 2016–2026")
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, path)


def chart_misspec(res, path):
    gap, pnl = res["misspec"]["gap"], res["misspec"]["pnl"]
    corr = np.corrcoef(gap, pnl)[0, 1]          # on all windows, before trimming for display
    keep = (np.abs(gap) < 1.0) & (np.abs(pnl) < np.percentile(np.abs(pnl), 99.5))
    gap, pnl = gap[keep], pnl[keep]
    fig, ax = _fig()
    ax.scatter(gap * 100, pnl, s=10, color=COLOR["delta"], alpha=0.35, linewidths=0, label="One window")
    edges = np.percentile(gap, np.linspace(0, 100, 13))
    mids = [(gap[(gap >= a) & (gap < b)].mean() * 100, pnl[(gap >= a) & (gap < b)].mean())
            for a, b in zip(edges[:-1], edges[1:]) if ((gap >= a) & (gap < b)).sum() > 5]
    ax.plot(*zip(*mids), color=INK, lw=2, marker="o", ms=6, label="Average by vol-gap bucket")
    ax.axhline(0, color=INK2, lw=1, ls=":")
    ax.axvline(0, color=INK2, lw=1, ls=":")
    ax.set_xlabel("Vol the trader used − vol that actually happened (percentage points)")
    ax.set_ylabel("Delta-hedge P&L, zero cost")
    _axes(ax, "Hedging P&L on real data is mostly a volatility bet",
          f"Correlation {corr:.2f}: sell options when vol is over-estimated → profit; under-estimated → loss")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    _save(fig, path)


def chart_periods(res, path):
    periods = [p for p in ("All test", "COVID crash", "Tech crackdown", "2023 onwards")
               if any(r["period"] == p and r["n"] > 0 for r in res["periods"])]
    fig, ax = _fig(10, 5)
    width = 0.26
    x = np.arange(len(periods))
    for i, s in enumerate(HEDGERS):
        vals = [next(r["cvar95"] for r in res["periods"] if r["period"] == p and r["strategy"] == s) for p in periods]
        bars = ax.bar(x + (i - 1) * (width + 0.02), vals, width, color=COLOR[s], label=LABEL[s],
                      edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ns = {p: next(r["n"] for r in res["periods"] if r["period"] == p) for p in periods}
    ax.set_xticks(x, [f"{p}\n(n={ns[p]})" for p in periods])
    ax.set_ylabel("95% CVaR of P&L (lower = better)")
    _axes(ax, f"Tail risk by market regime at {res['focus_cost']*1e4:.0f} bps",
          "Small-n periods (COVID) are noisy: CVaR₉₅ there rests on a handful of windows")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)   # headroom so the legend never sits on a bar
    ax.legend(frameon=False, labelcolor=INK, ncols=3, loc="upper center")
    _save(fig, path)


def chart_significance(res, path):
    sig = res["significance"]
    x = np.array([s["cost"] * 1e4 for s in sig])
    d = np.array([s["diff"] for s in sig])
    lo = np.array([s["ci_low"] for s in sig])
    hi = np.array([s["ci_high"] for s in sig])
    fig, ax = _fig()
    ax.axhline(0, color=INK2, lw=1)
    ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt="o", color=COLOR["deep"], ms=8, lw=2, capsize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="Deep − delta CVaR, 95% CI (month-clustered bootstrap)")
    for xi, di, s in zip(x, d, sig):
        p_txt = "p<0.001" if s["p_a_not_better"] < 0.001 else f"p={s['p_a_not_better']:.2f}"
        ax.annotate(p_txt, (xi, di), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Transaction cost (bps per side)")
    ax.set_ylabel("CVaR₉₅(deep) − CVaR₉₅(delta)")
    _axes(ax, "Is the deep hedger's edge on real data statistically real?",
          "Below 0 = deep hedger has less tail risk · interval entirely below 0 = significant at 5% · "
          "p = share of resamples where deep is not better")
    ax.legend(frameon=False, labelcolor=INK, loc="upper right")
    _save(fig, path)


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


CHARTS = {
    "real_fat_tails.png": chart_stylized,
    "real_sample_paths.png": chart_samples,
    "real_cvar_vs_cost.png": chart_cvar_vs_cost,
    "real_pnl_hist.png": chart_pnl_hist,
    "real_vol_misspecification.png": chart_misspec,
    "real_periods.png": chart_periods,
    "real_significance.png": chart_significance,
}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Deep hedging on real Hong Kong stock paths")
    p.add_argument("--data", default="data/hk")
    p.add_argument("--out", default="results/real")
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
    res = run_real_experiment(windows, **kw)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(write_tables(res, out))
    for name, fn in CHARTS.items():
        fn(res, out / name)
    return res


if __name__ == "__main__":
    main()
