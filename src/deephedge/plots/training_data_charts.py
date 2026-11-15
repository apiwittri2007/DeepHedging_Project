"""Charts for the training-data study (static PNGs for the paper).

Palette: the project's validated categorical slots; the diverging map is
blue (less tail risk than delta) <-> red (more), with a neutral gray midpoint.
Aqua is below 3:1 contrast, so every aqua mark carries a direct value label.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from deephedge.plots.real_charts import GRID, INK, INK2, NEUTRAL, SURFACE, _axes, _fig, _save  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"
DEEP = ("A0_gbm_true_vol", "A1_gbm", "A2_garch_t", "A3_bootstrap", "A4_real", "A5_finetune")
SHORT = {
    "A0_gbm_true_vol": "A0 GBM, known vol", "A1_gbm": "A1 GBM, est. vol", "A2_garch_t": "A2 GARCH-t",
    "A3_bootstrap": "A3 Block bootstrap", "A4_real": "A4 Real windows", "A5_finetune": "A5 GBM→real",
    "L_linear": "L Linear (no NN)", "delta": "BS delta", "banded_delta": "Delta + band", "no_hedge": "No hedge",
    "real": "Real data",
}
TICK = {"real": "Real", "A1_gbm": "A1 GBM", "A2_garch_t": "A2 GARCH", "A3_bootstrap": "A3 Bootstrap"}
DIVERGING = LinearSegmentedColormap.from_list("gap", [BLUE, "#f0efec", RED])


def _row(res, key, **match):
    return next((r for r in res[key] if all(r[k] == v for k, v in match.items())), None)


def _unstable(res):
    """Arms whose seeds disagree wildly (seed sd > 0.5 at the focus cost): drawn off-scale."""
    return {r["arm"] for r in res["summary"] if r["cost"] == res["focus_cost"] and r["cvar_sd"] > 0.5}


def _arms(res, with_linear=True):
    present = {r["arm"] for r in res["summary"]}
    arms = [a for a in DEEP if a in present]
    return arms + (["L_linear"] if with_linear and "L_linear" in present else [])


def chart_cvar_focus(res, path):
    fc = res["focus_cost"]
    names = _arms(res) + ["banded_delta", "delta"]
    rows = [_row(res, "summary", arm=a, cost=fc) for a in names]
    rows = [r for r in rows if r is not None]
    rows.sort(key=lambda r: r["cvar_mean"], reverse=True)
    fig, ax = _fig(9, 0.5 * len(rows) + 1.6)
    _axes(ax, f"Tail risk on real HK test windows at {fc*1e4:.0f} bps",
          f"CVaR₉₅ of hedging P&L (lower is better), n = {res['n_test']}; dot = mean of 3 seeds, line = ±1 sd")
    y = np.arange(len(rows))
    colors = [ORANGE if r["arm"] == "delta" else NEUTRAL if r["arm"] in ("banded_delta", "L_linear") else BLUE
              for r in rows]
    bad = _unstable(res)
    lo = min(r["cvar_mean"] for r in rows)
    hi = max(r["cvar_mean"] + r["cvar_sd"] for r in rows if r["arm"] not in bad) * 1.03
    # a dot plot, not bars: the axis does not start at zero, so bar length would mislead
    for yi, r, col in zip(y, rows, colors):
        if r["arm"] in bad:
            ax.annotate(f"off scale: {r['cvar_mean']:.2f} ± {r['cvar_sd']:.2f} (seeds unstable) →", (hi, yi),
                        xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
            continue
        ax.errorbar(r["cvar_mean"], yi, xerr=r["cvar_sd"], fmt="none", ecolor=col, elinewidth=2, capsize=0)
        ax.plot(r["cvar_mean"], yi, "o", ms=9, color=col, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{r['cvar_mean']:.2f}", (r["cvar_mean"] + r["cvar_sd"], yi), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [SHORT[r["arm"]] for r in rows])
    ax.set_xlim(lo - 0.1, hi + 0.05)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_xlabel("CVaR₉₅ (HK\\$ per HK\\$100 notional)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def chart_diff_vs_delta(res, path):
    fc = res["focus_cost"]
    rows = [_row(res, "significance", arm=a, cost=fc, vs="delta") for a in _arms(res)]
    rows = [r for r in rows if r is not None][::-1]
    fig, ax = _fig(9, 0.5 * len(rows) + 1.6)
    _axes(ax, f"Which training data beats BS delta? ({fc*1e4:.0f} bps)",
          "CVaR(arm) − CVaR(delta) with month-clustered 95% bootstrap CI; left of 0 = less tail risk")
    y = np.arange(len(rows))
    ax.axvline(0, color=INK2, linewidth=1)
    bad = _unstable(res)
    ok = [r for r in rows if r["arm"] not in bad]
    left, right = min(r["ci_low"] for r in ok), max(r["ci_high"] for r in ok)
    pad = 0.35 * (right - left)
    ax.set_xlim(left - 0.1 * (right - left), right + pad)
    for yi, r in zip(y, rows):
        if r["arm"] in bad:
            ax.annotate(f"off scale: {r['diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] →",
                        (right + pad, yi), xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8, color=INK2)
            continue
        sig = r["ci_high"] < 0
        c = BLUE if sig else NEUTRAL
        ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(r["diff"], yi, "o", ms=8, color=c, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{r['diff']:+.3f}", (r["ci_high"], yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [SHORT[r["arm"]] for r in rows])
    ax.set_xlabel("CVaR difference vs BS delta")
    ax.grid(axis="y", visible=False)
    ax.text(1, -0.12 - 0.02 * (8 - len(rows)), "blue = CI entirely below 0", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK2)
    _save(fig, path)


def chart_gap_by_cost(res, path):
    arms = _arms(res)
    costs = sorted({r["cost"] for r in res["summary"]})
    ncol = 4
    nrow = int(np.ceil(len(arms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.1 * nrow), sharex=True, squeeze=False)
    bad = _unstable(res)
    sig = [r for r in res["significance"] if r["vs"] == "delta" and r["arm"] in arms and r["arm"] not in bad]
    ylim = (min(r["ci_low"] for r in sig) * 1.1, max(r["ci_high"] for r in sig) * 1.1)
    fig.patch.set_facecolor(SURFACE)
    x = np.array(costs) * 1e4
    for ax, arm in zip(axes.ravel(), arms):
        rows = [_row(res, "significance", arm=arm, cost=c, vs="delta") for c in costs]
        d = np.array([r["diff"] for r in rows])
        lo = np.array([r["ci_low"] for r in rows])
        hi = np.array([r["ci_high"] for r in rows])
        _axes(ax, SHORT[arm] + ("  (own scale)" if arm in bad else ""))
        if arm not in bad:
            ax.set_ylim(*ylim)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.fill_between(x, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
        ax.plot(x, d, color=BLUE, linewidth=2, marker="o", ms=5, markeredgecolor=SURFACE)
    for ax in axes.ravel()[len(arms):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Cost per side (bps)")
    for ax in axes[:, 0]:
        ax.set_ylabel("CVaR − CVaR(delta)")
    fig.suptitle("CVaR gap to BS delta across transaction costs  (line = diff, band = 95% CI, below 0 = better)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    _save(fig, path)


def chart_gap_by_year(res, path):
    arms = _arms(res)
    years = sorted({r["year"] for r in res["by_year"]})
    delta = {r["year"]: r["cvar_mean"] for r in res["by_year"] if r["arm"] == "delta"}
    M = np.array([[(_row(res, "by_year", arm=a, year=y) or {"cvar_mean": np.nan})["cvar_mean"] - delta[y]
                   for y in years] for a in arms])
    bad = _unstable(res)
    keep = [i for i, a in enumerate(arms) if a not in bad]
    lim = np.nanmax(np.abs(M[keep])) if keep else np.nanmax(np.abs(M))   # outliers clip to full red
    fig, ax = _fig(11, 0.5 * len(arms) + 2)
    _axes(ax, f"Stability: CVaR gap to BS delta by test year ({res['focus_cost']*1e4:.0f} bps)",
          "blue = less tail risk than delta, red = more; numbers are CVaR differences")
    ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(arms)):
        for j in range(len(years)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.6 * lim else INK)
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(len(arms)), [SHORT[a] for a in arms])
    ax.grid(False)
    ax.set_xticks(np.arange(len(years) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(arms) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    _save(fig, path)


def chart_realism(res, path):
    sources = [s for s in ("real", "A1_gbm", "A2_garch_t", "A3_bootstrap")
               if any(r["source"] == s for r in res["realism"])]
    color = {"real": INK, "A1_gbm": BLUE, "A2_garch_t": ORANGE, "A3_bootstrap": AQUA}
    stats = [("excess_kurtosis", "Excess kurtosis of vol-scaled\nhedge-day returns (fat tails)"),
             ("acf_abs_1_10", "Autocorr. of |returns|, lags 1–10\n(volatility clustering)"),
             ("hill_index", "Hill tail index\n(lower = fatter tail)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.patch.set_facecolor(SURFACE)
    means = {}
    for ax, (k, title) in zip(axes, stats):
        _axes(ax, title)
        vals = [np.mean([r[k] for r in res["realism"] if r["source"] == s]) for s in sources]
        means[k] = dict(zip(sources, vals))
        ax.bar(range(len(sources)), vals, color=[color[s] for s in sources], width=0.65,
               edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(sources)), [TICK[s] for s in sources], fontsize=8)
        ax.grid(axis="x", visible=False)
    ax = axes[3]
    _axes(ax, f"Realism vs tail risk ({res['focus_cost']*1e4:.0f} bps)")
    arm_of = {"real": "A4_real", "A1_gbm": "A1_gbm", "A2_garch_t": "A2_garch_t", "A3_bootstrap": "A3_bootstrap"}
    for s in sources:
        r = _row(res, "summary", arm=arm_of[s], cost=res["focus_cost"])
        if r is None:
            continue
        x = means["hill_index"][s]
        ax.plot(x, r["cvar_mean"], "o", ms=9, color=color[s], markeredgecolor=SURFACE, markeredgewidth=2)
        off = {"real": (8, 6), "A3_bootstrap": (8, -12)}.get(s, (8, 6))
        ax.annotate(SHORT[arm_of[s]], (x, r["cvar_mean"]), xytext=off, textcoords="offset points",
                    ha="right" if off[0] < 0 else "left", fontsize=8, color=INK2)
    ax.set_xlabel("Hill tail index of training returns (lower = fatter)")
    ax.set_ylabel("CVaR₉₅ on real test set")
    ax.margins(0.25)
    fig.suptitle("How realistic is each training generator? (mean over walk-forward folds)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    _save(fig, path)


def chart_data_quantity(res, path):
    fs = res["fold_sizes"]
    rows = []
    for f in fs:
        a = _row(res, "by_year", arm="A4_real", year=f["year"])
        d = _row(res, "by_year", arm="delta", year=f["year"])
        if a and d:
            rows.append((f["n_real_train"], a["cvar_mean"] - d["cvar_mean"], f["year"]))
    fig, ax = _fig(8, 4.8)
    _axes(ax, "Does more real training data help? (A4 real windows)",
          f"Each dot = one test year; CVaR gap to BS delta at {res['focus_cost']*1e4:.0f} bps (below 0 = better)")
    ax.axhline(0, color=INK2, linewidth=1)
    for n, g, y in rows:
        ax.plot(n, g, "o", ms=8, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(str(y), (n, g), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Real training windows available before the test year")
    ax.set_ylabel("CVaR(A4) − CVaR(delta)")
    ax.margins(0.12)
    _save(fig, path)


CHARTS = {
    "td_cvar_focus.png": chart_cvar_focus,
    "td_diff_vs_delta.png": chart_diff_vs_delta,
    "td_gap_by_cost.png": chart_gap_by_cost,
    "td_gap_by_year.png": chart_gap_by_year,
    "td_realism.png": chart_realism,
    "td_data_quantity.png": chart_data_quantity,
}
