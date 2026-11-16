"""Generate LaTeX tables and number macros for paper/main.tex from results/training_data/results.pkl.

Every number quoted in the paper comes from here, so the text cannot drift from the results.
Usage: python scripts/make_paper_assets.py
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "training_data" / "results.pkl"
OUT = ROOT / "paper" / "generated"

NAMES = {
    "A0_gbm_true_vol": "A0 GBM, known vol", "A1_gbm": "A1 GBM, estimated vol", "A2_garch_t": "A2 GARCH-$t$",
    "A2b_garch_nu4": "A2b GARCH-$t$, $\\nu\\ge4$$^\\dagger$", "A3_bootstrap": "A3 Block bootstrap",
    "A4_real": "A4 Real windows", "A5_finetune": "A5 GBM $\\to$ real", "L_linear": "L Linear correction",
    "HW_mv_delta": "HW Min.-variance delta", "delta": "BS delta", "banded_delta": "Delta + band",
    "no_hedge": "No hedge",
}
ORDER = list(NAMES)


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> None:
    res = pickle.loads(RES.read_bytes())
    OUT.mkdir(parents=True, exist_ok=True)
    costs = sorted({r["cost"] for r in res["summary"]})
    fc = res["focus_cost"]
    seeds = max(r["n_seeds"] for r in res["summary"])

    # Table 1: CVaR by arm and cost
    lines = ["\\begin{tabular}{l" + "r" * len(costs) + "}", "\\toprule",
             "Arm & " + " & ".join(f"{c*1e4:.0f} bps" for c in costs) + " \\\\", "\\midrule"]
    for arm in ORDER:
        cells = []
        for c in costs:
            r = next((x for x in res["summary"] if x["arm"] == arm and x["cost"] == c), None)
            cells.append("--" if r is None else f"{r['cvar_mean']:.2f}")
        if any(x != "--" for x in cells):
            lines.append(f"{NAMES[arm]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_cvar.tex").write_text("\n".join(lines) + "\n")

    # Table 2: multiple testing at the focus cost and 50 bps
    mt = res.get("multiple_testing", [])
    lines = ["\\begin{tabular}{lrrlrrr}", "\\toprule",
             "Arm & Cost & $\\Delta$CVaR & 95\\% CI & $p$ & $p_{\\text{Holm}}$ & $p_{\\text{RW}}$ \\\\", "\\midrule"]
    for g in mt:
        if g["cost"] in (fc, 0.005):
            lines.append(f"{NAMES.get(g['arm'], g['arm'])} & {g['cost']*1e4:.0f} & {g['diff']:+.3f} & "
                         f"[{g['ci_low']:+.2f}, {g['ci_high']:+.2f}] & {g['p_raw']:.3f} & {g['p_holm']:.3f} & "
                         f"{g['p_romano_wolf']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_tests.tex").write_text("\n".join(lines) + "\n")

    # Table 3: economic size
    eco = res.get("economic", [])
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Arm & Cost & $\\Delta$CVaR (HK\\$ per HK\\$1m) & \\% of premium & $\\Delta$mean P\\&L (HK\\$ per HK\\$1m) \\\\",
             "\\midrule"]
    for e in eco:
        if e["cost"] in (fc, 0.005) and e["arm"] not in ("no_hedge",):
            lines.append(f"{NAMES.get(e['arm'], e['arm'])} & {e['cost']*1e4:.0f} & {e['cvar_diff_hkd_per_1m']:+,.0f} & "
                         f"{e['cvar_diff_pct_premium']:+.1f} & {e['mean_pnl_diff_hkd_per_1m']:+,.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_economic.tex").write_text("\n".join(lines).replace(",", "{,}") + "\n")

    # Number macros used in the text
    def sig(arm, cost, vs="delta"):
        return next((g for g in mt if g["arm"] == arm and g["cost"] == cost and g["vs"] == vs), None)

    def letters(s):  # LaTeX macro names cannot contain digits or underscores
        return s.replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three") \
                .replace("4", "Four").replace("5", "Five").replace("_", "").replace("b", "B", 0)

    m = [macro("NSeeds", str(seeds)), macro("NTest", f"{res['n_test']:,}".replace(",", "{,}")),
         macro("FocusBps", f"{fc*1e4:.0f}"),
         macro("AvgPremium", f"{float(np.mean(res['premium'])):.2f}")]
    n_rw = sum(1 for g in mt if g["p_romano_wolf"] < 0.05)
    m.append(macro("NFamily", str(len(mt))))
    m.append(macro("NRejectRW", str(n_rw)))
    for arm in ("A0_gbm_true_vol", "A1_gbm", "A3_bootstrap", "A4_real", "A5_finetune", "L_linear", "HW_mv_delta",
                "A2b_garch_nu4"):
        for cost, tag in ((fc, "Focus"), (0.005, "Fifty")):
            g = sig(arm, cost)
            if g:
                base = letters(arm.split("_")[0]) + tag
                m.append(macro(f"Diff{base}", f"{g['diff']:+.2f}".replace("-", "$-$")))
                m.append(macro(f"CI{base}", f"[{g['ci_low']:+.2f}, {g['ci_high']:+.2f}]".replace("-", "$-$")))
                m.append(macro(f"PRW{base}", f"{g['p_romano_wolf']:.3f}"))
    (OUT / "numbers.tex").write_text("".join(m))
    for f in ("table_tests.tex", "table_economic.tex"):          # typographic minus signs in numbers
        t = (OUT / f).read_text()
        (OUT / f).write_text(re.sub(r"(?<=[\s\[])-(?=[\d{])", "$-$", t))
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
