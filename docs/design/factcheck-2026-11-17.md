# Fact-check and cross-check: 17 November 2026

Three independent checks were run on the 10-seed results and on every document.

## 1. Independent recomputation
A reviewer who had not seen the code rebuilt all 2,650 test windows, the premiums and the delta-hedge P&L
from the raw price files. Their numbers matched the published ones to within 5e-6. Their CVaRs for all
12 arms × 6 costs matched `summary.md` to 3 decimals. Their own month-clustered bootstrap for L at 15 bps
gave p = 0.004 and CI [−0.237, −0.052] (published: p = 0.0035, CI [−0.229, −0.039]).

## 2. Code and statistics audit
- **Look-ahead:** none found.
- **Correct, checked against reference implementations:** the hedging engine and costs, CVaR,
  `weighted_cvar` (exact to 4e-15), Holm and Romano–Wolf.

**Major finding (fixed by disclosure):** the 15 bps significance of L depends on the bootstrap
cluster, because a 21-day window can span two months. Romano–Wolf p is 0.033 with 1-month clusters,
0.017 with 2-month clusters and 0.064 with 3-month clusters. All three are now reported; the result is
called borderline.

**Minor findings (fixed):**
- p-values could be exactly 0; they now use (1+#)/(B+1).
- "Studentised" was the wrong word; it now says "standardised".
- `economic_size` now compares common windows only.
- `merge_results` now carries the per-window P&L of new arms.
- The limitations now disclose total-return (dividend-adjusted) prices, the in-window data filter
  (one test window) and that inference is conditional on the trained seeds.

## 3. Citations and literature claims
All 22 references were verified on primary pages (arXiv, IDEAS/RePEc, publisher pages).

**Fixed:**
- Added missing DOIs and pages, and full author names.
- François et al. (2024) is now cited as published: *Finance Research Letters* 73 (2025) 106590.
- Two overstated claims were reworded. Ruf & Wang regress on delta, vega, vanna and gamma, not delta
  alone. Zernikov reports a delta reduction *consistent in direction* with Hull–White.
- The contrast with Mikkilä & Kanniainen (real-data training helped them) is now acknowledged.
- Six previously uncited entries are now cited.

**Confirmed:** Hong Kong stamp duty is 0.1% per side since 17 Nov 2023, and 0.13% from Aug 2021.

## 4. Own consistency check
- The job arithmetic was corrected: 3,960 jobs train 4,620 networks.
- The 50 bps range is now 0.30–0.37, and the ratio to L is now 1.7–2×.
- The paper abstract now says "the quadratic-in-delta correction" instead of "the linear corrections",
  because the Hull–White form is worse than delta.
- Only the known-vol GBM arm (A0), not "the GBM-trained hedger", stays significant at 50 bps.
