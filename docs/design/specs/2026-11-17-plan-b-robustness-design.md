# Plan B (robustness and honesty pass): design

**Date:** 2026-11-17 · **Status:** approved by delegation. The user chose Plan B on 16 Nov and asked
for the recommended choices to be made overnight without waiting for answers.

## Intent
Make DeepHedging_Project credible both to journal referees and to recruiters:
- every page tells the walk-forward result;
- the statistics survive the questions referees will ask.

The research report "Publishing deep hedging and CV" (16 Nov 2026) motivates each item.

## Goals
1. **Consistency.** README, PROJECT_SUMMARY, INTERVIEW and the CV bullet all report the
   walk-forward multi-seed result. Part 2's single-split result is labelled as a superseded pilot.
2. **Seeds.** Increase from 3 to 10 for every arm and cost. Seeds 0–2 must reproduce the earlier run exactly.
3. **Hull–White benchmark.** Add the Hull & White (2017) minimum-variance delta form:
   `delta + n(d1) * (a + b*delta + c*delta^2)`.
   - Real option prices are unavailable, so its coefficients are fitted the same way as L: to the
     variance of zero-cost hedging P&L on pre-year real windows.
4. **Multiple testing.** Treat all arm × cost comparisons with BS delta as one family. Report raw,
   Holm and Romano–Wolf step-down p-values.
   - They come from one shared month-clustered bootstrap.
   - The bootstrap uses an exact weighted-CVaR identity, so no resampled data is materialised.
5. **Economic size.** Report CVaR and mean-P&L changes in HK$ per HK$1m notional and as a percentage of the average premium.
6. **Replication infrastructure:**
   - MIT LICENSE for the code (the data keeps GPL-3.0, see NOTICE);
   - CITATION.cff;
   - pinned `requirements-lock.txt`;
   - per-window P&L exported as CSV;
   - `docs/superpowers` renamed to `docs/design`.
7. **Paper.** A LaTeX draft (`paper/main.tex`) and an expanded, verification-tagged bibliography.

## Non-goals (tonight)
- **Real option prices.** The planned source, philippdubach/options-data, is gone: the repo and the
  files return 404, checked 16 Nov. The replacement is Deribit BTC options through the public API.
  It needs the user's browser, so it is left for the next session.
- Any change to the pre-registered arms or to the hyper-parameters.

## Anti-goals
- Tuning anything on 2016+ test data.
- Dropping or rewording an unfavourable result.

## Calls made by Claude (user may revisit)
- **Code licence:** MIT. **Author name** for LICENSE/CITATION: "Apiwit Triwatana" (inferred from the
  user's account name; to be confirmed).
- **Multiple-testing family:** every arm (including L, HW and the post-hoc A2b) × all 6 costs vs delta.
- **Romano–Wolf statistic:** studentised.
- **Run cost:** the 10-seed run takes about 3.2 h on 2 CPUs.
