# Bailey & López de Prado (2014) — Sample-Selection Bias and the Scratch-Dropout Problem

**Source:** `resources/deflated-sharpe.pdf`
**Authors:** David H. Bailey, Marcos López de Prado
**Publication:** *Journal of Portfolio Management*, 2014
**Extracted:** 2026-04-27
**Pages cited:** 2–3, 9–10 (focused on the sample-selection-bias passages, not the full DSR formula)

**Criticality for our project:** 🔴 **HIGH** — directly grounds the institutional language of "sample-selection bias inflates measured performance" applied to the silent dropout of `outcome="scratch"` rows where `pnl_r IS NULL`. Companion file `bailey_lopez_de_prado_2014_deflated_sharpe.md` extracts the broader DSR formula and multiple-testing framing; this file zooms on the *selection-bias-from-incomplete-records* angle relevant to the canonical scratch-EOD-MTM fix.

For the broader DSR mathematical machinery (Equations 1–9, MinBTL bound, secretary-problem stopping rule) read the companion extract. This file extracts ONLY the sample-selection passages and maps them to scratch dropout.

---

## Verbatim — sample-selection bias as a class (page 2)

> "The problem of performance inflation extends beyond backtesting. More generally, researchers and investment managers tend to report only positive outcomes, a phenomenon known as *selection bias*. Not controlling for the number of trials involved in a particular discovery leads to over-optimistic performance expectations."

Bailey-LdP cleanly separate two inflation sources:
1. **Multiple testing bias** — searching N strategies and reporting the best (Eq 1, Eq 2 in companion extract).
2. **Selection bias** — reporting only the trials that yielded a clean positive outcome and silently dropping the rest.

The scratch-NULL bug is squarely in category (2): trades that did not cleanly resolve to a target/stop hit are silently excluded from every downstream aggregate. The selection criterion is "did the trade reach a tagged exit?", which correlates positively with target distance — at RR=4.0 the selection rate is 55.4% (44.6% scratch rate dropped); at RR=1.0 it is 90.1% (9.9% dropped).

## Verbatim — definition of selection bias inflation (pages 2–3)

> "The *Deflated Sharpe Ratio* (DSR) corrects for two leading sources of performance inflation: Selection bias under multiple testing and non-Normally distributed returns. In doing so, DSR helps separate legitimate empirical findings from statistical flukes."

The DSR machinery treats selection bias as a *first-class* source of inflation, on equal footing with non-Normality. This is the institutional precedent for treating silent scratch dropout as a first-class methodological defect, not a curiosity.

## Verbatim — the disclosure demand (page 3)

> "Put bluntly, *a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless, regardless of how excellent the reported performance might be*. Investors and journal referees should demand this information whenever a backtest is submitted to them, although even this will not remove the danger completely."

By analogy: a backtest that has not controlled for *which trades were included in the average* is similarly suspect. Every published lane ExpR in `live_config`, every fitness-verdict in `gold-db` MCP, every drift alarm in `sprt_monitor.py` has been computed without controlling for the selection effect of `WHERE pnl_r IS NOT NULL`. The 10–45% magnitude inflation observed empirically (`docs/audit/results/2026-04-27-mnq-unfiltered-high-rr-family-v1-CORRECTION.md`) is exactly the failure mode Bailey-LdP warn about.

## Verbatim — the worked example pattern (pages 9–10, paraphrased)

> "Suppose that a strategist is researching seasonality patterns in the treasury market... He uncovers that many combinations yield an annualized ŜR of 2, with a particular one yielding a ŜR of 2.5 over a daily sample of 5 years. Excited by this result, he calls an investor asking for funds to run this strategy, arguing that an annualized ŜR of 2.5 must be statistically significant. The investor [...] asks the strategist to disclose: i) The number of independent trials carried out (N); ii) the variance of the backtest results (V[ŜR_n]); iii) the sample length (T); and iv) the skewness and kurtosis of the returns (γ̂₃, γ̂₄)."

The institutional disclosure list is `(N, V, T, γ̂₃, γ̂₄)`. This file proposes that for ORB / intraday-futures backtests, the list must additionally include: **(v) the scratch-treatment policy** — `drop` / `include-as-zero` / `realized-eod`. Without this, the reported ExpR/Sharpe/PBO is incompletely specified.

---

## Application to canompx3

### Why selection bias by scratch-dropout is dangerous

The scratch-dropout selection rule is **NOT** ignorable noise. It correlates with the variable of interest (RR target distance) and with hidden lane characteristics (volatility, session compression, intraday momentum exhaustion). Specifically:

- **Higher RR → higher scratch rate.** Lanes that test farther targets have proportionally more dropped scratches. The reported ExpR is therefore conditional on `target hit OR stop hit` — a selection that systematically excludes the "neither hit" outcome which tends to be near-zero or modestly negative-after-cost.
- **Higher volatility → lower scratch rate at the same RR.** Volatile sessions reach targets/stops faster; the selection rule keeps a more representative sample. Quiet sessions have inflated scratch rates and therefore *more* missing data in the published expectancy.
- **Direction-asymmetric in trending regimes.** Bull-day trending markets resolve long entries to either target (early) or stop (also early); fewer scratches. Mean-reverting / chop regimes leave entries unresolved at session-end; those days are silently dropped.

Each of these channels can flip the published ranking of two lanes — Lane A reports better ExpR than Lane B not because Lane A is better, but because Lane A's selection rule kept a more favorable subsample.

### The fix is institutional-grounded

Populating `pnl_r` for scratches with realized session-end close P&L does not "add new data" — it simply reports the trades the backtest already simulated. The data was always there; only the recording was incomplete. This is consistent with Bailey-LdP's framing: the institutional defense against selection bias is **disclose what was tested**, not "select harder."

### Action items derived from Bailey-LdP 2014

1. **Add `scratch_policy:` field to every pre-reg.** (Stage 3 of plan, Criterion 13.) Disclosure is the institutional defense.
2. **Backtest must include all simulated trades.** Force populated `pnl_r` for scratches via `pnl_points_to_r(cost_spec, entry_price, stop_price, last_close - entry_price)`. (Stage 5.)
3. **Drift check enforces disclosure.** Any new research script using `WHERE pnl_r IS NOT NULL` must annotate `# scratch-policy: <drop|include-as-zero|realized-eod>`. (Stage 2.)
4. **Recompute Bailey-LdP DSR for affected lanes.** Stage 6 downstream re-verification must report whether DSR survives at the corrected ExpR baselines.

---

## Cross-references

- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` — full DSR formula extract.
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` — MinBTL bound (companion paper).
- `docs/institutional/literature/carver_2015_ch12_speed_and_size.md` — cost-realism parallel.
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` — unified backtest/live doctrine.
- `docs/institutional/pre_registered_criteria.md` § Criterion 13 (added Stage 3).
- `docs/audit/results/2026-04-27-mnq-unfiltered-high-rr-family-v1-CORRECTION.md` — empirical magnitude.
