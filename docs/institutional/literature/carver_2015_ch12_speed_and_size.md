# Carver 2015 — Chapter 12: Speed and Size (Systematic Trading)

**Source:** `resources/Robert Carver - Systematic Trading.pdf`
**Author:** Robert Carver (ex-AHL portfolio manager)
**Publication:** Harriman House, 2015 (ISBN 9780857194459)
**Extracted:** 2026-04-27
**Pages cited:** 177–203 (Chapter 12 "Speed and Size", PDF idx 194–220)

**Criticality for our project:** 🟡 **MEDIUM** — directly grounds the cost-aware backtest-vs-live-execution-fidelity principle that motivates the canonical scratch-EOD-MTM fix in `trading_app/outcome_builder.py`. Carver Ch 12 does NOT explicitly cover session-end forced flat (that is not the chapter's subject), but its framing of backtest cost realism is the institutional template our scratch handling must satisfy. For a literature-grounded treatment of the *selection-bias* angle on dropping scratch rows see `bailey_lopezdeprado_2014_dsr_sample_selection.md`.

---

## What this chapter actually covers

Chapter 12 ("Speed and Size") addresses two interrelated practical issues for any systematic system:
1. **Speed:** how quickly should you trade given the costs of doing so? — pp 178–185
2. **Size:** how do trading costs and frictions change as account size scales up or down? — pp 196–203

The chapter does NOT cover session-end forced exits or unfilled-order P&L attribution. It is cited here because Carver's general framing of backtest cost realism — that **every trade in a backtest must be priced as it would be priced in live execution** — is the same principle that requires forced-flat scratches to be priced (not silently dropped).

## Verbatim — backtest mid-price assumption is wrong (p 179)

> "Back-tests nearly always assume that when executing a trade you will pay the mid-price. But in practice the difference between the mid and the price you achieve will depend on how large your trades are compared to the available volume. This difference is the execution cost."

Carver names a class of silent backtest bias: backtests that assume an idealized exit price diverge from what the live system actually books. The same logic extends to **trades that are never closed at all in the backtest** — if the backtest assumes "no fill, no record," the live system that is forced flat at session end produces a P&L the backtest never priced.

## Verbatim — overconfidence as the source of overtrading (p 179)

> "Personally I think it is very unwise indeed [to assume the actual pre-cost SR will be 1.0 or higher]. If the actual pre-cost SR turned out to be less than 1.0, giving raw returns below 20% a year, then the strategy will lose money after costs. You'd be trading far too quickly relative to the available pre-cost performance.
>
> Overtrading is a result of overconfidence, one of the cognitive biases I discussed in chapter one. Only someone who was very bullish would assume the realised pre-cost SR would definitely be 1.0 or over in actual trading. You shouldn't trade systems like this and hope for the best. Instead it's much better to design trading systems that aren't vulnerable to such high levels of costs."

The transferable lesson: backtest performance must be *defended* against systematic upward biases. Silent dropout of unfilled trades is one such systematic upward bias — every dropped scratch is the absence of a (typically modest, near-zero, sometimes negative) realized P&L that would otherwise drag the average expectancy down.

## Verbatim — standardised cost in SR units (p 181–182)

> "The measure is defined as follows: if you buy one instrument block and then sell it, how much does that round trip cost when divided by the annualised risk of that instrument? This standardised cost is equivalent to how much of your annualised raw Sharpe ratio (SR) you'll lose in costs for each round trip."

Carver's framework is built on round-trip costs — entry plus exit, never just entry. A scratch trade in our system **is a round trip** (the position is opened, held until session end, then forced flat by the exchange/prop-firm rules). Booking only the entry-side cost while leaving the exit unpriced (NULL `pnl_r`) is therefore inconsistent with Carver's institutional cost-accounting frame.

---

## Application to canompx3

### Why this chapter is cited for the scratch-EOD-MTM fix

`trading_app/outcome_builder.py` currently produces `outcome="scratch"` rows with `pnl_r = NULL` when neither stop nor target is hit by trading-day-end. Downstream consumers using `WHERE pnl_r IS NOT NULL` silently drop these rows. The bias scales with target distance: 9.9% scratch rate at RR=1.0, 44.6% at RR=4.0 on MNQ NYSE_OPEN 15m.

Carver's framing (p 179) — "back-tests nearly always assume [an idealized exit price]" — is the parallel pathology. Our backtest assumes scratched trades have no exit at all, when the live execution path (TopStep prop-firm session-end flat rules, AMP/EdgeClear futures flat-by-close) **must close the position**. The live system books a realized P&L at the session-end close; the backtest books NULL. The two are not "the same program with different data" (Chan 2013 Ch 1 doctrine — see `chan_2013_ch1_backtesting_lookahead.md`).

### What Carver does NOT cover

Carver Ch 12 does NOT discuss:
- Forced-flat-by-session-end as a backtest realism issue
- Sample-selection bias from dropping unfilled trades
- The specific question of marking-to-market open positions at a daily boundary

Those are grounded by `bailey_lopezdeprado_2014_dsr_sample_selection.md` and `chan_2013_ch1_backtesting_lookahead.md` respectively. Carver Ch 12 is the cost-realism prior; the other two are the selection-bias and unified-program priors.

### Action items derived from Carver Ch 12

1. **Backtest must price every round trip.** A scratch trade is a round trip. `outcome_builder` must populate `pnl_r` for `outcome="scratch"` using realized session-end close. (Stage 5 of plan.)
2. **Round-trip costs are already in `pnl_points_to_r` via `pipeline.cost_model.COST_SPECS`.** The fix re-uses that canonical function — no new cost logic. Consistent with Carver's standardised-cost framework.
3. **Defend against systematic upward biases.** Add `pipeline/check_drift.py::check_orb_outcomes_scratch_pnl` to assert ≥99% of scratch rows have non-NULL `pnl_r` post-fix. (Stage 5b of plan.)

---

## Cross-references

- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` — Ch 9–10, position sizing (different scope).
- `docs/institutional/literature/carver_2015_ch11_portfolios.md` — Ch 11, portfolio construction (different scope).
- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` — unified backtest/live program doctrine (the strongest direct parallel).
- `docs/institutional/literature/bailey_lopezdeprado_2014_dsr_sample_selection.md` — the selection-bias frame.
- `docs/institutional/pre_registered_criteria.md` § Criterion 13 (added Stage 3) — the locked criterion citing this extract.
