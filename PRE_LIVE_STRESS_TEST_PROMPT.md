# Pre-Live Capital Stress Test Audit

You are conducting a **pre-live capital deployment audit** on an ORB breakout futures trading pipeline. This is the final gate before real money. Your job is adversarial — find every reason NOT to go live. Silence is approval. If something is wrong, say it plainly.

**Mode: READ-ONLY AUDIT. Do not edit any files. Do not fix anything. Report findings only.**

Read `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, and `docs/STRATEGY_BLUEPRINT.md` before starting. Use MCP tools (`gold-db`) for all database queries. Follow the volatile data rule — never cite stats from memory or docs; query canonical sources.

---

## PHASE 1: Data Integrity (bars_1m → daily_features)

Run each check. Report PASS/FAIL with evidence (row counts, sample output).

1. **Bar coverage gaps.** For each active instrument (MGC, MNQ, MES), query `bars_1m` for gaps > 2 calendar days in the last 2 years. List any gaps with dates. Cross-reference against known exchange holidays (CME holiday calendar). Unexplained gaps = FAIL.

2. **Bar staleness.** What is the most recent `ts_event` in `bars_1m` per instrument? Is it within the last 5 trading days? If not, the pipeline is stale.

3. **5m bar alignment.** Pick 3 random trading days per instrument. Verify `bars_5m` aggregates match manual SUM/MIN/MAX/FIRST/LAST of the underlying `bars_1m` rows. Any mismatch = aggregation bug.

4. **Daily features completeness.** For each instrument, count trading days in `daily_features` vs `bars_1m`. The daily_features count should be <= bars_1m distinct trading days (some days have no session). Flag any day present in bars_1m but missing from daily_features in the last 6 months.

5. **ORB computation spot check.** Pick 5 random trading days per instrument. For each, manually compute the 5-minute ORB (high/low of first 5 bars after session open from `SESSION_CATALOG`) from `bars_1m`. Compare to `daily_features.orb_high` / `orb_low`. Any discrepancy > 1 tick = FAIL.

6. **DST boundary verification.** For each US DST transition in 2024 and 2025 (Mar and Nov), verify that session times in `daily_features` shifted correctly. Check NYSE_OPEN and US_DATA_830 specifically — these cross midnight Brisbane time and are the most DST-sensitive. Compare actual bar timestamps around the session open vs expected times from `pipeline/dst.py SESSION_CATALOG`.

7. **Triple-join sanity.** Confirm `daily_features` has exactly 3 rows per (trading_day, symbol) — one per orb_minutes (5, 15, 30). Any day with != 3 rows = schema corruption.

8. **Source symbol mapping.** Verify that MGC data comes from GC, MES from ES (pre-Feb 2024) then MES native, MNQ is native. Check `source_symbol` column in `bars_1m` for any mismatches vs the mapping in `docs/ARCHITECTURE.md`.

---

## PHASE 2: Outcome Builder Integrity (orb_outcomes)

9. **Outcome row counts.** For each instrument × orb_minutes (5, 15, 30), count rows in `orb_outcomes`. Compare to `daily_features` row count for the same slice. Outcomes should exist for every day that has an ORB break. Flag significant discrepancies (>5%).

10. **Cost model verification.** Read `pipeline/cost_model.py` COST_SPECS. For each instrument, verify:
    - Point values match CME contract specs (MGC=$10, MNQ=$2, MES=$5)
    - Commission matches current broker rates
    - Total friction is correctly computed (commission + spread_doubled + slippage)
    - Run `python -c "from pipeline.cost_model import COST_SPECS; [print(f'{k}: ${v.total_friction:.2f} RT, {v.friction_in_points:.2f} pts') for k,v in COST_SPECS.items()]"`

11. **R-multiple computation audit.** Pick 10 random trades from `orb_outcomes` (mix of winners and losers across instruments). For each, manually verify:
    - `risk_points = |entry_price - stop_price|`
    - `pnl_r = (raw_pnl_points - friction_points) / risk_points`
    - Friction points = `total_friction / point_value`
    - Entry price logic matches the entry model (E1: next bar open after confirmation; E2: ORB boundary + slippage ticks)
    - Stop price = opposite side of ORB

12. **E1 vs E2 entry price sanity.** Query 50 E1 and 50 E2 trades. Verify:
    - E1 entries are OUTSIDE the ORB range (overshoot expected ~15-22%)
    - E2 entries are at ORB boundary ± slippage (within E2_SLIPPAGE_TICKS)
    - No E0 entries exist anywhere (E0 was purged Feb 2026)

13. **Winner/loser outcome logic.** For 20 random winning trades: verify target was actually hit by checking `bars_1m` price action between entry time and exit time. For 20 random losing trades: verify stop was hit. Any trade where neither target nor stop was hit but an outcome was recorded = FAIL.

14. **Fakeout inclusion (E2).** E2 must include fakeout fills (bar touches ORB level but closes back inside). Query E2 trades where the break bar closed back inside the ORB. These MUST exist — their absence means fakeout exclusion bias (the E0 bug).

15. **Time-in-force / session boundary.** Verify no trade has exit_time after the session's natural end. Check for trades spanning midnight UTC — these must be handled correctly with Brisbane timezone logic.

---

## PHASE 3: Validation & Statistical Integrity

16. **BH FDR verification.** Read `trading_app/strategy_validator.py`. For the most recent validation run:
    - What K was used? (Must match honest test count, not cherry-picked)
    - Were both global K and instrument-level K reported?
    - Pick 5 validated strategies — recompute their p-values from `orb_outcomes` using a two-tailed t-test on pnl_r. Do they match what's stored?

17. **Walk-forward efficiency.** For 10 validated strategies, check WFE (out-of-sample Sharpe / in-sample Sharpe). Any WFE < 0.50 that was promoted = suspicious. List them.

18. **Sample size compliance.** Query `validated_setups` for any strategy with N < 30 classified as anything other than INVALID. Query for N < 100 classified as CORE. Both = classification bug.

19. **2026 holdout integrity.** Verify that ONLY the 3 pre-registered strategies (from `docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md`) have been tested against 2026 data. Query `orb_outcomes` for 2026 dates — these should exist (pipeline runs). But check `validated_setups` and `experimental_strategies` — any strategy validated using 2026 data that is NOT one of the 3 pre-registered = holdout contamination. **This is the most important statistical check in the audit.**

20. **Dead instrument contamination.** Verify M2K, MCL, SIL, M6E, MBT do NOT appear in `validated_setups`, `edge_families`, or `live_config`. Their data can exist in `bars_1m`/`orb_outcomes` for historical research, but they must not leak into anything downstream of validation.

21. **Year-by-year stability.** For each strategy in the live portfolio (from `trading_app/live_config.py` LIVE_PORTFOLIO), query yearly ExpR from `orb_outcomes`. Flag any strategy where >40% of years are negative. A strategy that's positive on average but negative in 3/5 years is a regime artifact, not an edge.

---

## PHASE 4: Execution Engine & Paper Trader Parity

22. **Backtest-live parity.** Run paper_trader on a 6-month historical window for 3 live portfolio strategies. Compare:
    - Trade count (paper_trader vs orb_outcomes for same period/filter)
    - Average PnL (should match within 5%)
    - Win rate (should match within 3%)
    - Any material discrepancy = execution engine diverges from outcome_builder = you're trading something different than what you backtested.

23. **Filter application in execution.** Read `trading_app/execution_engine.py`. Trace the filter application path. Verify:
    - `filter_type` strings match EXACTLY what `trading_app/config.py ALL_FILTERS` defines
    - Unknown filter strings are rejected (fail-closed), not silently passed
    - G4/G5/G6/G8 thresholds match what's in daily_features computation

24. **Early exit implementation.** Verify `EARLY_EXIT_MINUTES` in `config.py` is correctly wired in execution_engine. Check that T80 kill times are applied to CME_REOPEN and TOKYO_OPEN. Verify the early exit fires on losing trades only (not winners).

25. **IB-conditional exit is DISABLED.** Verify `SESSION_EXIT_MODE["TOKYO_OPEN"] = "fixed_target"` in config. The IB logic was never validated in outcome_builder — if it's enabled, there's a backtest-live parity gap.

26. **Direction filter enforcement.** TOKYO_OPEN is LONG-ONLY. Verify the execution engine actually filters short signals at TOKYO_OPEN. Check that no TOKYO_OPEN short trades exist in recent paper_trader output.

27. **Calendar overlay wiring.** If NFP/OPEX/DOW calendar overlays are in `config.py`, verify they're applied correctly per-instrument-per-session (not as blanket filters). Cross-reference with the calendar effects table in TRADING_RULES.md. A blanket NFP skip would HURT strategies where NFP days are BETTER (MNQ NYSE_OPEN, MES US_DATA_1000).

---

## PHASE 5: Cost Model Stress Test

28. **Slippage sensitivity.** The MNQ slippage model is 1 tick ($0.50). The MGC TBBO pilot showed mean=6.75 ticks. For each live portfolio strategy:
    - Compute ExpR at 1x modeled slippage (current)
    - Compute ExpR at 2x modeled slippage
    - Compute ExpR at 3x modeled slippage
    - Which strategies go negative at 2x? At 3x? Those are FRAGILE.
    - Cross-reference with the break-even analysis: COMEX_SETTLE breaks even at +4.9 extra ticks (FRAGILE), NYSE_OPEN at +17.7 (ROBUST).

29. **Friction as % of risk.** For each live strategy, compute median `total_friction / (risk_points × point_value)`. The ORB-size-is-the-edge finding says trades where friction > ~15% of risk lose money. Any live strategy with median friction > 12% of risk = red flag.

30. **Commission rate verification.** Check current NinjaTrader/broker commission schedule against what's in COST_SPECS. If broker raised rates since the model was built, all backtests are optimistic.

---

## PHASE 6: Portfolio Construction & Risk

31. **Live portfolio audit.** Read `trading_app/live_config.py`. For each `LiveStrategySpec`:
    - Is the family_id actually present and FIT/WATCH in `edge_families`?
    - Does the tier (core/regime) match the sample size rules (CORE >= 100, REGIME 30-99)?
    - Are `exclude_instruments` correctly set per BH FDR results?
    - Is `rr_target` locked and matching the validated strategy's RR?

32. **Correlation matrix.** Compute daily PnL correlation between all live portfolio strategies using `orb_outcomes`. Flag any pair with correlation > 0.60 — they're the same trade. Specifically check MNQ vs MES at the same session (expected ~0.83, which means DON'T stack both).

33. **Max drawdown simulation.** For each live strategy, compute:
    - Historical max drawdown in R
    - Max consecutive losing days
    - Worst single month
    - Is this survivable at the planned position size? (What IS the planned position size?)

34. **Prop firm compatibility (if applicable).** If trading under prop rules:
    - Does the max DD fit within the prop firm's daily/trailing DD limit?
    - Is the risk per trade sized so that 5 consecutive max losses don't breach the DD limit?
    - Is the stop multiplier (0.75x for prop) applied?

35. **Simultaneous position risk.** Can multiple strategies trigger at overlapping times? (e.g., TOKYO_OPEN and SINGAPORE_OPEN are 1 hour apart). What's the max concurrent exposure if all trigger on the same day?

---

## PHASE 7: Infrastructure & Guardrails

36. **Drift check clean.** Run `python pipeline/check_drift.py`. ALL checks must pass. Report the total check count and any failures.

37. **Test suite clean.** Run `python -m pytest tests/ -x -q`. ALL tests must pass. Report count and any failures.

38. **Health check clean.** Run `python pipeline/health_check.py`. Report results.

39. **Pre-commit hook active.** Verify `git config core.hooksPath` returns `.githooks`. Run a test: stage a trivially broken file and verify the hook catches it.

40. **Database concurrency guard.** DuckDB does not support concurrent writers. Verify there is NO cron job, scheduler, or background process that writes to `gold.db` while trading is active. Check for any `pipeline_status.py --rebuild` in crontab.

41. **Stale derived layers.** Check timestamps on `validated_setups`, `edge_families`. If either was built > 7 days ago and the pipeline has ingested new data since, the derived layers are stale. Flag it.

42. **Config.py consistency.** Verify `ALL_FILTERS` in config.py matches exactly what `build_daily_features.py` computes. Any filter string in the live portfolio that doesn't exist in ALL_FILTERS = silent trade drops (trades just disappear with no error).

---

## PHASE 8: Adversarial Checks (Red Team)

43. **Survivorship bias scan.** The project killed M2K, MCL, SIL, M6E, MBT. Were these instruments' failures considered when computing portfolio-level stats? Or do the "portfolio" numbers only include survivors? If only survivors, the portfolio Sharpe is biased upward.

44. **Regime dependency.** Gold has been trending hard 2024-2026. What happens to MGC strategies in a mean-reverting gold regime? Check 2019-2021 (COVID chop) separately. If MGC strategies were negative pre-2022, the edge is regime-dependent, not structural.

45. **MNQ unfiltered positive — real or artifact?** MNQ E2 is the only instrument positive without a size filter. This is unusual. Stress test: compute MNQ E2 NO_FILTER ExpR for each individual year. If only 2-3 years are positive and the rest are zero/negative, it's a regime artifact despite passing BH FDR.

46. **E3 retrace fill rate.** E3 uses limit orders. What percentage of E3 signals actually get filled? (break happens but price never retraces to ORB boundary). If fill rate is < 40%, the sample is severely selection-biased — only "nice" retraces get in, ugly ones don't. Report actual fill rate from orb_outcomes.

47. **Look-ahead contamination scan.** Grep the codebase for any use of `double_break` as a pre-entry filter (it's look-ahead — computed over the full session). Check `build_daily_features.py` and `outcome_builder.py`. If double_break is used to EXCLUDE days before computing outcomes, every outcome stat is contaminated.

48. **Data leakage in walk-forward.** Read the walk-forward implementation in `strategy_validator.py`. Verify:
    - The expanding/rolling window NEVER includes OOS data in the IS training period
    - Parameters are NOT re-optimized on OOS data
    - The OOS window advances strictly forward in time

49. **Worst-case scenario modeling.** Assume the REAL slippage is 3x modeled, commissions increase 50%, and the next 12 months are a mean-reverting regime. How many live portfolio strategies survive? If < 50% survive this stress scenario, the portfolio is fragile.

50. **The "would you bet your own money" test.** After completing all checks above, give an honest overall assessment:
    - GREEN: Go live. Evidence supports deployment.
    - YELLOW: Go live with reduced size. Specific concerns listed.
    - AMBER: Paper trade longer. Material issues found.
    - RED: Do not go live. Critical issues found.

    State your assessment and the TOP 3 reasons for it. No hedging. No "on the other hand." A clear call.

---

## Output Format

For each numbered check, report:

```
## Check N: [Name]
**Status:** PASS | FAIL | WARNING | SKIPPED (with reason)
**Evidence:** [Exact query, row counts, or command output]
**Finding:** [One sentence if PASS, detailed explanation if FAIL/WARNING]
```

At the end, produce:

```
## AUDIT SUMMARY
- Total checks: 50
- PASS: X
- FAIL: X
- WARNING: X
- SKIPPED: X

## CRITICAL FAILURES (must fix before live)
[list]

## WARNINGS (accept risk or fix)
[list]

## OVERALL ASSESSMENT: [GREEN/YELLOW/AMBER/RED]
[Top 3 reasons]
```

**Remember: your job is to find problems, not to reassure. Every PASS must have evidence. Every FAIL must have a specific finding. "Looks fine" is not evidence. Run the query, show the output.**
