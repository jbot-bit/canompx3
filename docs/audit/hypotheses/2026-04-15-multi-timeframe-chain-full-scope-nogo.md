# NO-GO — Multi-Timeframe Indicator-on-Indicator Chaining (Full-Scope)

**Date:** 2026-04-15
**Session:** HTF
**Status:** NO-GO (structural, not empirical)

---

## Claim

Chaining 2-3 higher-timeframe indicators (e.g., "weekly trend direction × daily pivot × intraday VWAP alignment") as a stacked filter on ORB breakouts.

## Verdict

**NO-GO at full scope.** This is an arithmetic-structural verdict, not an empirical one — the configuration space exceeds our MinBTL bound by 25-100× before any data is touched.

## Arithmetic (why)

A modest 2-step multi-TF chain:
- 3 higher-timeframe features × 3 timeframes (D/W/M) × 2 directions × 2 confluence rules × 3 instruments × 6 sessions × 3 apertures × 3 RR targets = **5,832 trials**

Against our MinBTL bounds (`docs/institutional/pre_registered_criteria.md:90`):
- Strict Bailey E[max_N]=1.0 for MNQ/MES (6.65 clean yr): N ≤ 28 → **208× over**
- Relaxed E=1.2: N ≤ 120 → **49× over**
- Relaxed E=1.5: N ≤ 1,774 → **3.3× over**
- Operational ceiling: N ≤ 300 → **19× over**

Even at the most generous relaxation (E=1.5, "professional Sharpe"), the search exceeds the data horizon by 3×. The expected-max-Sharpe under null grows as `ln(N)^0.5`; at N=5,832, expected max IS Sharpe under zero true edge is ~3.6 annualized (Bailey 2013 Prop.1). Any observed IS Sharpe below 3.6 is indistinguishable from noise.

## Literature grounding

`docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md:84`:
> "A relatively simple strategy with just 7 binomial independent parameters offers N = 2⁷ = 128 trials, with an expected maximum Sharpe ratio above 2.6."

Our 2-step chain has more than 7 parameters. The paper explicitly rules out brute-force searches of this scale as inevitable overfit.

`docs/STRATEGY_BLUEPRINT.md:270`:
> "RSI, MACD, Bollinger, MA cross | GUILTY | RESEARCH_RULES.md: guilty until proven | Sensitivity + OOS + mechanism required"

Classical multi-TF indicator-on-indicator chains (which this proposal generalizes) live in the "guilty until proven" space. Burden of proof is on the proposer, not the null.

## Reopen criteria

This NO-GO can be reopened by a pre-registered hypothesis that:
1. Specifies a single, literature-grounded chain (cite passage from `docs/institutional/literature/` OR extract one from `resources/` PDFs before registering).
2. Constrains K ≤ 120 (Bailey relaxed E=1.2 at 6.65 yr MNQ/MES).
3. Declares the mechanism category (CONDITIONING / DIRECTIONAL / FILTER / STOP-TARGET).
4. Passes the same 14 binding gates from `2026-04-15-prior-day-zone-positional-features-orb.md`.
5. Shows a negative control (noise-chain) and positive control (known-validated chain).

No reopen via post-hoc mechanism explanation or retroactive literature citation.

## Relationship to Phase 1 study

`docs/audit/hypotheses/2026-04-15-prior-day-zone-positional-features-orb.md` is the bounded subset of the multi-TF question. Phase 1 tested 8 single-feature prior-day signals (K_local=96, well under bounds) and returned NO-GO on all 8. The full-scope chain is structurally larger and would fail harder.

## Committed as companion verdict

Per the parent pre-registration §14 point 7, this entry is written regardless of Phase 1 outcome to close the multi-TF-chain search path permanently at full scope.
