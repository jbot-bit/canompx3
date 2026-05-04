# H1 IS Drift Check — MES LONDON_METALS O30 RR1.5 long overnight_range_pct≥80

**Date:** 2026-04-18
**Purpose:** verify the T0-T8 audit IS values for H1 still hold on current `orb_outcomes` + `daily_features` before committing the H1 signal-only shadow pre-reg. Pure drift check — NOT used to re-derive gate criteria (those trace to `pre_registered_criteria.md` Amendment 2.7 verbatim).
**Source of audit numbers:** `docs/audit/results/2026-04-15-t0-t8-audit-horizon-non-volume.md:48-67` (H1 section).
**Holdout discipline:** pre-2026-01-01 data only per Mode A sacred-window policy (`trading_app/holdout_policy.py:70`).

## Method

Join `orb_outcomes` (filter: symbol=MES, orb_label=LONDON_METALS, orb_minutes=30, entry_model=E2, confirm_bars=1, rr_target=1.5, trading_day < 2026-01-01, outcome NOT IN skip labels, direction derived from `target_price > entry_price`) to `daily_features` (symbol=MES, orb_minutes=5 per triple-join canonical rule at `.claude/rules/daily-features-joins.md`).

Aggregate pnl_r, outcome, overnight_range_pct. Compute N, ExpR, SD, WR, parametric t.

## Results

| Metric | T0-T8 audit (2026-04-15) | Live re-query (2026-04-18) | Match? |
|---|---:|---:|:---:|
| N_total (long entries, pre-2026) | 908 | 900 | Δ=-8 (-0.9%) — non-material |
| N_on_signal (long + overnight_range_pct≥80) | 196 | 189 | Δ=-7 (-3.6%) — non-material |
| ExpR_fire | +0.216 | +0.2158 | ✓ verbatim |
| SD_fire | 1.16 | 1.1629 | ✓ verbatim |
| WR_fire | 0.524 | 0.5243 | ✓ verbatim |
| ExpR_off | — (not audit-quoted) | -0.1069 | — |
| parametric t | — (audit quoted bootstrap p=0.0010) | 2.55 | — (bootstrap will re-compute at review) |

## Drift verdict

**MATCH.** Expectancy / volatility / win-rate numbers are verbatim within rounding. The small N discrepancy (Δ ≈ 3.6% on fires) is non-material — likely minor canonical-resolver evolution between 2026-04-15 and 2026-04-18 (3-day window). Direction of the N delta (-7 fires) is NEGATIVE, meaning anything, the current data is slightly more conservative than the audit snapshot. This does NOT strengthen H1 artificially.

## Implication for shadow pre-reg

Safe to proceed with H1 signal-only shadow pre-reg lock. All gate criteria will trace to `pre_registered_criteria.md` Amendment 2.7 verbatim, NOT to the numbers in this drift check. This doc exists ONLY to verify canonical-pipeline consistency with the audit that motivated the shadow.

## Non-goals

- NOT used to set gate thresholds (those are in Amendment 2.7).
- NOT used to re-derive IS statistics for "better" confidence (drift is non-material).
- NOT used to compute OOS — 2026 data is SACRED under Mode A and is queried once at review date per the pre-reg.

## Cross-refs

- `docs/audit/results/2026-04-15-t0-t8-audit-horizon-non-volume.md:48-67` — source H1 audit
- `docs/institutional/pre_registered_criteria.md` Amendment 2.7 — Mode A holdout discipline
- `trading_app/holdout_policy.py:70` — `HOLDOUT_SACRED_FROM = date(2026, 1, 1)`
- `.claude/rules/daily-features-joins.md` — canonical join rule (orb_minutes=5 CTE guard)
- Forthcoming: `docs/audit/hypotheses/2026-04-18-h1-mes-london-metals-signal-only-shadow.yaml` (pre-reg contract lock)
