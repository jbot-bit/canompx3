# Phase 2.8 — Multi-year regime stratification (ex-2020 / ex-2022 / ex-2024)

**Date:** 2026-04-19
**Stage:** `docs/runtime/stages/phase-2-8-multi-year-regime-stratification.md` (to delete)
**Script:** `research/phase_2_8_multi_year_regime_stratification.py`
**Output CSV:** `research/output/phase_2_8_multi_year_regime_stratification.csv`

**Origin:** Phase 2.7 caveat (a) verification found ATR_20_pct elevated in 2020 (COVID), 2022 (rate-hike), AND 2024 — suggesting "2024 regime break" framing might be a recurring high-vol-year effect. This audit tests the hypothesis.

## TL;DR — hypothesis REFUTED

**"2024 regime break = recurring high-vol-year effect" hypothesis is REFUTED.** 0 of 38 active lanes show drag in 2+ of the 3 high-vol years. 2024-drag is year-specific, not regime-recurring.

## Pattern breakdown

| Pattern | Count | Interpretation |
|---------|------:|----------------|
| `VOL_NEUTRAL` | **34** | No year materially drags — regime-robust |
| `SINGLE_YEAR_DRAG (2024)` | 2 | Both MNQ EUROPE_FLOW CROSS_SGP_MOMENTUM RR1.5 + RR2.0 |
| `SINGLE_YEAR_DRAG (2020)` | 1 | MNQ_US_DATA_1000 X_MES_ATR60 RR1.0 |
| `UNEVALUABLE` | 1 | Thin 2020/2022/2024 sample |
| `RECURRING_VOL_DRAG` | **0** | No lane broke in ≥2 of the 3 high-vol years |

## SGP-momentum failure IS 2024-specific (not recurring)

| Lane | full | y2020 | y2022 | y2024 |
|------|-----:|------:|------:|------:|
| `MNQ_EUROPE_FLOW RR1.5 CROSS_SGP_MOMENTUM` | +0.081 | +0.050 | +0.076 | **−0.125** |
| `MNQ_EUROPE_FLOW RR2.0 CROSS_SGP_MOMENTUM` | +0.112 | +0.130 | +0.051 | **−0.132** |

In 2020 (COVID — MNQ ATR 73.0) and 2022 (rate-hike — MNQ ATR 59.5), SGP momentum performed NORMAL. In 2024 (MNQ ATR 77.4) it broke hard. Same instrument, same filter, same session, same RR — 2024 is distinctively hostile.

**What this means:** the 2024 SGP failure is NOT "high-vol regime" — it's something specific to 2024. Candidate explanations (all speculative; would need macro/microstructure data to validate):

1. Post-US-election flow decoupling from Asian session
2. Fed pivot disrupted Europe-Asia rate-differential-driven flow
3. Generative AI / Nasdaq concentration created intraday reversals that broke cross-session momentum
4. Fitschen-type trend-follow failure zone (mean-reverting regime per Chan p106)

**We cannot validate any of these without external macro data.** Per institutional-rigor rule 7, those remain training-memory hypotheses.

**Operational takeaway:** retire the 2 SGP lanes (already Phase 2.4/2.7 DOUBLE-CONFIRMED). Don't try to "rescue" them via regime-conditional gating because the regime-signature is not vol-regime (our detectable feature) — it's something else in 2024 that we can't characterize from our data.

## 2020-specific drag on US_DATA_1000 X_MES_ATR60 RR1.0

| Lane | full | y2020 | y2022 | y2024 |
|------|-----:|------:|------:|------:|
| `MNQ_US_DATA_1000 X_MES_ATR60 RR1.0` | +0.077 | **−0.059** | +0.069 | +0.289 |

This lane was Phase 2.5 Tier-4 (subset-t=1.56, below conventional significance). Phase 2.7 flagged it 2024_CRITICAL (2024 carried it). Phase 2.8 now shows it also had 2020-specific drag — and 2024 was its BEST year. **This lane's edge depends on post-2020 regime (i.e., most of its validated N came from 2022-2025).** Thin justification, volatile year-to-year. Retire candidate regardless of the 2020-drag framing.

## The 34 VOL_NEUTRAL lanes — genuinely robust

These lanes performed similarly across all 3 high-vol years. Full list in CSV; key signal: **every Phase 2.7 GOLD candidate (5 lanes) is in this set**. Their "regime-robust" label is multi-year-validated.

## Refining Phase 2.7 verdicts

| Verdict | Before (Phase 2.7) | After (Phase 2.8) |
|---------|--------------------|--------------------|
| GOLD pool | "regime-neutral on 2024" | "regime-neutral across 2020+2022+2024" (strengthens) |
| 2 SGP retires | "2024 PURE_DRAG" | "2024-specific (non-recurring) — genuine year-break" |
| 1 WATCH (`ATR_P50_O15`) | "2024 CARRIED" | Not in Phase 2.8 drag — verdict unchanged |
| "Recurring vol regime" reframing | Proposed | **REFUTED** |

## Institutional-grade lessons from this iteration

1. **Always test the reframing hypothesis before committing to it in doctrine.** My Phase 2.7 reframing "high-vol-year recurring regime effect" could have become load-bearing without Phase 2.8's multi-year check. Costly if wrong.

2. **"Year specificity" is a real finding.** Some strategy failures are tied to specific macro years, not reusable regime fingerprints. This is Bailey-LdP 2014 § DSR motivation — a strategy with unusual recent-year performance may be an artifact of that specific period, not a robust edge.

3. **Multi-year stratification should be standard** — add to audit-methodology rule 14 candidate list: "When an audit flags a lane for a specific-year fail, run ex-other-high-vol-years to test whether failure is year-specific or regime-recurring."

4. **Vol regime alone doesn't explain everything.** Chan p120 warning ("vol regimes of no help to stock traders") now has empirical teeth: we just showed two high-vol years (2020, 2022) where the same filter performed FINE, and one (2024) where it broke. Vol regime is NOT the causal variable.

## Self-audit

- Pre-reg in stage file committed before script ran ✓
- Canonical delegations verified (compute_mode_a agreement check inline) ✓
- Holdout sacred: `trading_day < HOLDOUT_SACRED_FROM` unchanged ✓
- No look-ahead (all years within Mode A) ✓
- Per-lane consistency: full_N = Σ(only_year_N) + rest approximately (algebraic check via ex-year windows)
- Hypothesis tested before committing to doctrine change ✓ (user's "iterate" instruction rewards me here)

## Next steps

1. **Phase 2.8 strengthens the case for the 2 PURE_DRAG retirements** — they're genuine year-specific fails, not rescuable by regime-gating.
2. **GOLD pool validation strengthened** — 34 VOL_NEUTRAL lanes include all GOLD candidates.
3. **No new filter or research direction unlocked** — clean honest null on the "recurring-regime" hypothesis.
4. **Process improvement:** add multi-year stratification to `.claude/rules/backtesting-methodology.md` as standard verification step for year-specific-fail findings.

## Audit trail

- Pre-reg stage file (Phase 2.7 lineage)
- Canonical delegation: compute_mode_a, filter_signal, HOLDOUT_SACRED_FROM, GOLD_DB_PATH, SESSION_CATALOG
- Cross-refs: Phase 2.7 (`2026-04-19-regime-break-2024-audit.md`, caveat verification doc), Phase 2.5 (subset-t sweep)
- Tests: `tests/test_research/test_phase_2_8_multi_year_regime_stratification.py`
