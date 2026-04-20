# Phase 1 — Enumeration of memory-cited pooled / universal claims

**Date:** 2026-04-20
**Author:** retroactive heterogeneity audit (CURRENT-C from `next_session_mandates_2026_04_20.md`)
**Purpose:** list every memory-cited finding that frames a result as
"universal", "pooled", "K_global", "across all X", or "cross-instrument X/X positive"
— the framings that RULE 14 targets for per-cell decomposition.

## Enumeration methodology

- Grep memory + `docs/audit/results/` for the trigger keywords
- Cite topic file / result doc for each
- Classify status: still-claimed-live / already-decomposed / archived
- Do NOT yet run breakdowns — that is Phase 3 after user triage

## Claims found (sorted by current-status priority)

### TIER A — queued for activation OR cited as live research state

| # | Claim | Cited scope | Source | Already decomposed? |
|---|---|---|---|---|
| A1 | Exchange range signal (pit range / ATR T1-T8) — "3/3 inst, +17% WR, K=320 BH, 75-94% years, T8 cross-inst 3/3 positive" | 3 instruments × sessions pooled | `memory/exchange_range_signal.md`; `scripts/research/exchange_range_t2t8.py` | **NO** — per-cell breakdown not on record |
| A2 | H2 Path C — "garch_vol_pct≥70 universal but no synergy with rel_vol" | All sessions × instruments | MEMORY.md line 45; `docs/audit/results/2026-04-15-path-c-h2-closure.md` | **PARTIAL** — `2026-04-15-garch-all-sessions-universality.md` has per-(inst,session) table, no sign-heterogeneity tally |
| A3 | "Volume confirmation universal" — 14,261 cells, 13 K_global BH-FDR survivors framed as 5 `rel_vol_HIGH_Q3` + 8 others | 12 sessions × 3 inst × 3 apt × 3 RR | `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` | **PARTIAL** — cross-instrument overlap decomposed (MES×MNQ COMEX Jaccard 0.491 on rel_vol_HIGH_Q3); NOT sign-heterogeneity decomposed on the other 8 |
| A4 | HORIZON T0-T8 H2 "VALIDATED" — MNQ COMEX_SETTLE O5 RR1.0 garch_forecast_vol_pct≥70 | **SINGLE CELL** | MEMORY.md line 47 | N/A — not a pooled claim |
| A5 | Cross-session momentum — "SGP RR1.5 genuinely better per-trade than EUROPE_FLOW L1" | 2 lanes compared | `memory/cross_session_momentum_research.md` | N/A — pairwise, not pooled |

### TIER B — historical finding, still used as doctrine or NO-GO basis

| # | Claim | Cited scope | Source | Already decomposed? |
|---|---|---|---|---|
| B1 | G-filter redundancy — "G4/G5/G6/G8 add 0.0% WR across ALL sessions and instruments" | 3 inst × 11 sessions pooled | `memory/archive/g_filter_redundancy_proof_apr3.md` | Archived — but informs current COST_LT12 vs ORB_G5 deployment split |
| B2 | "ATR60 universal" from 2026-03-30 golden_nuggets — "MES beats own ATR 11/12, CME_PRECLOSE $519/yr" | 11/12 combos pos | `memory/archive/golden_nuggets_mar30.md` | NO — but archived |
| B3 | break_quality universal pooled — N=18-27K across MGC+MNQ+MES+all sessions | Cross-inst × cross-session | `memory/break_quality_research.md` | NO — but this memory file already explicitly notes Simpson's paradox risk |
| B4 | exit timing / recovery rate universal — "recovery rates stable at 17-23% across all thresholds" | 404K winning trades × 3 inst × all sessions | `memory/exit_rules_and_timing.md` | NO |
| B5 | E3 universally negative at 1000 (timeout) | Pooled E3 sample | `memory/exit_rules_and_timing.md` line 83 | NO |

### TIER C — already-known Simpson's-paradox cases (lesson filed, no re-audit needed)

| # | Claim | Outcome | Source |
|---|---|---|---|
| C1 | break_delay_min "6% WR spread" universal | REFUTED — session-specific; lesson filed | `memory/archive/break_delay_nogo.md` |
| C2 | 1s break speed universal | REFUTED — session-specific | `memory/archive/1s_break_speed_killed.md` |
| C3 | E0 wins 33/33 combos | REFUTED — structural artifact, NOT edge | `memory/entry_model_audit.md` |
| C4 | overnight_range pooled spread 4.5% | REFUTED — per-session ranges -2.3% to +13.5% | `memory/archive/mar26_session_recovery.md` |
| C5 | bull_short_avoidance universal p=0.0007 | REFUTED today — NYSE_OPEN-only | `memory/bull_short_avoidance_signal.md` |

### TIER D — phase-grouped research findings (already use BH-FDR per-family, low heterogeneity risk)

| # | Claim | Scope | Status |
|---|---|---|---|
| D1 | Phase 2.9 comprehensive 7-year stratification | 38 lanes × 7 yrs K=266 | Already per-session / per-year reported; heterogeneity explicit. |
| D2 | Phase 2.5 subset-t sweep | 9 Tier-1 / 14 Tier-4 | Per-lane by construction |
| D3 | Phase 2.4 / 2.7 retirement queue | Per-lane decisions | Per-lane |
| D4 | MGC portfolio diversifier | Per-lane | Per-lane |
| D5 | Phase D volume pilot D-0 | Single cell | Single cell |

## Summary

**Tier A count:** 5 (1 queued-activation + 2 partially-decomposed + 2 single-cell/N/A)
**Tier B count:** 5 (historical but still referenced)
**Tier C count:** 5 (already resolved)
**Tier D count:** 5 (intrinsically per-cell structured)

### Heterogeneity-risk assessment (my ranking before triage gate)

Ranked by (impact × plausibility × actionability):

1. **A1 — exchange_range / pit range** — QUEUED_FOR_ACTIVATION, framed 3/3 positive pooled, potentially deployable as size filter. If heterogeneity hides, next-session deployment decision would be wrong. **TOP PRIORITY.**
2. **A3 — comprehensive scan 8 non-rel_vol BH-global survivors** — not individually enumerated in memory but cited as "universal volume confirmation." If any survivor has per-cell sign-flip heterogeneity it propagates to feature-design assumptions. **SECOND.**
3. **B3 — break_quality pooled N=18-27K** — the memory file itself flags Simpson's-paradox risk explicitly ("session-specific findings from 164 tests are noise unless they survive universal pooling" — that framing is exactly backwards post-RULE 14). **THIRD.**
4. **A2 — H2 Path C garch_vol_pct≥70** — already partially decomposed; "NO CAPITAL" status reduces urgency but universal framing informs future sizing work. **FOURTH.**
5. **B4 + B5 — exit timing universal** — deeply baked into deployment expectations. If session-heterogeneous, the runbook's rescue/timeout rules may be session-wrong. **FIFTH.**

## Triage gate (stops here for user input)

- Does the above ranking match your priorities?
- Should we cap at top-3 Tier-A re-audits for this session, or go deeper?
- Are there claims I missed that should be added (e.g., any live-deployment filter rule you cite from memory)?

**No code changes yet.** Phase 2 triage decision drives Phase 3 per-claim audits.
