# MGC Regime-Deployment Truth Audit

**Date:** 2026-03-23
**Mode:** AUDIT + POLICY DESIGN | FAIL-CLOSED | NO CODE CHANGES
**Auditor:** Claude Code (institutional review terminal)

---

## Authority Conflict Resolution (HARDENING FIX 1)

**CONFLICT:** `RESEARCH_RULES.md` defines 5-tier classification (INVALID < 30, REGIME 30-99, PRELIMINARY 100-199, CORE 200+, HIGH-CONFIDENCE 500+). `docs/ARCHITECTURE.md` and `trading_app/config.py` define 3-tier (INVALID < 30, REGIME 30-99, CORE >= 100). No PRELIMINARY tier in code.

**RESOLUTION:** RESEARCH_RULES.md is methodology authority per CLAUDE.md conflict rules. For this audit:
- MGC N=175-177 is **PRELIMINARY** (RESEARCH_RULES.md), not CORE
- Code classifies it as CORE (config.py CORE_MIN_SAMPLES=100) — this is a code/docs drift, not an audit assumption
- `docs/ARCHITECTURE.md` classification table is **STALE** on this point — should be updated to match RESEARCH_RULES.md or the conflict should be explicitly documented

## Pinned Live State (HARDENING FIX 2)

**HANDOFF.md contains multiple truth blocks.** This audit pins to the LATEST:

> ### Truth State (verified Mar 23 2026, post-#82 cleanup)
> - validated_setups: 772 rows (MGC 6, MES 9, MNQ 757)
> - LIVE_PORTFOLIO: 8 specs (2 CORE, 6 REGIME). Resolves **MNQ 7, MES 1, MGC 0** (8 total)
> - Drift checks: 74 pass, 3 violations (all ML #61, frozen). #82 RESOLVED.

Earlier HANDOFF sections referencing "MNQ 8, MES 0" or "404 validated" are superseded.

---

## Frozen Baseline (verified from raw)

- 772 validated total. MGC 6, MES 9, MNQ 757.
- MGC 6 validated = 1 independent TOKYO_OPEN edge at 3 RR levels x 2 filters (ORB_G4, ORB_G4_CONT)
- All 6: noise_risk=True (OOS ExpR 0.143-0.186 vs floor 0.21)
- All 6: FDR-significant (adj_p 0.006-0.019 at K=105,640)
- All 6: WF-passed (WFE 0.534-0.833)
- All 6: all_years_positive=False (2019 waived as DORMANT)
- RR lock: MAX_SHARPE -> RR1.0 (ExpR=0.186, below LIVE_MIN_EXPECTANCY_R=0.22)
- ML: research-only, 3 open methodology FAILs
- Live: MNQ 7 + MES 1 + MGC 0 = 8

**All values verified from raw orb_outcomes + apply_tight_stop().** Zero mismatches with stored metadata.

---

## Key Findings

### 1. Noise Floor Sigma Overshoot

**CANDIDATE DEFECT — NOT RESOLVED**

Units confirmed: both null sigma (1.2) and real MGC 1-min increment std are in **points per bar**.

| Period | Real std (pts) | vs null sigma=1.2 | % of MGC trades |
|--------|---------------|-------------------|-----------------|
| 2017-2019 | 0.23-0.30 | 4.0-5.3x overshoot | ~0% (0-1 G4+ trades/year) |
| 2020-2024 | 0.45-0.65 | 1.9-2.7x overshoot | ~15% |
| **2025** | **1.17** | **1.02x (matched)** | **~55%** |
| **2026** | **3.24** | **0.37x (UNDER)** | **~30%** |

**Full-history overshoot:** 1.65x (full std), 2.82x (trimmed). Real and in correct units.

**Practical impact: UNCERTAIN.** sigma=1.2 matches the regime where 85% of MGC trades occur (2025-2026). The overshoot is dominated by low-vol years that produce near-zero G4+ trades. The noise floor may be approximately correct for the tradeable regime, even though it overstates the all-history null.

**Action required:** Block bootstrap from real MGC bars (bypasses sigma entirely) OR time-varying sigma null. Must apply to ALL instruments, not just MGC. Until then: no gate change.

### 2. RR Lock Policy

**STRUCTURALLY HONEST BUT CREATES UNNECESSARY CLIFF**

| RR | ExpR | Sharpe | OOS ExpR | WFE | Passes LIVE_MIN? |
|----|------|--------|----------|-----|-----------------|
| 1.0 | 0.186 | 0.236 | 0.172 | 0.749 | NO (< 0.22) |
| 1.5 | 0.235 | 0.228 | 0.186 | 0.833 | YES |
| 2.0 | 0.257 | 0.208 | 0.143 | 0.534 | YES |

Sharpe surface CV = 5.2% (flat). MAX_SHARPE is a coin flip that picks lowest-ExpR option. Creates double-block (noise + live-min) where only noise is structurally necessary.

**NOT the binding constraint.** All RR levels fail noise_risk regardless of lock. Lock cliff is secondary.

### 3. Regime Dependency

149/177 trades (84%) in 2025-2026. Gold tripled ($1,400 -> $4,900). If gold reverts to $2,000, G4+ qualifying days collapse. Edge is regime-conditional, not structural.

Double-break rate at TOKYO_OPEN: 83.5% (from raw daily_features).

### 4. Friction

$5.74 RT friction on $10/pt. At G4+ filter, median ORB = 6.20pts, friction = 9.3% of risk. Not the 57% claimed in earlier unfiltered analysis.

---

## Verdicts

- **MGC currently live-tradeable?** NO
- **Noise floor sound?** CANDIDATE DEFECT (sigma overshoot real, practical impact uncertain)
- **RR lock sound?** YES (structurally honest, creates secondary cliff on flat surface)
- **Any gate distorting truth?** Noise floor is the only candidate. Impact requires bootstrap to quantify.
- **Phase 3 binding?** NO (all 6 pass with DORMANT waiver)
- **ML justified?** NO (3 open methodology FAILs)
- **Honest next state:** SHADOW ONLY

---

## SURVIVED SCRUTINY
- BH FDR significance at K=105,640 (adj_p 0.006-0.019)
- Walk-forward (WFE 0.53-0.83, 3 windows)
- One independent edge, not 6 (honest accounting)
- Friction mechanism (structural, arithmetic)
- Pipeline instrument-agnostic (no MGC-specific suppression)
- Metadata accuracy (raw recompute = exact match)
- Sigma units confirmed (both in points per 1-min bar)

## DID NOT SURVIVE
- "MGC has a deployable edge" (0/6 pass noise floor)
- "Noise floor is too strict" (sigma matches 2025 regime where 55% of trades live)
- "Gates are unfair" (structural weakness is primary, gates are secondary)
- "57% friction" (wrong — 9.3% after G4 filter)

## UNSUPPORTED CLAIMS
- "Noise floor 0.21 is correctly calibrated" (sigma overshoot is real, direction is conservative, magnitude unknown without bootstrap)
- "Recalibration would let MGC pass" (unknown — depends on bootstrap result)
- "MGC will continue working in 2026+" (regime-dependent on gold > $2,400)

## POLICY CONFLICTS
- RESEARCH_RULES.md PRELIMINARY (100-199) vs ARCHITECTURE.md/config.py CORE (>= 100). MGC N=177 is PRELIMINARY by methodology authority but CORE by code.
- HANDOFF.md contains superseded truth blocks. Latest: MNQ 7, MES 1, MGC 0.

---

## Minimal Next Moves

1. **Shadow-track** MGC TOKYO_OPEN E2 RR1.5 ORB_G4 S075 — zero capital, record every signal
2. **Block bootstrap recalibration** — run across ALL instruments, adopt only if method is better (not because MGC passes). This is a methodology improvement, not an MGC rescue.
3. **Document the PRELIMINARY/CORE conflict** — either update ARCHITECTURE.md and config.py to match RESEARCH_RULES.md, or explicitly document the design choice

**Current canon: MGC 0 live. All 6 baseline variants noise-blocked. Shadow only.**
