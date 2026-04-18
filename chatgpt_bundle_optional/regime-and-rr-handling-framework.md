# Institutional-Grade Regime & RR Handling Framework

**Date authored:** 2026-04-15
**Authority:** Complementary to `.claude/rules/backtesting-methodology.md`, `docs/institutional/pre_registered_criteria.md`, `docs/institutional/edge-finding-playbook.md`, `docs/institutional/mechanism_priors.md`.
**Purpose:** Codify how we handle RR variant validation and regime-conditional trading. Address gaps identified 2026-04-15.

---

## 1. Motivation

The 2026-04-15 audits surfaced two architectural gaps:

1. **RR variant handling:** each (session, instrument, apt, RR, entry, dir, filter) combination is validated independently. Cross-RR concordance is informational (T5 INFO) not a deployment gate. Drift in one RR while others hold can go undetected.
2. **Regime-aware trading:** we have regime features, some binary regime filters (ATR_P50, OVNRNG_100), but no unified regime classifier and no conditional-activation infrastructure. Criterion 12 (Shiryaev-Roberts monitor) is a pre-reg requirement, not live production state.

Both gaps reduce our ability to deploy true institutional-grade strategies. **This document proposes the framework; implementation is staged.**

---

## 2. Part A — RR Variant Handling

### 2.1 Current state

- 3 RR variants per cell: {1.0, 1.5, 2.0}
- Each validates independently through backtest → T0-T8 → pre_registered_criteria 12-gate
- `validated_setups` stores one row per (session, instrument, apt, RR, entry, dir, filter, confirm_bars)
- `T5 T5_family` in `t0_t8_audit_prior_day_patterns.py` reports "INFO — family evaluated in mega-exploration" — does NOT enforce
- No explicit cross-RR deployment gate

### 2.2 Gap

A cell may pass at RR1.5 but fail at RR1.0 and RR2.0. Our current system would deploy RR1.5 alone. The missing-family is a red flag:

- **Robust edge:** should work across at least 2 of 3 RRs (allowing RR-specific noise/fit on extremes)
- **Single-RR survivor:** likely overfit to RR-specific outcomes or at a knife-edge of stop/target geometry

### 2.3 Proposed rule — cross-RR family gate

**Institutional promotion to `validated_setups` requires:**

1. Target RR passes the 12-criterion gate (existing requirement)
2. AT LEAST ONE other RR passes T0 + T1 + T2 + T6 (core robustness tests)
3. All 3 RRs agree in direction (sign of Δ_IS matches)
4. T4 sensitivity passes across all 3 RRs (if applicable)
5. No RR has ExpR opposite-direction in trailing 12 months (recent behavior check)

If 2+ of 3 RRs meet criteria 1 AND others meet 2-5 → **FAMILY_CONCORDANT** → deploy
If only 1 RR meets criterion 1 → **SINGLE_RR_ISOLATED** → research-provisional only, not live

### 2.4 Validation of existing portfolio under this rule

Need retroactive audit: for each of 6 deployed MNQ lanes + 124 validated_setups grandfathered, check cross-RR family concordance. Deferred — one-session batch job.

### 2.5 Implementation

- New function `trading_app/rr_family_audit.py::check_family_concordance(cell)` → returns `FAMILY_CONCORDANT` / `SINGLE_RR_ISOLATED`
- New drift check in `pipeline/check_drift.py` — drift check #N enforces family concordance on any new validated_setups insert
- Update `docs/institutional/pre_registered_criteria.md` to add criterion 13: Cross-RR family concordance
- `research/` scripts for cross-RR T0-T8 (done ad-hoc tonight for MGC LONDON_METALS F2_NEAR_PDL_30)

### 2.6 Cost / benefit

- **Cost:** audit per new validated_setup = 3× the compute (3 RR variants tested). Cheap.
- **Benefit:** kills single-RR overfits before they waste live capital. Institutional-grade gate.

---

## 3. Part B — Regime-Aware Trading

### 3.1 Current state

What we have:
- Regime features: `atr_vel_regime`, `day_type`, `atr_20_pct`, `garch_forecast_vol_pct`, `gap_type`, `is_nfp_day`, `is_opex_day`, DOW flags, `overnight_range_pct`
- Binary regime filters deployed: ATR_P50 (SINGAPORE_OPEN), OVNRNG_100 (COMEX_SETTLE), ORB_G5 (multiple), VWAP_MID_ALIGNED (US_DATA_1000)
- T7 per-year test catches year-level instability (labels as ERA_DEPENDENT)

What we LACK:
- Unified regime classifier (integrated state: vol × trend × day-type × calendar)
- Conditional-activation (lane X trades only in regime A, monitors in regime B)
- Shiryaev-Roberts monitor (criterion 12 NOT in production)
- Regime-conditional sizing (Carver framework Phase D — stub only)
- "Monitor-only in regime Y" mode (we have signal-only shadow globally, not per-regime)

### 3.2 Three-tier regime framework (PROPOSED)

#### Tier R1 — Binary regime filter (DEPLOYED TODAY)
Single feature gate. Fire → take day. Fail → skip. Currently our only mode.
- Examples: ATR_P50, ORB_G5, OVNRNG_100
- Strength: simple, stage-1 deployable
- Weakness: binary, no intensity info, single-feature only

#### Tier R2 — Composite regime classifier (PROPOSED BUILD)
Multi-feature state machine. Combines {vol percentile, vol-velocity, day-type, calendar, overnight-dynamics} into a discrete regime label.
- Candidate regime labels: `calm_trend`, `vol_expansion`, `range_day`, `event_day`, `reversion_setup`, `unknown`
- Each lane has a `compatible_regimes` list — trades ONLY when current regime is in list
- Dashboard shows current regime + which lanes are live/monitor

#### Tier R3 — Continuous regime forecast (Carver-grounded, stub as Phase D)
Regime intensity as continuous forecast feeding size-scaling.
- Maps regime state to position size [0, 2x]
- Canonical: Carver Ch 9-10 vol targeting + regime-conditional forecast combination
- Requires forecast_combiner + backtest infrastructure

### 3.3 User's "monitor-only" request — explicit architecture

User said: "some people trade only in regime — they just monitor it." Valid institutional pattern.

**Proposed `lane_state` enumeration** (extends current `DEPLOY` / `WATCH` / etc.):
- `LIVE` — take all trades when filter fires
- `LIVE_REGIME_ONLY` — take trades only when filter fires AND current regime in `compatible_regimes`
- `MONITOR_ONLY_OUT_OF_REGIME` — signal-only logging when filter fires but regime is NOT compatible (accumulate OOS without capital at risk)
- `DORMANT` — strategy exists, allocator hasn't picked it

Infrastructure:
- `lane_allocation.json` extended with `compatible_regimes: [...]` per lane
- `prop_profiles.py` → `pre_session_check.py` gates orders on regime state
- `trading_app/regime_classifier.py` (NEW module) → computes current regime from features, emits categorical state
- Dashboard displays: lane × regime status matrix

### 3.4 Regime-stability test addition to T0-T8

**New test T9 — regime stability** (propose):
- Split IS data by computed regime
- Compute per-regime ExpR, Sharpe
- PASS if: ExpR positive in ≥ 50% of regimes where N≥30, AND no regime shows ExpR < -0.15 at N≥30
- FAIL (label `REGIME_BRITTLE`): signal only works in 1 narrow regime — deployment restricted to `LIVE_REGIME_ONLY` in that regime

### 3.5 Shiryaev-Roberts monitor — wire to production

Criterion 12 currently calls for live drift detection. Implementation:
- Per-lane live Shiryaev-Roberts statistic computed daily from trailing 30-day realized ExpR vs IS baseline
- Alarm threshold: standard ARL=200 per Pepelyshev-Polunchenko 2015
- On alarm: lane auto-moves to `MONITOR_ONLY_OUT_OF_REGIME` state pending manual review
- Canonical SR implementation needed in `trading_app/sr_monitor.py`

---

## 4. Staged delivery — what to build when

### Stage R-1 (this sprint, cheap)
- Retroactively cross-RR audit 6 deployed MNQ lanes — identify any SINGLE_RR_ISOLATED cells → demote to research-provisional
- Document current regime handling limits in `lane_allocation.json` comments
- No architecture changes yet

### Stage R-2 (2-3 weeks)
- Build `trading_app/regime_classifier.py` with 6-label state machine using existing features only
- Add `compatible_regimes` field to `lane_allocation.json` (default: all regimes for existing lanes)
- Wire classifier output to dashboard (read-only)
- No order-gate changes yet (just visibility)

### Stage R-3 (4-6 weeks)
- Implement `LIVE_REGIME_ONLY` lane state in `pre_session_check.py`
- Signal-only shadow for any lane transitioning to regime-conditional
- Validate 1 lane per week (start with deployed lanes not currently regime-aware)
- Backtest the 6-lane portfolio with regime gates — compare Sharpe uplift

### Stage R-4 (6-10 weeks)
- Wire Shiryaev-Roberts monitor to production
- Auto-transition on alarm
- Weekly regime-report in dashboard

### Stage R-5 (merges with Phase D)
- Tier R3 continuous regime forecast via Carver combiner
- Size scaling by regime intensity

---

## 5. What tonight's MGC findings imply for this framework

Applying the proposed rules to tonight's verified MGC cells:

### MGC LONDON_METALS O30 long F2_NEAR_PDL_30 (NEW finding)
- RR1.0: 6P/2F (CONDITIONAL if T8 methodology fixed)
- RR1.5: 7P/1F CONDITIONAL
- RR2.0: 7P/1F CONDITIONAL
- **Cross-RR family gate:** FAMILY_CONCORDANT (2 of 3 RRs meet core robustness; direction matches; T4 sensitivity passes all 3)
- **Regime test T9 (not yet implemented):** DEFERRED
- **Verdict under new framework:** eligible for research-provisional → signal-only shadow → live after OOS accumulates and T9 passes

### MGC SINGAPORE_OPEN O15 long F3_NEAR_PIVOT_15 (previously verified, SKIP signal)
- RR1.5: CONDITIONAL
- RR2.0: CONDITIONAL
- RR1.0: status needs re-audit per this framework
- **Cross-RR family:** likely CONCORDANT (same direction, similar magnitude on RR1.5 and 2.0)

### MGC LONDON_METALS O5 short rel_vol_HIGH (volume finding)
- RR1.0: CONDITIONAL
- RR1.5: CONDITIONAL
- RR2.0: not audited — ADD TO NEXT SESSION

### MGC 2026 regime shift flag (M1 WFE=0.33)
- Under regime framework: **regime classifier would detect this** as a state change, move lane to MONITOR_ONLY, preventing unaware deployment. Today we have the FLAG but no automated response — exactly the gap this framework fills.

---

## 6. Honest self-assessment of current institutional-grade level

| Capability | Current state | Target | Gap |
|------------|---------------|--------|-----|
| Per-cell T0-T8 audit | 100% | 100% | — |
| Pre-registration before discovery | 90% (Phase 0 wired) | 100% | some confirmatory scans skip YAML — acceptable exploratory |
| Multi-framing BH-FDR | 100% (as of tonight) | 100% | — |
| Look-ahead feature gates | 100% (as of tonight) | 100% | — |
| Cross-RR family gate | **0%** (informational only) | 100% | Stage R-1 |
| Unified regime classifier | 0% | 100% | Stage R-2 |
| Conditional-activation lanes | 0% | 100% | Stage R-3 |
| Shiryaev-Roberts monitor | 0% (documented, not wired) | 100% | Stage R-4 |
| Continuous regime forecast | 0% | Optional (Phase D) | Stage R-5 |
| T8 cross-instrument twin correctness | Partial (MNQ for all — wrong for MGC) | 100% | Fix in queue |

Overall institutional-grade: **~65%** for discovery methodology, **~40%** for deployment infrastructure. Gaps are architecture, not discipline — we know what to do, just haven't built it yet.

---

## 7. Resource requirements

User asked: do we have the resources?

- **Data:** YES. All regime features exist in `daily_features`. Nothing new to ingest.
- **Research capacity:** YES. Can run cross-RR audits and regime T9 tests with existing infrastructure.
- **Dev capacity:** Stage R-1 cheap (retroactive audit). Stage R-2/R-3 each ~2-3 weeks focused. Stage R-4 ~2 weeks. Can be done serially over 2-3 months without blocking current trading.
- **Capital risk during build:** ZERO — all changes gated behind `LIVE` / `MONITOR_ONLY` states. No existing lanes disrupted until explicit migration.

**Verdict:** we have the resources. The question is prioritization — this vs Phase D (Carver) vs tick-delta ingestion vs Phase C (E_RETEST). All have real value.

---

## 8. Recommended sequencing

Priority-ranked for next 10 weeks:

1. **Stage R-1 cross-RR retroactive audit** (1 session) — cheapest, highest information, no build
2. **Stage R-2 regime classifier visibility** (2-3 weeks) — foundation for everything else, no deployment risk
3. **Phase D-0 backtest** (1 week, already specced) — parallel track
4. **Stage R-3 conditional-activation** (3-4 weeks) — requires R-2 complete
5. **Phase D-1 signal-only shadow** (4 weeks, overlaps R-3) — volume size-scaling
6. **Stage R-4 Shiryaev-Roberts wire** (2 weeks) — completes criterion 12
7. **Phase D-2 live size-scaled** (overlapping)
8. **Phase C E_RETEST** (deferred — independent research thread)
9. **Tick-delta ingestion** (deferred — biggest data investment)

By end of 10 weeks: Stage R-1 through R-4 done, Phase D in live deployment, criterion 12 active. Institutional-grade deployment infrastructure complete enough for confident scaling.

---

## 9. What to NOT do

Per user's "no skipping or rushing institutional grade":

- Do NOT deploy any new cell under the old single-RR rule. Use the cross-RR family gate retroactively.
- Do NOT claim criterion 12 compliance until Shiryaev-Roberts is actually wired and alarming in production.
- Do NOT short-circuit regime classification — build the classifier even if no lanes initially use it (visibility first).
- Do NOT deploy MGC lanes to live before regime T9 test is implemented (MGC 2026 regime shift flag needs architectural handling).
- Do NOT skip T8 methodology fix — MGC twin=MNQ is wrong, fix before any MGC T8 result is treated as binding.
