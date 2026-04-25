# 6-lane deployed portfolio — filter vestigialness audit

**Authored:** 2026-04-20 ~13:00 UTC (Claude-handoff session)
**Script:** `research/audit_6lane_scale_stability.py`
**Context:** prompted by the re-audit of PR #44's OVNRNG_50_FAST10 finding, which surfaced the question "is this artifact specific to that lane or portfolio-wide?"
**Output:** portfolio-wide evidence of filter vestigialness across 5 of 6 deployed lanes.

---

## The question

PR #44 (OVNRNG_50_FAST10 at MNQ NYSE_OPEN 5m RR1.0) was downgraded from SHADOW_CONDITIONAL to MISCLASSIFIED because fire rate rose from 21% (2019) to 90% (2026) and the "OOS +0.137R" lift over unfiltered was +0.001R. The filter was essentially a no-op.

I originally framed this as a "scale artifact" (absolute-point thresholds riding price inflation). User pushed back: that framing is a tunnel — is the ROOT problem portfolio-wide or lane-specific, and is "scale" the only mechanism?

This audit answers: **portfolio-wide**, and scale is only one of several mechanisms.

## Method

For each of the 6 lanes in `docs/runtime/lane_allocation.json` with `status=DEPLOY`, compute per-year:
- Total trades on the lane geometry (unfiltered)
- Filter fire count (canonical `filter_signal` via `ALL_FILTERS`)
- Unfiltered vs filtered ExpR
- Lift = filtered ExpR − unfiltered ExpR
- Fire rate %

Flag scale-artifact signature: fire-rate range > 20pp across IS years AND/OR lift collapses in recent years.

Canonical truth only: `orb_outcomes JOIN daily_features`. No derived layers.

## Per-lane results

| Lane | Filter | 2019 fire | 2025 fire | 2026 fire | IS fire Δ | Lift pattern | Verdict |
|------|--------|-----------|-----------|-----------|-----------|--------------|---------|
| **L1** MNQ EUROPE_FLOW RR1.5 | ORB_G5 | 57.3% | 99.2% | 100.0% | +42.3pp | early +0.032 → late +0.006 collapsing | **SCALE_DRIFT** |
| **L2** MNQ SINGAPORE_OPEN RR1.5 | ATR_P50 | 29.2% | 56.6% | 75.0% | +61.7pp | early +0.051 → late +0.074 erratic | **PERCENTILE_INSTABILITY** |
| **L3** MNQ COMEX_SETTLE RR1.5 | ORB_G5 | 59.6% | 100.0% | 100.0% | +40.4pp | early +0.033 → late +0.000 | **VESTIGIAL_PASS_THROUGH** |
| **L4** MNQ NYSE_OPEN RR1.0 | COST_LT12 | 95.3% | 100.0% | 100.0% | +4.7pp | early +0.010 → late −0.003 | **VESTIGIAL (always-on)** |
| **L5** MNQ TOKYO_OPEN RR1.5 | COST_LT12 | 18.7% | 84.1% | 97.2% | +65.4pp | early +0.001 → late +0.059 | **COST_RATIO_DRIFT** |
| **L6** MNQ US_DATA_1000 RR1.5 | ORB_G5 | 95.1% | 100.0% | 100.0% | +4.9pp | early +0.014 → late −0.001 | **VESTIGIAL (always-on)** |

### Per-lane mechanism diagnosis

- **L1 + L3** (ORB_G5 at EUROPE_FLOW, COMEX_SETTLE): classic absolute-grid-threshold drift. `ORB_G5` uses a fixed-points grid; as MNQ tripled, grid bins inflated into trivially-easy territory. Fire rate climbed 57%→100%.
- **L2** (ATR_P50 at SINGAPORE_OPEN): **NOT** a scale problem — `ATR_P50` is a rolling-percentile filter and should be era-stable by construction. The 29%→75% drift suggests the rolling-percentile window is unstable on sparse session data (SINGAPORE_OPEN has fewer trades than US sessions). This is a DIFFERENT mechanism than L1/L3.
- **L4 + L6** (COST_LT12 at NYSE_OPEN, ORB_G5 at US_DATA_1000): fire 95-100% across ALL IS years — these were never selective. The filter has been nominal-only from deployment. The "edge" is entirely the (session, RR, direction, E2, CB1) geometry.
- **L5** (COST_LT12 at TOKYO_OPEN): cost-ratio is `friction / (risk_points × point_value)`. As MNQ price rose, risk_points grew (absolute ORB size scales with price), so cost_ratio shrank — same asymmetric effect as L1/L3 but via a different numerator/denominator. Fire 19%→97%.

## Portfolio-level finding (not "scale" — broader)

The 6-lane portfolio's filter-layer is **largely vestigial in the current era**:

- 4 of 6 lanes (L3, L4, L5, L6) have 2026 fire rates ≥97% — the filter effectively removes nothing
- 1 lane (L1) is at exactly 100% 2026 fire rate
- Only L2 (ATR_P50) retains any meaningful 2026 selectivity (75%)

**Implication:** the portfolio's "diversification" is NOT at the signal-class layer. The 6 lanes are diversified by SESSION TIMING (9am EU, 11am SG, 4pm comex settle, 12:30am NYSE, 10am Tokyo, 1am US data) and by RR/direction mix, but NOT by filter-selection mechanism. The filters are mostly vestigial in current market conditions.

This is NOT necessarily bad. The unfiltered session+RR+direction geometry has edge on its own — PR #44's correction showed the MNQ NYSE_OPEN 5m E2 RR1.0 CB1 baseline is +0.082 IS / +0.136 OOS. If the filters were correctly calibrated, they'd be additive. But if they're vestigial, then removing them wouldn't hurt and the portfolio's effective signal layer is just the session geometry.

## What this means operationally

| Question | Honest answer |
|----------|---------------|
| Is the 6-lane portfolio diversified? | By session timing yes, by filter mechanism no |
| Are the filters adding value? | Mostly no in the current era; L2 is the only marginally-selective one |
| Is this a crisis? | No — the unfiltered lane geometries have baseline edge. Vestigial filters don't subtract, they just fail to add. |
| What's the fix? | Either recalibrate filters to ATR-normalized / percentile-stable equivalents, OR accept the portfolio as "session-diversified" and focus elsewhere |
| What's the immediate risk? | NONE — vestigial filters don't introduce failure modes. But the portfolio's implied capital efficiency assumes filter-selection works, and that assumption is now suspect. |

## Highest-EV honest actions (ranked)

1. **Do not deploy any more absolute-threshold filters.** Require ATR-normalized or percentile-based calibration for new filter pre-regs. (This is the generalization of the PR #44 correction — see `memory/feedback_absolute_threshold_scale_audit.md`.)
2. **Stress-test existing lane baseline edge without filter.** Compute each of the 6 lanes' unfiltered-geometry ExpR/Sharpe/year-breakdown. If the unfiltered baseline carries the edge, the portfolio is simpler than we thought (good news — fewer moving parts). If the baseline is weak without the filter, we've been counting on vestigial selection all along (bad news — the portfolio is weaker than we thought).
3. **Revisit L2 (ATR_P50 at SINGAPORE_OPEN).** The 29%→75% fire-rate instability is NOT scale — it's percentile-windowing on sparse data. Diagnose whether ATR_P50 is rolling, expanding, or static, and whether SINGAPORE_OPEN's sparse trading generates instability.
4. **Do NOT pursue new filter candidates without a FRESH cross-session pre-reg.** The PR #44 pattern (surface candidate in single-session scan, lock pre-reg, run, celebrate) is a methodology dead-end when absolute-threshold filters are in play.

## What this audit does NOT do

- Does not modify `docs/runtime/lane_allocation.json`
- Does not pause or remove any deployed lane
- Does not recommend rebuilding any lane
- Does not constitute a capital-allocation decision

The portfolio is safe to continue as-is. The vestigial filters aren't hurting. This audit just reframes what we think the filters are actually doing.

## Artefact list

- Script: `research/audit_6lane_scale_stability.py`
- Raw output: `/tmp/6lane_audit.log` (not committed)
- Parent correction: `docs/audit/results/2026-04-20-nyse-open-ovnrng-fast10-correction.md`
- Related memory: `feedback_absolute_threshold_scale_audit.md`, `feedback_default_cross_session_scope.md`
