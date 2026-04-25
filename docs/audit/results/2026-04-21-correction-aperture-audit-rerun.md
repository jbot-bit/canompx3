# CORRECTION — PR #52 / PR #54 / L6 diagnostic aperture bug

**Date:** 2026-04-21
**Branch:** `research/correction-aperture-audit-rerun`
**Script:** `research/audit_lane_baseline_decomposition_v2.py`
**Raw JSON:** `docs/audit/results/2026-04-21-correction-aperture-audit-rerun.json`

**Supersedes (for L2 and L6 only — L1/L3/L4/L5 numbers unchanged):**
- PR #52 / commit `371c97f4` — 6-lane unfiltered baseline stress-test
- PR #54 / commit `d2a931f4` — L2 ATR_P50 stability audit (entirely)
- Uncommitted L6 diagnostic draft (stashed on `research/l6-us-data-1000-2026-diagnostic`)

---

## What was wrong

### The bug

Prior audits re-encoded `parse_strategy_id` in
`research/audit_6lane_scale_stability.py` (PR #47, commit `007ecd6b`)
and hardcoded `orb_minutes = 5` in the SQL query:

```sql
WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = 5
      AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
```

The canonical parser at `trading_app/eligibility/builder.py:122-125`
extracts `orb_minutes` from the `_O*` suffix:

```python
orb_minutes = 5
if remaining and remaining[-1].startswith("O") and remaining[-1][1:].isdigit():
    orb_minutes = int(remaining[-1][1:])  # _O15 → orb_minutes = 15
```

Two of six DEPLOY lanes carry the `_O15` aperture suffix:
- L2 `MNQ_SINGAPORE_OPEN_..._ATR_P50_O15` → canonical `orb_minutes=15`
- L6 `MNQ_US_DATA_1000_..._ORB_G5_O15` → canonical `orb_minutes=15`

**All prior PR #52, PR #54, and the L6 diagnostic draft queried
the wrong 5-minute ORB outcomes for L2 and L6.** Live deployment runs
these lanes on 15-minute ORB windows.

This is the exact failure mode `.claude/rules/research-truth-protocol.md`
§ "Canonical filter delegation" is designed to prevent: research scripts
that re-encode canonical parsing logic drift from the canonical source
over time. PR #47 added the re-encoding; PR #52, #54, and the L6 draft
inherited the drift.

### How it was caught

Code-review agent dispatched on 2026-04-21 after user asked for audit
of recent commits. Finding **H1** identified the discarded `_O15`
suffix. Sanity check on `orb_outcomes`:

| Lane | orb_minutes=5 (wrong) | orb_minutes=15 (correct) |
|------|-----------------------|--------------------------|
| L2 baseline full | n=1794 ExpR=−0.002R | n=1786 ExpR=**+0.052R** |
| L6 baseline full | n=1742 ExpR=+0.088R | n=1550 ExpR=**+0.108R** |

Both lanes move materially. L2's shift from net-negative to
net-positive overturns the central finding of both PR #52 (L2 row)
and PR #54 (entire audit).

### Fix applied

This correction audit:

1. Imports canonical `parse_strategy_id` from
   `trading_app.eligibility.builder` (delegates — does not re-encode).
2. Passes `orb_minutes` from the parsed spec into the SQL query.
3. Also addresses code-review findings **H2**, **H3**, **M1**–**M4**
   (see § Methodology fixes below).

The buggy `research/audit_6lane_scale_stability.py::load_lane_universe`
is untouched by this correction (it's a PR #47 artifact and out of
scope here). Future callers of that helper inherit the same bug until
it is fixed. Filed as follow-up below.

---

## Corrected results

### Section A — 6-lane baseline × filter 2×2 (CORRECTED)

All 6 lanes, classification on IS (n=1495–1786 per lane) with
Welch fire-vs-non-fire t-test. Lanes with `_O15` suffix use
`orb_minutes=15`; others use 5.

| Lane | orb_min | unfilt IS t | filt IS t | Welch p | non_fire_n | NEW verdict | OLD verdict (PR #52) |
|------|---------|-------------|-----------|---------|------------|-------------|----------------------|
| L1 EUROPE_FLOW ORB_G5 | 5 | +1.61 | +2.28 | 0.001 | 135 | FILTER_CORRELATES_WITH_EDGE | FILTER_IS_THE_EDGE |
| L2 SINGAPORE_OPEN ATR_P50 O15 | **15** | **+1.86** | **+2.96** | **0.014** | 806 | FILTER_CORRELATES_WITH_EDGE | FILTER_IS_THE_EDGE |
| L3 COMEX_SETTLE ORB_G5 | 5 | +2.37 | +3.15 | <0.001 | 81 | BOTH_CONTRIBUTE | BOTH_CONTRIBUTE |
| L4 NYSE_OPEN COST_LT12 | 5 | +3.47 | +3.50 | 0.593 | 24 | FILTER_VESTIGIAL | FILTER_VESTIGIAL |
| L5 TOKYO_OPEN COST_LT12 | 5 | +2.64 | +3.27 | 0.028 | 772 | BOTH_CONTRIBUTE | BOTH_CONTRIBUTE |
| L6 US_DATA_1000 ORB_G5 O15 | **15** | **+3.42** | +3.39 | 0.56 | **4** | FILTER_UNTESTABLE_BASELINE_EDGE | FILTER_VESTIGIAL |

**Naming change (per code-review H3):** "FILTER_IS_THE_EDGE" is renamed
to "FILTER_CORRELATES_WITH_EDGE." The Welch test shows selection
correlates with pnl_r; it does NOT establish causation vs an omitted
vol-regime confound. The L2 audit (Section B) shows this filter's
mechanism is MNQ-wide vol regime, not session-specific selection.

**L2 now has a positive baseline.** unfilt IS t=+1.86, p≈0.06. Still
marginal, but NOT net-negative. The PR #54 premise "lane would be
deploy-killed without the filter" is wrong — more accurate:
"deploy-not-selected on unfiltered geometry alone."

**L6 flipped classification.** At correct aperture, L6 has unfilt IS
t=+3.42 (stronger than at wrong aperture), and the filter fires on
1491/1495 IS trades (n_non_fire=4) — the Welch test is structurally
untestable. L6 is now `FILTER_UNTESTABLE_BASELINE_EDGE` rather than
`FILTER_VESTIGIAL`; the interpretation is similar (filter is
operationally a no-op at 99.7% fire rate) but the label is honest
about the untestability.

### Multiple-testing (K=6, BH-FDR at q=0.05) — added per code-review M4

Welch p-values sorted: [<0.001, 0.001, 0.014, 0.028, 0.56, 0.59].

| Rank | Lane | p | BH threshold | Status |
|------|------|---|--------------|--------|
| 1 | L3 COMEX_SETTLE | <0.001 | 0.0083 | PASS |
| 2 | L1 EUROPE_FLOW | 0.001 | 0.0167 | PASS |
| 3 | L2 SINGAPORE_OPEN | 0.014 | 0.0250 | PASS |
| 4 | L5 TOKYO_OPEN | 0.028 | 0.0333 | PASS |
| 5 | L6 US_DATA_1000 | 0.56 | 0.0417 | FAIL |
| 6 | L4 NYSE_OPEN | 0.59 | 0.0500 | FAIL |

4 of 6 pass BH-FDR at q=0.05. L2's Welch discrimination survives
multiple-testing correction. L4/L6 fail (confirming their vestigial /
untestable status).

### 2026 OOS row (UNDERPOWERED — monitoring only)

| Lane | OOS n | unfilt ExpR | filt ExpR | Δ |
|------|-------|-------------|-----------|---|
| L1 | 72 | +0.293 | +0.293 | 0 |
| L2 | 67 | +0.099 | **+0.062** | **−0.037** |
| L3 | 66 | +0.006 | +0.006 | 0 |
| L4 | 71 | +0.136 | +0.136 | 0 |
| L5 | 72 | +0.149 | +0.153 | +0.004 |
| L6 | 55 | **+0.161** | +0.161 | 0 |

**L6 2026 is POSITIVE (+0.161R)**, not negative (−0.034R). The previous
"only 2026-negative lane" framing was an aperture artifact. The lane
is currently 2026-profitable at correct data.

**L2 2026 filter contribution is NEGATIVE (−0.037R).** The filter
selected 54 of 67 2026 trades and those 54 underperformed the full 67 by
−0.037R. At n=54/67 this is within noise (95% CI ±0.24R per-trade) but
reverses the sign from the PR #54 report (+0.024R at wrong aperture).

---

### Section B — L2 ATR_P50 stability deep-dive (CORRECTED, orb_minutes=15)

#### Per-year Welch fire-vs-non-fire (corrected aperture)

| Year | N | fire% | atr_μ | atr_p25 | atr_p75 | ExpR_fire | ExpR_nonf | Δ | Welch t | p |
|------|---|-------|-------|---------|---------|-----------|-----------|---|---------|---|
| 2019 | 171 | 29.2% | 31.6 | 5.5 | 55.0 | +0.187 | −0.055 | +0.242 | +1.33 | 0.189 |
| 2020 | 258 | 79.8% | 67.5 | 59.6 | 82.9 | +0.056 | +0.219 | **−0.163** | **−0.94** | 0.348 |
| 2021 | 259 | 42.1% | 42.1 | 9.1 | 72.2 | +0.054 | −0.067 | +0.122 | +0.86 | 0.392 |
| 2022 | 258 | 59.3% | 57.8 | 32.7 | 82.9 | +0.210 | −0.017 | +0.227 | +1.58 | 0.117 |
| 2023 | 258 | 18.2% | 28.8 | 9.9 | 43.5 | −0.127 | −0.008 | **−0.120** | **−0.69** | 0.494 |
| 2024 | 259 | 78.8% | 69.5 | 54.0 | 90.9 | +0.128 | −0.181 | +0.309 | +1.90 | 0.061 |
| 2025 | 256 | 56.2% | 55.9 | 20.9 | 88.1 | +0.168 | +0.019 | +0.149 | +1.02 | 0.309 |
| 2026 | 67 | 80.6% | 74.6 | 56.0 | 87.3 | +0.062 | +0.252 | **−0.190** | **−0.51** | 0.616 |

**Zero years individually significant at p<0.05** (vs. 2 years at wrong
aperture). **Three years show REVERSE sign** (2020, 2023, 2026 — filter
selects worse than non-fire in those years). The aggregate Welch p=0.014
depends on the ~0.24 lift in 2019 and 2022.

#### Early-vs-late IS (trade-count median, not calendar midpoint — M3 clarification)

IS midpoint: trade #859 on 2022-08-31 (trade-count, not calendar).

| Half | Trade # | Calendar | n | Welch t | p | Δ |
|------|---------|----------|---|---------|---|---|
| Early | 1–859 | 2019-05-06 to 2022-08-30 | 859 | +1.73 | **0.083** | +0.133 |
| Late | 860–1719 | 2022-08-31 to 2025-12-31 | 860 | +1.77 | **0.077** | +0.137 |

**Neither half individually passes p<0.05 at corrected aperture.** Both
are borderline (p≈0.08). This is weaker than at the wrong aperture,
where both halves cleared p<0.05.

#### Per-year KS: SINGAPORE_OPEN atr_20_pct vs all-MNQ (M1 fix — replaces pooled KS)

| Year | KS D | KS p |
|------|------|------|
| 2019 | 0.029 | 1.00 |
| 2020 | 0.025 | 1.00 |
| 2021 | 0.021 | 1.00 |
| 2022 | 0.018 | 1.00 |
| 2023 | 0.008 | 1.00 |
| 2024 | 0.021 | 1.00 |
| 2025 | 0.026 | 1.00 |
| 2026 | 0.036 | 1.00 |

No per-year distribution divergence. PR #47's "sparse-session
instability" diagnosis is still refuted — but by per-year KS stats,
not by a single pooled KS (which was the code-review M1 concern).

#### Trailing 3-year Welch (corrected aperture)

| End year | N | Δ | Welch t | p |
|---------|---|---|---------|---|
| 2021 | 688 | +0.090 | +1.06 | 0.288 |
| 2022 | 775 | +0.108 | +1.30 | 0.193 |
| 2023 | 775 | +0.133 | +1.60 | 0.109 |
| 2024 | 775 | +0.165 | +2.06 | **0.040** |
| 2025 | 773 | +0.137 | +1.69 | 0.091 |
| 2026 | 582 | +0.159 | +1.56 | 0.120 |

Only the 2022–2024 window clears p<0.05. All other recent windows are
borderline.

#### Revised L2 classification

**FILTER_CORRELATES_WITH_EDGE (marginal, softened per H3).** The filter
and the lane's pnl_r are correlated — aggregate Welch p=0.014. But:

- Zero individual years pass p<0.05
- Three years have reverse sign (incl. 2026 OOS)
- Both IS halves are p≈0.08 (borderline)
- Rolling 3y p is 0.04–0.29 with only one window passing

This filter is a vol-regime gate. The 2019 and 2022 outlier years
drive the aggregate signal. Operationally, it adds ~+0.06R to the
baseline on the full IS sample, which matters less than it appeared
in the wrong-aperture audit.

**L2 is NOT "fully filter-dependent."** The baseline (unfilt IS
ExpR=+0.050R, t=+1.86) is positive-leaning, not net-negative. Removing
the filter would leave a borderline-positive lane, not a losing lane.

---

### Section C — L6 2026 diagnostic (CORRECTED, orb_minutes=15)

**Primary verdict (per code-review H2):** one-sample t-test on OOS.

- OOS n=55, ExpR=**+0.161R**, t=+0.96, p=0.339.
- Cannot reject H0: lane mean = 0 at α=0.05.
- But ExpR is positive, so the lane is not "broken in 2026."

**Power analysis:**
- IS effect size d = IS_mean/IS_std = +0.0885
- Power to detect IS-size effect at n=55: **10.1%**
- Verdict: **SEVERELY UNDERPOWERED**

At 10% power, the absence of a significant OOS signal tells us nothing
about whether the IS edge persists. We'd need ~200+ OOS trades before
a non-significant t-test became informative.

#### Descriptive bootstrap (not primary verdict — per H2)

10,000 iid draws of size 55 from IS pnl_r distribution:
- Observed OOS mean = +0.161 sits at bootstrap percentile 63.5 (well
  within null distribution).
- 5th–95th pct of size-55 bootstrap means: [−0.165, +0.365]

**Caveat (new):** iid resampling does NOT model trade autocorrelation.
If trades cluster (consecutive similar days), a block-bootstrap would
give wider intervals. This is descriptive, not inferential.

#### Calendar decomposition (CORRECTED aperture)

| Bucket | IS | OOS |
|--------|----|-----|
| NFP days | n=61 ExpR=−0.011 | n=3 ExpR=+0.649 |
| non-NFP days | n=1434 ExpR=+0.111 | n=52 ExpR=+0.133 |
| OPEX days | n=69 ExpR=+0.189 | n=2 ExpR=+0.219 |
| non-OPEX days | n=1426 ExpR=+0.102 | n=53 ExpR=+0.159 |
| Monday | n=301 ExpR=+0.047 | n=11 ExpR=+0.114 |
| Tuesday | n=299 ExpR=+0.188 | n=11 ExpR=−0.104 |
| Friday | n=283 ExpR=+0.108 | n=10 ExpR=+0.477 |
| CPI-adj (day 10–15) | n=299 ExpR=+0.122 | n=10 ExpR=+0.232 |

**The "3 max-losers on OPEX" cell from the wrong-aperture draft does
not exist at correct aperture.** OPEX 2026 OOS is n=2, ExpR=+0.219
(positive). The draft's "3 OPEX trades all −1.0R" was an aperture-5
observation; at aperture 15 the cell is profitable.

#### Vol regime (corrected aperture)

| Bin | IS n | IS ExpR | OOS n | OOS ExpR | OOS % |
|-----|------|---------|-------|----------|-------|
| Q1 (0–25) | 418 | +0.161 | 0 | — | 0.0% |
| Q2 (25–50) | 270 | +0.024 | 12 | +0.018 | 21.8% |
| Q3 (50–75) | 334 | +0.070 | 8 | +0.225 | 14.5% |
| Q4 (75–100) | 471 | +0.131 | 35 | +0.195 | 63.6% |

2026 OOS is Q4-heavy (63.6%) and profitable in Q4 (+0.195R). Q1 has
no OOS observations (MNQ's 2026 ATR-20 percentile has not dipped below
25 so far). Q2/Q3 are thin-sample positive.

#### Revised L6 classification

**UNINFORMATIVE_OOS.** At n=55 with 10% power, the OOS cannot confirm
or refute the IS edge. The descriptive numbers are positive-leaning
(ExpR=+0.161, Q4 +0.195) and fully consistent with the IS distribution.
No structural break signal. No pause recommended.

The lane is currently working.

---

## Summary of what changed vs PR #52 / #54 / L6 draft

| Finding | PR #52/#54/draft (wrong aperture) | This correction (correct aperture) |
|---------|-----------------------------------|-------------------------------------|
| L2 unfilt IS ExpR | −0.010R, t=−0.38 (net-negative) | +0.050R, t=+1.86 (marginal positive) |
| L2 classification | FILTER_IS_THE_EDGE | FILTER_CORRELATES_WITH_EDGE (softened) |
| L2 deploy risk | "would be deploy-killed without filter" | "would be deploy-not-selected; borderline positive alone" |
| L2 2026 filter delta | +0.024R (filter helps) | **−0.037R** (filter hurts at small n) |
| L2 per-year stability | 2 years individually significant | **0 years individually significant** |
| L2 both IS halves Welch | p=0.049 / 0.027 | p=0.083 / 0.077 (both borderline) |
| L6 2026 unfilt ExpR | **−0.034R** (negative) | **+0.161R** (positive) |
| L6 "problem lane" framing | Only 2026-negative lane | **Lane is 2026-profitable** |
| L6 classification | FILTER_VESTIGIAL | FILTER_UNTESTABLE (99.7% fire rate) |
| Portfolio rollup | 2 FILTER_IS_THE_EDGE + 2 BOTH + 2 VESTIGIAL | 2 FILTER_CORRELATES + 2 BOTH + 1 VESTIGIAL + 1 UNTESTABLE |

### What stands from prior PRs

- PR #47 (filter vestigialness) L1/L3/L4/L5 fire-rate stats are unchanged.
  Its L2/L6 fire rates also happen to be unchanged (fire rate on ATR_P50
  and ORB_G5 depends on the daily-features data, not on orb_minutes).
  The PR #47 per-lane MECHANISM diagnoses for L2 are wrong (no
  "sparse-session instability") — noted above.
- PR #52 L1, L3, L4, L5 numbers and classifications. All unaffected
  by the bug.
- PR #54 methodology (per-year Welch, early/late, rolling 3y, KS) is
  preserved; only the data aperture was wrong. Same methodology
  re-applied at the correct aperture in Section B above.

---

## Methodology fixes applied (per code-reviewer findings)

| Finding | Fix |
|---------|-----|
| **H1** | `parse_strategy_id` imported from `trading_app.eligibility.builder` (canonical). orb_minutes passed into SQL. |
| **H2** | L6 primary verdict is one-sample t + power analysis. Bootstrap demoted to descriptive context with autocorrelation caveat. |
| **H3** | Label renamed FILTER_IS_THE_EDGE → FILTER_CORRELATES_WITH_EDGE. Soft language in MD: "day-selection correlates with edge; mechanism is vol-regime gate, not session-specific." |
| **M1** | L2 KS test: per-year (not pooled). 8 per-year KS results reported. |
| **M2** | L2 deploy-risk wording: "deploy-killed" → "deploy-not-selected"; t=−0.38 described as "zero-consistent" before observing it's actually +1.86 at correct aperture. |
| **M3** | IS/OOS split explicitly labeled as "trade-count median" with calendar date annotation. |
| **M4** | K=6 BH-FDR multiple-testing table added to Section A. |

---

## Operational conclusions

1. **Portfolio posture UNCHANGED.** All 6 lanes continue. No pauses.

2. **L2 is no longer flagged as fully filter-dependent.** It's a
   borderline-positive baseline with a filter that adds ~+0.06R of
   marginal edge. The "highest-priority filter audit" framing from
   PR #54 was driven by the aperture bug; downgrade L2 ATR_P50 to a
   "watch, don't urgently replace" status.

3. **L6 is no longer flagged as a 2026-negative problem lane.** It's
   currently +0.161R in 2026. The L6 diagnostic is closed with
   "working as expected" verdict.

4. **L1 EUROPE_FLOW ORB_G5 classification unchanged** —
   FILTER_CORRELATES_WITH_EDGE (baseline weak, filter correlates at
   Welch p=0.001). This one was correct before; only the label name
   softened.

5. **The MNQ broad-lane / broad-family scans (PRs #49, #50, #51) are
   not re-audited here** — they scan unfiltered geometry across
   many (session × aperture × RR) combos and may not carry the same
   bug. Worth a spot-check next session.

---

## Follow-ups

1. **Fix `research/audit_6lane_scale_stability.py` (PR #47).** It
   contains the same hardcoded `orb_minutes=5` bug. L2/L6 fire rates
   reported in PR #47 were computed at wrong aperture. The
   fire-rate-drift story may be partially or fully different at
   correct aperture. Separate correction PR.

2. **Fix `research/audit_6lane_unfiltered_baseline.py` (PR #52).**
   Source script has the bug. Either delete (superseded) or patch
   to delegate. Recommend deleting since this v2 is comprehensive.

3. **Fix `research/audit_l2_atr_p50_stability.py` (PR #54).** Source
   script has the bug. Delete (superseded).

4. **Sweep `research/` for any other scripts hardcoding
   `orb_minutes=5`.** If other audits quoted in MEMORY.md or HANDOFF.md
   used L2/L6 numbers from the wrong aperture, those findings need
   revisiting. Spot-check the PR #47 6-lane filter-vestigialness MD —
   its L2 and L6 rows are wrong.

5. **Update `memory/session_2026-04-20-claude-handoff.md`** and
   `memory/MEMORY.md` to reference this correction.

---

## Provenance

- Canonical data: `orb_outcomes` and `daily_features` (triple-joined
  on `trading_day`, `symbol`, `orb_minutes`).
- Canonical parser: `trading_app.eligibility.builder.parse_strategy_id`
  (IMPORTED, not re-encoded).
- Filter delegation: `research.filter_utils.filter_signal` → 
  `trading_app.config.ALL_FILTERS[key].matches_df`.
- Holdout: 2026-01-01 (Mode A sacred).
- Read-only. No production code touched.
- No deployment changes recommended by this PR.
