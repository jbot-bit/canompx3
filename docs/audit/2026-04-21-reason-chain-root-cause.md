# 2026-04-21 Shared-Reason-Pattern Root Cause

Scope: PCC-5 additive posture-clearing evidence for the Fork D orthogonal hunt.

Authority:
- Phase B helper on `origin/research/pr48-sizer-rule-oos-backtest`: `research/phase_b_live_lane_verdicts.py`
- Phase B lane verdict docs on `origin/research/pr48-sizer-rule-oos-backtest`
- Binding posture context: `docs/audit/2026-04-21-phase-b-institutional-reeval.md` on `origin/research/pr48-sizer-rule-oos-backtest`

## Question

Was the token-identical shared reason chain in five `DEGRADE` lane verdicts caused by:
- a code artifact that collapses lane-specific evidence into boilerplate, or
- a genuine common defect across the five lanes?

## Evidence

### 1. The generator does hard-code the reason ordering

From `research/phase_b_live_lane_verdicts.py` on `origin/research/pr48-sizer-rule-oos-backtest`:

```python
def _verdict(lane, matrix, g3_rows):
    reasons = []
    sr_state = str(matrix["sr_state"])
    if sr_state == "ALARM":
        reasons.append("Criterion 12 live SR monitor is ALARM in the binding Phase A snapshot.")
        return "PAUSE-PENDING-REVIEW", reasons
    if not bool(matrix["holdout_clean"]):
        reasons.append("Holdout integrity fails: discovery_date is after 2026-01-01 and OOS evidence is already populated.")
    if float(matrix["t"]) < 3.79:
        reasons.append("Chordia strict band fails and no local-literature theory band was verified for t>=3.00.")
    conservative = next(row for row in g3_rows if math.isclose(row["rho"], 0.7))
    if conservative["dsr"] <= 0.95:
        reasons.append("Bracketed DSR fails at the directive's conservative rho=0.7 bound.")
    if not reasons:
        return "KEEP", ["All Phase B gates clear on Phase A evidence."]
    return "DEGRADE", reasons
```

This proves the prose reason chain is generator-driven and ordered, not hand-authored per lane.

### 2. The lane docs do carry different underlying evidence

Examples from the committed Phase B lane verdict docs:

| Lane | `t` | `WFE` | `DSR` | `SR state` | `holdout` |
| --- | ---: | ---: | ---: | --- | --- |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `2.528` | `2.8551` | `0.0000000004` | `CONTINUE` | `FAIL` |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `3.717` | `2.6151` | `0.0000005135` | `CONTINUE` | `FAIL` |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `3.400` | `0.8225` | `0.0003150000` | `CONTINUE` | `FAIL` |

These metrics differ materially by lane. The shared text is not because the evidence tables are identical.

### 3. The common failure pattern is still real

Despite the prose compression, the same three gate failures genuinely held across the five non-alarmed lanes:
- `holdout_clean = FAIL`
- `t < 3.79`
- `DSR(rho=0.7) <= 0.95`

The shared reason chain therefore reflects a real common gate pattern, but only at the top-level gate result.

## Diagnosis

The root cause is **mixed**:

- **Code artifact in prose generation:** yes. `_verdict()` emits the same ordered English reasons whenever the same gate pattern occurs.
- **Shared truth-level defect:** also yes. The five non-alarmed lanes did actually share the same Phase B fail-closed gate pattern.

The right interpretation is:
- the identical prose is **not** independent corroboration of five separate narrative analyses
- but it is **not** proof of a broken gate stack by itself either

## Practical impact

- This does **not** overturn the Phase B posture result.
- It does mean later readers must inspect the per-lane gate tables and certificates, not just the copied rationale block.
- Any future Phase B rerun should separate:
  1. lane-specific evidence block
  2. shared gate outcome summary
  so common outcomes are visible without pretending the written rationale was independently derived five times.

## Verdict

`CODE-ARTIFACT-ON-PROSE + REAL-SHARED-GATE-PATTERN`

No patch is proposed here. This is diagnosis only.
