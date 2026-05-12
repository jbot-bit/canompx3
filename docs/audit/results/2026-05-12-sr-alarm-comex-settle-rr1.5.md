# SR alarm diagnosis — MNQ COMEX_SETTLE E2 RR1.5 CB1 OVNRNG_100

Date: 2026-05-12
Author: Claude Code session (feat/sr-alarm-diagnosis-2026-05-12)
Pre-reg: `docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml`
Lane: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`

## Scope

Capital-class diagnostic on the deployed lane
`MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` to identify whether its 2026-05-11
SR ALARM is consistent with (a) Harris mechanism still valid but variance
compressing, (b) a Harris-cited falsification trigger having fired, or (c)
Bailey-style sample-selection / deflated-Sharpe artifact catching up
post-deployment. Scope locked to this lane only; no writes to validated_setups,
sr_review_registry, or any deployed-state file.

## Verdict

**MECHANISM_FALSIFIED** — F3(b) FIRED: OVNRNG_100 fire rate jumped from 8% (2023)
and 58% (2025) to **92% (2026 partial)**, a >10× scale shift across 2 years.
The 100-point absolute threshold no longer discriminates because MNQ overnight
ranges have expanded materially. This is the canonical
`feedback_absolute_threshold_scale_audit.md` class bug.

Recommended action (informs separate stage, not executed here): pause via
`sr_review_registry`; recommend full revalidation with a relative-scale
overnight-range condition (e.g., `OVNRNG_PCTILE_75` instead of `OVNRNG_100`).

## Step 1 — SR peak-vs-current decomposition

Source: `data/state/sr_state.json` (snapshot 2026-05-11T22:26:39Z, git_head `398693ea`).

| Field | Value |
|---|---|
| `sr_stat` (peak) | 35.3897 |
| `current_sr_stat` | 2.877 |
| `threshold` | 31.96 |
| `n_monitored` | 66 |
| `alarm_trade` | 24 |
| `trades_since_alarm` | 42 |
| `recent_10_mean_r` | +0.4243 |
| `expected_r` (baseline) | 0.2151 |
| `std_r` (baseline) | 1.2518 |

**Peak-vs-current ratio:** `current_sr_stat / sr_stat_peak = 2.88 / 35.39 = 0.081`
— **PEAK_DECAYED** trigger fires.

`recent_10_mean_r = +0.42` is strongly positive — nearly 2× the long-run
expected_r (0.215). The lane's recent live performance is excellent. The peak
SR alarm fired at trade 24 then decayed; current_sr_stat is well below threshold.

F1 verdict: **PEAK_DECAYED** — `current_sr_stat / sr_stat_peak = 0.081 < 0.10`
AND `recent_10_mean_r > 0`. The alarm trigger no longer reflects current health.

## Step 2 — Rolling-60 trade decomposition

Source: `research/sr_alarm_decomposition_2026_05_12.py --pressure-test`.
Pressure test (RULE 13): PASS.

| Metric | Full history | IS (`<2026-01-01`) | OOS (`≥2026-01-01`) | Recent 60 |
|---|---|---|---|---|
| N | 598 | 529 | 69 | 60 |
| mean(pnl_r) | +0.2016 | +0.2085 | +0.1484 | +0.1676 |
| stdev(pnl_r) | 1.1647 | 1.1649 | 1.1708 | 1.1740 |
| win rate | 0.515 | — | — | 0.500 |
| fire-rate (full hist) | 0.345 | — | — | — |
| cadence (d/trade, recent) | — | — | — | 1.81 |

**Component flags (F2 kill criteria):**
- `variance`: **NORMAL** (recent_std/full_std = 1.008)
- `mean`: **WITHIN_BAND** (|recent − expected| = 0.048 R, ≤ 1·pooled_std)
- `win_rate`: **NORMAL** (recent_wr 0.500 ≥ threshold 0.360 = 0.7 × full_wr 0.515)

**Power floor (RULE 3.3):** Cohen's d = 0.027 → power 5.5% → **STATISTICALLY_USELESS**.
Recent-window vs history cannot be statistically distinguished. N for 80% power:
21,331 per group.

**Last 5 rolling-window snapshots:**

```
[2025-06-13 → 2025-12-16] N=60  mean=+0.216  std=1.188  wr=0.517  cad=3.2 d/t
[2025-08-18 → 2026-01-21] N=60  mean=+0.099  std=1.186  wr=0.467  cad=2.6 d/t
[2025-10-21 → 2026-02-12] N=60  mean=-0.006  std=1.166  wr=0.417  cad=1.9 d/t
[2025-11-17 → 2026-03-06] N=60  mean=+0.015  std=1.162  wr=0.433  cad=1.8 d/t
[2025-12-17 → 2026-03-27] N=60  mean=+0.040  std=1.156  wr=0.450  cad=1.7 d/t
```

The rolling windows show a **mean-decay trough** in late 2025 / early 2026
(min −0.006R), then recovery. Recent-60 (+0.168) is below long-run (+0.202)
but not below expectancy floor.

**Cadence shift:** 3.2 d/trade (mid-2025) → 1.7-1.8 d/trade (recent). Trade
frequency nearly DOUBLED in the recent window — consistent with the F3(b) fire
rate jumping from ~58% to ~92%. The lane is firing on more days than it did
in IS.

F2 verdict: **NORMAL** across components, BUT cadence-shift indicates the
filter selectivity is degrading.

## Step 3 — Harris falsification-trigger check

Mechanism per `mechanism_priors.md` § 2.5: Harris 2002 Ch 14 § 14.2
adverse-selection at the settlement-price re-mark + Ch 21 § 21.2 effective-spread
in low-depth windows.

### Trigger (a) — CME settlement-window methodology change

CME COMEX gold settlement window: 13:25–13:30 ET (canonical; unchanged across
the IS+OOS horizon). `pipeline.dst.SESSION_CATALOG['COMEX_SETTLE']` reflects
this. No methodology change since lane deployment 2026-05-10. **NOT_FIRED.**

### Trigger (b) — OVNRNG_100 fire-rate drift outside calibration band

| Year | Fires / Total | Fire rate |
|---|---|---|
| 2019 | 7/164 | **0.043** |
| 2020 | 104/249 | 0.418 |
| 2021 | 66/251 | 0.263 |
| 2022 | 116/250 | 0.464 |
| 2023 | 20/248 | 0.081 |
| 2024 | 73/249 | 0.293 |
| 2025 | 143/247 | 0.579 |
| 2026 | 69/75 | **0.920** |

Mean fire rate IS (2019-2025): 30.6%. Recent OOS (2026): 92%. Z-score of 92%
against IS distribution: well over 2σ.

The 100-point absolute overnight-range threshold was calibrated against a
universe where MNQ price was in a different scale band. As MNQ price has risen
from ~7,000 (2020) to >19,000 (2026), the same absolute 100-point threshold
selects a much larger fraction of overnight ranges. The filter has lost
selectivity.

**TRIGGER FIRED** — fire-rate drift far exceeds 2σ outside the IS calibration
band. Same class-bug as `feedback_absolute_threshold_scale_audit.md` documents.

### Trigger (c) — Per-(year, direction) sign-flip rate ≥ 25%

Per-year filtered ExpR (per `pooled-finding-rule.md` heterogeneity check):

| Year | N (filtered) | mean(pnl_r) | sign vs pooled |
|---|---|---|---|
| 2019 | 7 | -0.433 | (N<10, excluded) |
| 2020 | 104 | +0.261 | + |
| 2021 | 66 | +0.216 | + |
| 2022 | 116 | +0.005 | + (within ±0.02 noise) |
| 2023 | 20 | -0.082 | **−** (FLIP) |
| 2024 | 73 | +0.298 | + |
| 2025 | 143 | +0.358 | + |
| 2026 | 69 | +0.148 | + |

Pooled-finding flip rate: **1/7 = 14.3%** (excluding N<10 in 2019).

Below the 25% heterogeneity-ack threshold per `pooled-finding-rule.md`. **NOT_FIRED.**

F3 verdict: **MECHANISM_FALSIFIED** via TRIGGER (b) FIRED.

## Step 4 — Bailey deflated-Sharpe revisit

| Quantity | Value |
|---|---|
| Filtered T | 598 |
| mean(pnl_r) | +0.2016 |
| stdev(pnl_r) | 1.1647 |
| SR (non-ann) | +0.1731 |
| SR (annualised, ×√252) | +2.747 |
| skew | -0.042 |
| kurt (Pearson) | +1.027 |
| Scan cells N≥50 | 864 |
| V[ŜR_n] | 0.003202 |
| √V | 0.0566 |
| `n_trials_at_discovery` (validated_setups) | 36,372 |
| E[max] z @ N=36,372 | 4.1655 |
| SR_0 | 0.2357 |
| **LIVE_DSR** | **0.0637** |
| deployment_DSR (validated_setups.dsr_score) | 7.65e-12 |

LIVE_DSR (0.064) is dramatically higher than deployment-time DSR (≈0) —
the lane has accumulated more independent live evidence than at promotion.
Still below the 0.50 cross-check floor, but the trajectory is positive
(opposite of decay).

F4 verdict: **DSR_DECAYED (cross-check)** by absolute threshold (0.064 < 0.50)
but trajectory is improving. CROSS-CHECK only per Amendment 2.1.

## Internal consistency

- F1: **PEAK_DECAYED** — alarm trigger no longer reflects live health
- F2: NORMAL components, cadence shift indicates filter degradation
- F3: **MECHANISM_FALSIFIED** (fire-rate scale drift)
- F4: DSR_DECAYED (cross-check, but trajectory improving)

The verdict is dominated by F3 — the OVNRNG_100 filter has structurally lost
selectivity. F1 PEAK_DECAYED + F2 healthy components + F4 improving trajectory
ALL point to a HEALTHY lane on a BROKEN selector. The mechanism (overnight-range
gating of settlement-period adverse-selection) MAY still hold for the original
"selective" subset (high-relative-overnight-range days), but OVNRNG_100 no
longer selects for that subset.

## Action queue (informs separate stage)

1. Pause via `sr_review_registry` watch outcome (`MECHANISM_FALSIFIED`).
2. Recommend full revalidation with a **relative-scale** overnight-range
   condition: `OVNRNG_PCTILE_75` (top quartile of trailing-N overnight
   ranges) preserves the mechanism's selection intent across price scales.
3. Add explicit fire-rate bounds (`0.20 ≤ fire_rate ≤ 0.50`) to the SR
   monitor as an early-warning alarm — would have caught this in 2024-Q4.
4. Document this case as the second confirmed instance in
   `feedback_absolute_threshold_scale_audit.md`.

## Reproduction

- Pre-reg yaml: `docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml`
- Step 2 script: `research/sr_alarm_decomposition_2026_05_12.py --pressure-test
  --out research/output/sr_alarm_decomposition_2026_05_12.json`
- Steps 3/4/5 script (added 2026-05-12 post-review):
  `python research/sr_alarm_steps_3_4_5_2026_05_12.py` — reproduces the
  fire-rate-by-year, per-year sign-flip, live Bailey DSR, regime
  distribution shift, and cost-spec drift numbers cited in this MD. Reads
  lane list from the pre-reg yaml `scope.lanes[]`; delegates DSR math to
  `research.audit_ovnrng50_canonical_dsr.bailey_dsr` (top-level import safe
  after the same-commit `__main__`-guard refactor). Default args run all
  3 steps on all 3 lanes; `--steps 4` runs DSR only.
- DSR helper: `research.audit_ovnrng50_canonical_dsr.bailey_dsr` (Bailey 2014
  Eq 2; Bailey-example sanity check 0.9004 = paper)
- SR state snapshot: `data/state/sr_state.json` git_head `398693ea` (read-only)
- Lane metadata: `validated_setups` row promoted_at 2026-05-10 13:37:49+10:00
- Drift check: 125 / 125 PASS; pressure test PASS

## Limitations

- Power floor (RULE 3.3): recent-vs-history Welch t-test on rolling-60 returned
  **STATISTICALLY_USELESS** (Cohen's d 0.027 → power 5.5%). F2 component flags
  are descriptive, not refutational. The MECHANISM_FALSIFIED verdict rests on
  F3 (binary structural fire-rate scale-drift check), unaffected by power.
- F1 PEAK_DECAYED + F2 NORMAL + F4 trajectory-improving suggest the LANE is
  healthy — only the SELECTOR is broken. The Harris adverse-selection mechanism
  (Ch 14 § 14.2) at the COMEX settlement window MAY still hold for the original
  "selective" subset (high-relative-overnight-range days). A separate research
  task with `OVNRNG_PCTILE_75` would re-test that.
- LIVE_DSR (0.0637) is dramatically higher than deployment_DSR (≈0) — trajectory
  is improving, not decaying. F4 verdict label DSR_DECAYED is by absolute
  threshold (<0.50), not by trajectory.
- Per-(year, direction) sign-flip rate 1/7 (14.3%) is below the 25% heterogeneity
  threshold. The pooled "OVNRNG_100 has positive edge" claim is safe to quote;
  the 2023 negative year (N=20, mean −0.082) is a single underpowered cell.

## Out of scope

- Tweaking OVNRNG_100 threshold to 150 / 200 / etc (data-snooping per
  `feedback_bias_discipline.md`).
- Pausing the lane (separate stage gated by user approval).
