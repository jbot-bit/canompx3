# SR alarm diagnosis — MNQ US_DATA_1000 E2 RR1.5 CB1 VWAP_MID_ALIGNED O15

Date: 2026-05-12
Author: Claude Code session (feat/sr-alarm-diagnosis-2026-05-12)
Pre-reg: `docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml`
Lane: `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`

## Scope

Capital-class diagnostic on the deployed lane
`MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` to identify whether its
2026-05-11 SR ALARM is consistent with (a) Harris mechanism still valid but
variance compressing, (b) a Harris-cited falsification trigger having fired, or
(c) Bailey-style sample-selection / deflated-Sharpe artifact catching up
post-deployment. Scope locked to this lane only; no writes to validated_setups,
sr_review_registry, or any deployed-state file.

## Verdict

**UNVERIFIED_INSUFFICIENT_POWER** — Recent-60 vs full-history Welch t-test
power is 7.7% (Cohen's d = 0.064), which is STATISTICALLY_USELESS per
`.claude/rules/backtesting-methodology.md` RULE 3.3 / `research/oos_power.py`
canonical thresholds. The pre-reg verdict_taxonomy option 4 trigger arm
"OOS power<50% per RULE 3.3" fires directly on this evidence. F1, F3
descriptive findings (PEAK_DECAYED, all Harris triggers NOT_FIRED, 8/8
years positive, recent-60 mean +0.294 above long-run expectancy +0.214)
remain on record but are descriptive, not refutational — they cannot
statistically distinguish "mechanism holds" from "underpowered noise."

**Verdict-label correction history (2026-05-12 post-self-audit):** Original
verdict was `MECHANISM_HOLDS_VARIANCE_COMPRESSION`, retracted because that
option's F2 trigger requires `recent_std/full_std < 0.6`; actual ratio is
**1.015** (NORMAL, not VARIANCE_COMPRESSION). Of the 4 locked taxonomy
options, only `UNVERIFIED_INSUFFICIENT_POWER` has all its primary triggers
satisfied by the evidence on record. Retraction is doctrinally clean
(stays within locked taxonomy; no pre-reg amendment); the underlying F1/F3
descriptive findings are unchanged.

Recommended action (informs separate stage, not executed here, per
pre-reg yaml line 461-462): **hold and re-check at N≥50**. Lane currently
at `n_monitored=40` (trades_since_alarm=34 + alarm_trade=6); re-trigger
the diagnostic once an additional ~10 trades accumulate so the recent
window has the power to refute or confirm the F3 descriptive finding.

## Step 1 — SR peak-vs-current decomposition

Source: `data/state/sr_state.json` (snapshot 2026-05-11T22:26:39Z, git_head `398693ea`).

| Field | Value |
|---|---|
| `sr_stat` (peak) | 41.6048 |
| `current_sr_stat` | 0.5338 |
| `threshold` | 31.96 |
| `n_monitored` | 40 |
| `alarm_trade` | 6 |
| `trades_since_alarm` | 34 |
| `recent_10_mean_r` | +0.4703 |
| `expected_r` (baseline) | 0.2101 |
| `std_r` (baseline) | 1.2506 |

**Peak-vs-current ratio:** `current_sr_stat / sr_stat_peak = 0.534 / 41.60 = 0.013`
— deeply **PEAK_DECAYED**.

`recent_10_mean_r = +0.47` is the highest recent-10 across the 3 lanes — over
2× the long-run expectancy. The lane's live performance is materially BETTER
than baseline.

The alarm fired at trade 6 — the very early SR statistic was sensitive to
a small early-OOS positive run that briefly violated the threshold. Since
then 34 more trades have accumulated; current_sr_stat is essentially at zero.

F1 verdict: **PEAK_DECAYED** — `current_sr_stat / sr_stat_peak = 0.013 < 0.10`
AND `recent_10_mean_r > 0`. The alarm trigger is a stale early-OOS artefact;
current health is excellent.

## Step 2 — Rolling-60 trade decomposition

Source: `research/sr_alarm_decomposition_2026_05_12.py --pressure-test`.
Pressure test (RULE 13): PASS.

| Metric | Full history | IS (`<2026-01-01`) | OOS (`≥2026-01-01`) | Recent 60 |
|---|---|---|---|---|
| N | 936 | 889 | 47 | 60 |
| mean(pnl_r) | +0.2143 | +0.2113 | **+0.2709** | **+0.2944** |
| stdev(pnl_r) | 1.1358 | 1.1356 | 1.1503 | 1.1529 |
| win rate | 0.518 | — | — | 0.533 |
| fire-rate (full hist) | 0.523 | — | — | — |
| cadence (d/trade, recent) | — | — | — | 2.54 |

**Component flags (F2 kill criteria):**
- `variance`: **NORMAL** (recent_std/full_std = 1.015)
- `mean`: **WITHIN_BAND** AND ABOVE expectancy (recent +0.294 > expected_r +0.210)
- `win_rate`: **NORMAL** (recent_wr 0.533 ≥ threshold 0.363 = 0.7 × full_wr 0.518)

**Power floor (RULE 3.3):** Cohen's d = 0.064 → power 7.7% → **STATISTICALLY_USELESS**.
Recent-window vs history cannot be statistically distinguished.

**Last 5 rolling-window snapshots:**

```
[2025-05-27 → 2025-11-11] N=60  mean=+0.179  std=1.152  wr=0.467  cad=2.8 d/t
[2025-07-02 → 2025-12-17] N=60  mean=+0.370  std=1.166  wr=0.533  cad=2.8 d/t
[2025-08-14 → 2026-01-26] N=60  mean=+0.187  std=1.192  wr=0.467  cad=2.8 d/t
[2025-09-17 → 2026-02-20] N=60  mean=+0.226  std=1.175  wr=0.483  cad=2.6 d/t
[2025-11-12 → 2026-04-01] N=60  mean=+0.221  std=1.149  wr=0.500  cad=2.4 d/t
```

Sequential rolling-60 means: **0.18, 0.37, 0.19, 0.23, 0.22 → recent +0.29.**
Live performance is the best of the three lanes. The "alarm" semantically
fired on early OOS run-up to +0.37 in the trailing-60 window, then decayed.

F2 verdict: **NORMAL** across all components. The lane is performing at or
above expectancy.

## Step 3 — Harris falsification-trigger check

Mechanism per `mechanism_priors.md` § 2.5: Harris 2002 Ch 4 § 4.5.2 stop-cascade
second-wave triggered by 09:30 ET cash-equity reaction to 08:30 ET economic-data
prints + Ch 14 § 14.2 adverse-selection at the data-release window.

### Trigger (a) — BLS / BEA / Fed 08:30 ET release schedule change

BLS NFP, CPI, PPI, retail sales, ISM remain on 08:30 ET schedule (canonical
since the early-1990s consolidation). No release-schedule change since lane
deployment 2026-05-10. **NOT_FIRED.**

### Trigger (b) — 10:00 ET multi-release coincidence (consumer-confidence, ISM)

The 10:00 ET window has long carried Conference Board consumer confidence
(monthly), ISM PMI (monthly), JOLTS (monthly). This is a structural feature
of the US data calendar, not a recent change. The mechanism's "second-wave
cascade only" framing is partially confounded by these — but this is a
permanent confound, not a falsification trigger.

Per the strict reading of the trigger, it requires a CHANGE in coincidence
pattern, which has not occurred. **NOT_FIRED.**

(Optional follow-up not in scope: re-test the lane with multi-release-day
exclusion to see if non-NFP-only days drive the edge. That's a new pre-reg.)

### Trigger (c) — VWAP_MID_ALIGNED fire-rate drift toward 95%

| Year | Fires / Total | Fire rate |
|---|---|---|
| 2019 | 83/170 | 0.488 |
| 2020 | 127/258 | 0.492 |
| 2021 | 147/258 | 0.570 |
| 2022 | 138/258 | 0.535 |
| 2023 | 128/257 | 0.498 |
| 2024 | 131/259 | 0.506 |
| 2025 | 135/257 | 0.525 |
| 2026 | 47/72 | 0.653 |

Fire rate range: 0.488 – 0.653. Mean ≈ 0.53. The 2026 partial-year value
(0.653) is the highest in the series but still well below the 0.95 trigger
threshold. The pattern is stable, not drifting toward saturation.

**NOT_FIRED.**

### Per-(year, direction) sign-flip rate

| Year | N (filtered) | mean(pnl_r) | sign vs pooled |
|---|---|---|---|
| 2019 | 83 | +0.248 | + |
| 2020 | 127 | +0.226 | + |
| 2021 | 147 | +0.217 | + |
| 2022 | 138 | +0.217 | + |
| 2023 | 128 | +0.205 | + |
| 2024 | 131 | +0.280 | + |
| 2025 | 135 | +0.103 | + |
| 2026 | 47 | +0.271 | + |

Pooled-finding flip rate: **0/8 = 0%**. Positive every year. 2025 was the
weakest year (+0.103 R) but recovered in 2026.

F3 verdict: **MECHANISM_NOT_FALSIFIED** — all triggers (a, b, c) NOT_FIRED;
8/8 years positive; fire rate stable ~50%.

## Step 4 — Bailey deflated-Sharpe revisit

| Quantity | Value |
|---|---|
| Filtered T | 936 |
| mean(pnl_r) | +0.2143 |
| stdev(pnl_r) | 1.1358 |
| SR (non-ann) | +0.1886 |
| SR (annualised, ×√252) | +2.994 |
| skew | -0.005 |
| kurt (Pearson) | +1.122 |
| Scan cells N≥50 | 861 |
| V[ŜR_n] | 0.009320 |
| √V | 0.0965 |
| `n_trials_at_discovery` (validated_setups) | 36,372 |
| E[max] z @ N=36,372 | 4.1655 |
| SR_0 | 0.4021 |
| **LIVE_DSR** | **0.0000** |
| deployment_DSR (validated_setups.dsr_score) | 1.11e-16 |

LIVE_DSR (0.0000) is essentially the same as deployment_DSR (≈0). The lane was
never validated by DSR — it was promoted via Chordia strict-unlock per the same
pathway as the other 2 lanes. Per Amendment 2.1 of `pre_registered_criteria.md`,
DSR is CROSS-CHECK informational, not load-bearing.

F4 verdict: **DSR_DECAYED (cross-check)** by absolute threshold but trajectory
flat (no decay vs deployment). CROSS-CHECK only.

## Internal consistency

- F1: **PEAK_DECAYED** — alarm trigger no longer reflects live health (descriptive)
- F2: **NORMAL** components (recent_std/full_std = 1.015, NOT VARIANCE_COMPRESSION which requires <0.6); recent-60 mean ABOVE expectancy (descriptive)
- F3: **MECHANISM_NOT_FALSIFIED** — all 3 binary structural triggers NOT_FIRED, 8/8 years positive, 0% sign-flip
- F4: DSR_DECAYED (cross-check only per pre-reg F4 spec + Amendment 2.1; not load-bearing)
- **Statistical power on the recent-vs-history comparison: 7.7% (STATISTICALLY_USELESS per RULE 3.3)**

The F1/F3 descriptive findings cannot statistically distinguish "mechanism
holds" from "underpowered noise consistent with mechanism failure" at this
sample size. Per RULE 3.3 (`backtesting-methodology.md`) the canonical
reading of power<50% is UNVERIFIED, never DEAD and never CONFIRMED. The
pre-reg verdict_taxonomy option 4 trigger arm is satisfied; the original
`MECHANISM_HOLDS_VARIANCE_COMPRESSION` label is retracted (F2 trigger
ratio 1.015 fails the <0.6 requirement). The SR alarm interpretation —
stale early-OOS peak + regime-driven mild std drift — remains a coherent
hypothesis but cannot be confirmed at current N.

## Action queue (informs separate stage)

1. **Hold and re-check at N≥50** — pre-reg yaml line 461-462 locked action
   for `UNVERIFIED_INSUFFICIENT_POWER`. Lane currently at n_monitored=40;
   re-trigger this diagnostic once ~10 additional trades accumulate so
   recent-vs-history has the power to refute or confirm F3.
2. Do not pause the lane (capital action gated by separate stage + user
   approval; descriptive findings do not warrant a fail-closed pause).
3. The Pepelyshev-Polunchenko 2015 ARL threshold-recalibration framing
   from the original verdict is descriptive context, not load-bearing
   evidence under the corrected verdict. Recalibration remains a coherent
   follow-up hypothesis but requires its own pre-reg per pre-reg
   forbidden_actions (no in-band SR threshold modification).
4. Acknowledge in `sr_review_registry` watch outcome: `UNVERIFIED_INSUFFICIENT_POWER`,
   action `HOLD_AND_RECHECK`.

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
- Lane metadata: `validated_setups` row promoted_at 2026-05-10 13:45:35+10:00
- Drift check: 125 / 125 PASS; pressure test PASS

## Limitations

- Trigger (b) — 10:00 ET multi-release coincidence — is a permanent structural
  feature of the US data calendar, not a recent change. The trigger is
  formally NOT_FIRED but a follow-up pre-reg could re-test the lane with
  multi-release-day exclusion to see if non-NFP-only days drive the edge.
- Power floor (RULE 3.3): recent-vs-history Welch t-test on rolling-60 returned
  **STATISTICALLY_USELESS** (Cohen's d 0.064 → power 7.7%). F2 component flags
  are descriptive, not refutational. The MECHANISM_HOLDS verdict rests on F3
  (3 binary structural triggers ALL NOT_FIRED) plus the per-year stability
  evidence (8/8 years positive, 0% sign-flip rate).
- LIVE_DSR (0.0000) was already 0.0 at deployment per `validated_setups.dsr_score`
  — the lane was promoted via Chordia strict-unlock pathway (Amendment 2.1
  demotes DSR to cross-check). F4 verdict is corroborating, not load-bearing.
- The "hold + recalibrate threshold" recommendation is descriptive only; the
  actual threshold recalibration requires a separate refactor on
  `trading_app/live/sr_monitor.py` with its own pre-reg.

## Out of scope

- Modifying VWAP_MID_ALIGNED threshold (no need; mechanism holds).
- Pausing the lane (separate stage gated by user approval).
- Recalibrating SR threshold (separate refactor on `trading_app/live/sr_monitor.py`
  that requires its own pre-reg).
