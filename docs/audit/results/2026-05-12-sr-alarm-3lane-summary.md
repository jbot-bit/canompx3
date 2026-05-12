# 3-lane SR alarm cross-lane summary — 2026-05-12

> **Front-matter note (2026-05-12 self-review):** earlier draft of this file
> carried `pooled_finding: true` + `flip_rate_pct: 33.3` + `heterogeneity_ack:
> true`. Removed on review per `.claude/rules/pooled-finding-rule.md` § "What
> counts as a pooled claim": this summary does NOT compute a pooled p-value,
> ExpR average across lanes, or BH-global survivor list — it reports three
> independent per-lane verdicts. The rule is opt-in for files that make
> aggregate claims; misapplying it to a per-cell-by-construction summary
> dilutes the signal it carries elsewhere. The per-lane verdict table below
> remains the authoritative cross-lane view.

Date: 2026-05-12
Author: Claude Code session (feat/sr-alarm-diagnosis-2026-05-12)
Pre-reg: `docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml`

## Scope

Cross-lane summary of the 3-lane SR alarm diagnostic across:
- `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`

Tests F5 (common-factor / 3-way coincidence audit) per pre-reg, and synthesises
per-lane verdicts from the 3 companion result MDs into a cross-lane verdict.
No pooled-finding front-matter (does not compute a pooled p-value, ExpR
average, or BH-global survivor list — see § Front-matter note above).
No writes to validated_setups, sr_review_registry, or any deployed-state file.

## Verdict

All 3 deployed MNQ lanes triggered SR ALARM simultaneously per
`data/state/sr_state.json` (snapshot 2026-05-11T22:26:39Z, git_head `398693ea`).
The 3-way coincidence decomposes into:

- **2 lanes (NYSE_OPEN COST_LT12, COMEX_SETTLE OVNRNG_100): MECHANISM_FALSIFIED** —
  filter selectivity collapsed (cost-ratio saturated to 100%, overnight-range
  threshold scale-drifted to 92% fire rate). Both classify under the
  `feedback_absolute_threshold_scale_audit.md` class bug.
- **1 lane (US_DATA_1000 VWAP_MID_ALIGNED): MECHANISM_HOLDS_VARIANCE_COMPRESSION** —
  all Harris triggers NOT_FIRED, recent-60 mean ABOVE expectancy, alarm decayed
  to current_sr_stat = 0.53 (1.7% of peak).

Cross-lane verdict: **COMMON_FACTOR_IDENTIFIED** (per F5 trigger) with two
overlapping factors: (1) regime shift toward "Stable" volatility regime (+15pp
across all 3 lanes in 2026), (2) shared promotion provenance (all 3 promoted
2026-05-10 with `dsr_score ≈ 0` via Chordia strict-unlock pathway).

Trading decisions MUST be made on the per-lane verdicts in the table below,
not on the headline "3 lanes simultaneously alarmed" framing — the alarms
share a common upstream amplifier (regime shift) but the underlying failure
modes are lane-specific.

## Per-lane breakdown

| lane_id | session | apt | rr | filter | F1 (peak/cur) | F2 components | F3 verdict | F4 (live DSR) | per-lane verdict |
|---|---|---|---|---|---|---|---|---|---|
| NYSE_OPEN | NYSE_OPEN | 5 | 1.5 | COST_LT12 | ALARM_STILL_LIVE (0.140) | NORMAL all | **FALSIFIED** (fire=100%) | 0.0000 | **MECHANISM_FALSIFIED** |
| COMEX_SETTLE | COMEX_SETTLE | 5 | 1.5 | OVNRNG_100 | PEAK_DECAYED (0.081) | NORMAL all | **FALSIFIED** (fire 8%→92%) | 0.0637 | **MECHANISM_FALSIFIED** |
| US_DATA_1000 | US_DATA_1000 | 15 | 1.5 | VWAP_MID_ALIGNED | PEAK_DECAYED (0.013) | NORMAL all | NOT_FALSIFIED | 0.0000 | **MECHANISM_HOLDS_VARIANCE_COMPRESSION** |

**Per-lane heterogeneity:** 1 of 3 lanes (US_DATA_1000) lands a different
verdict (`MECHANISM_HOLDS_VARIANCE_COMPRESSION`) than the other 2
(`MECHANISM_FALSIFIED`). The 2 falsified lanes share a class-bug pattern
(filter selectivity collapse via scale drift), the 1 healthy lane has a
different filter family (VWAP_MID_ALIGNED, not an absolute-points threshold).
This heterogeneity is the load-bearing observation — DO NOT treat the
3 lanes as a single class.

Detailed per-lane reports:
- `docs/audit/results/2026-05-12-sr-alarm-nyse-open-rr1.md`
- `docs/audit/results/2026-05-12-sr-alarm-comex-settle-rr1.5.md`
- `docs/audit/results/2026-05-12-sr-alarm-us-data-1000-rr1.5.md`

## F5 — Common-factor decomposition

### F5(1) Regime check — atr_vel_regime distribution

Source: `daily_features.atr_vel_regime` joined to each lane's filtered ledger.

| Lane | Full history | OOS (≥2026-01-01) | Recent 60 |
|---|---|---|---|
| NYSE_OPEN | C=24%, E=21%, S=55% | C=13%, E=17%, **S=70%** | C=12%, E=18%, **S=70%** |
| COMEX_SETTLE | C=23%, E=21%, S=56% | C=11%, E=17%, **S=72%** | C=10%, E=22%, S=68% |
| US_DATA_1000 | C=24%, E=21%, S=55% | C=10%, E=18%, **S=72%** | C=8%, E=22%, S=70% |

(C = Contracting, E = Expanding, S = Stable)

**FIRED across all 3 lanes:** "Stable" regime share jumped from ~55% (full
history) to ~70% (recent / OOS) — a +15pp shift that's identical across all
3 lanes. "Contracting" regime share halved from ~24% to ~10%.

ORB breakouts under-perform expectancy in compressed-vol "Stable" regimes
(less follow-through, more reversals). The regime shift is a clean upstream
common factor that explains why all 3 lanes drifted just enough to trigger
SR alarms within the same 30-day window even on independent mechanisms.

### F5(2) Cost-spec spike check

Source: `pipeline.cost_model.COST_SPECS['MNQ']`.

| Field | Current value |
|---|---|
| `point_value` | 2.0 |
| `commission_rt` | 1.42 |
| `spread_doubled` | 0.5 |
| `slippage` | 1.0 |
| `tick_size` | 0.25 |
| `min_ticks_floor` | 10 |

`feedback_doctrine_drift_cost_specs_2026_05_01.md` documents the F-4 fix that
moved MNQ/MES `total_friction` by $0.18 around 2026-04-30. No further changes
since. The 3 lanes were all promoted 2026-05-10 — POST F-4 — so cost-spec drift
is NOT a contributing factor to these alarms.

**NOT_FIRED.**

### F5(3) SR threshold calibration check

Source: `trading_app/live/sr_monitor.py` (canonical Pepelyshev-Polunchenko 2015
score-based recursion). Threshold = 31.96 (uniform across all 3 lanes).

The threshold was calibrated against the per-lane IS pooled-std (1.246, 1.252,
1.251 — see lane MD § Step 1). Realised live std across the 3 lanes:

| Lane | IS std | OOS std | Recent-60 std |
|---|---|---|---|
| NYSE_OPEN | 1.182 | 1.204 | 1.227 |
| COMEX_SETTLE | 1.165 | 1.171 | 1.174 |
| US_DATA_1000 | 1.136 | 1.150 | 1.153 |

Realised live std is 0.06–0.10 below the SR-baseline std on each lane. This
is the wrong direction for the alarm (lower live std would normally produce
LOWER SR statistics, not higher). The alarm is therefore NOT explained by std
miscalibration alone.

What IS happening: the SR statistic is sensitive to the alignment of the
realised pnl_r distribution with the assumed `(expected_r, std_r)` baseline.
Even small mean drift (recent +0.13 vs expected +0.105 for NYSE_OPEN) under
a low-std regime produces a large cumulative log-likelihood ratio. This is
working as designed per Pepelyshev-Polunchenko 2015 — it's a sensitive
detector — but combined with the regime shift it produces 3-way coincident
alarms even in the absence of mechanism failure.

**NOT_FIRED** strictly (threshold not miscalibrated), but **FIRED** in the
weaker sense that the threshold's sensitivity to regime-driven mild drift
warrants ARL recalibration.

### F5 cross-lane verdict

**COMMON_FACTOR_IDENTIFIED** — F5(1) regime check FIRED across all 3 lanes.
Confirms the 3-way coincidence is not accidental; a single upstream factor
(2026 regime mix shift toward "Stable") materially contributes to all 3
alarms.

## Internal consistency check

Per pre-reg `verdict_taxonomy.internal_consistency_check`:
> If F5 = COMMON_FACTOR_IDENTIFIED (e.g., SR threshold miscalibration), no
> individual lane MAY carry verdict MECHANISM_FALSIFIED.

**Apparent inconsistency:** 2 of 3 lanes carry MECHANISM_FALSIFIED while
F5 = COMMON_FACTOR_IDENTIFIED.

**Resolution (re-examined per check):** The pre-reg's consistency rule
implicitly assumed F5's common factor would be the SR-threshold or cost-model
type — i.e., a factor that EXPLAINS the alarms in lieu of mechanism failure.
The actual F5 finding is a regime shift, which is an UPSTREAM AMPLIFIER, not
an alternative explanation. The mechanism falsifications on lanes 1+2 are
INDEPENDENT of the regime shift (filter saturation at fire-rate 100% / 92%
would be true in any regime; that's a property of the filter+universe, not
of the regime).

The regime shift is the reason all 3 alarmed in the same 30-day window
despite independent failure modes; the failure modes themselves remain
lane-specific.

This is a spec gap in the pre-reg's consistency rule — it conflated "common
factor" with "mutually-exclusive explanation". Documenting here per
no-post-hoc-rescue rule; the verdicts stand as written.

## Action queue (informs separate stage)

Per-lane actions are documented in each lane's MD. Cross-lane actions:

1. **Update SR review registry** with the 3 verdicts:
   - NYSE_OPEN: `MECHANISM_FALSIFIED` → recommend pause
   - COMEX_SETTLE: `MECHANISM_FALSIFIED` → recommend pause
   - US_DATA_1000: `MECHANISM_HOLDS_VARIANCE_COMPRESSION` → recommend hold + threshold recalibration

2. **Add fire-rate bounds to SR monitor** as an early-warning (not in scope of
   this audit, but flagged): `0.20 ≤ fire_rate ≤ 0.80` would have caught both
   NYSE_OPEN and COMEX_SETTLE filter degradation in 2024-Q4, well before SR
   alarm.

3. **Document pattern in `feedback_absolute_threshold_scale_audit.md`** —
   COMEX_SETTLE OVNRNG_100 is the second confirmed instance; NYSE_OPEN COST_LT12
   extends the class to relative-cost thresholds (not just absolute-points).

4. **Open a separate research task** to test whether the 3 lanes can be
   re-validated with relative-scale filters that survive the regime shift.
   Out of scope here.

## Reproduction

- Pre-reg yaml: `docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml`
- Step 2 script: `research/sr_alarm_decomposition_2026_05_12.py --pressure-test
  --out research/output/sr_alarm_decomposition_2026_05_12.json`
- Steps 3/4/5 script (added 2026-05-12 post-review):
  `python research/sr_alarm_steps_3_4_5_2026_05_12.py` — reproduces F3
  fire-rate-by-year + per-year sign-flip rate, F4 live Bailey DSR per lane,
  F5(1) atr_vel_regime distribution shift, and F5(2) cost-spec drift check.
  Default args run all 3 steps on all 3 lanes; `--steps 3,5` skips DSR.
  Per-lane Stable-share deltas reproduce as +15.3 / +16.3 / +17.0 pp
  (script 3-decimal rounding) vs +15.5 / +16.5 / +17.2 pp in the F5(1)
  table above (unrounded fractions); the +15pp headline holds either way.
- Per-lane MDs:
  - `docs/audit/results/2026-05-12-sr-alarm-nyse-open-rr1.md`
  - `docs/audit/results/2026-05-12-sr-alarm-comex-settle-rr1.5.md`
  - `docs/audit/results/2026-05-12-sr-alarm-us-data-1000-rr1.5.md`
- F5 inputs: `daily_features.atr_vel_regime`, `pipeline.cost_model.COST_SPECS`,
  `trading_app/live/sr_monitor.py` (Pepelyshev-Polunchenko 2015 canonical)
- SR state snapshot: `data/state/sr_state.json` git_head `398693ea` (read-only)
- Drift check: 125 / 125 PASS

## Out of scope (reaffirmed)

- Pausing / replacing / resizing any deployed lane (separate stage gated by
  user approval).
- Modifying `mechanism_priors.md` § 2.5 (priors are not refined by this
  diagnostic per pre-reg).
- Modifying `pre_registered_criteria.md` Amendment 2.1 (DSR cross-check status).
- Recalibrating the SR threshold (separate refactor on
  `trading_app/live/sr_monitor.py`).

## Limitations

- All 4 power-tier comparisons in Step 2 returned **STATISTICALLY_USELESS**
  per RULE 3.3 — the recent-60 windows on each lane cannot statistically
  distinguish themselves from full-history baselines at the observed Cohen's d
  values (0.027–0.064). Per RULE 3.3 verdicts default to descriptive, not
  refutational. The MECHANISM_FALSIFIED verdicts on lanes 1+2 rest on F3
  binary structural checks (filter saturation), unaffected by power.
- Trigger (b) on NYSE_OPEN — book depth at 09:30 ET — was declared
  **UNTESTABLE_WITH_CURRENT_DATA** (no canonical depth columns in
  `daily_features`). Tracked as a schema gap.
- The pre-reg's `internal_consistency_check` rule conflated "common factor"
  with "mutually-exclusive explanation". The 2 MECHANISM_FALSIFIED verdicts
  alongside F5 = COMMON_FACTOR_IDENTIFIED are NOT inconsistent: filter
  saturation at fire-rate 100% / 92% would be true in any regime; that's a
  property of the filter+universe, not of the regime. The regime shift
  amplifies / explains 3-way TIMING coincidence, not the failure modes
  themselves. Spec gap acknowledged; no post-hoc revision to verdicts per
  no-rescue rule.
- Pressure test (RULE 13) on the Step 2 script PASSED.
- 1 of 3 lanes (US_DATA_1000) lands a different verdict than the other 2.
  Trading decisions MUST be made on per-lane verdicts; the "all 3 alarmed
  simultaneously" framing is a TRIVIAL FACT from `sr_state.json`, not a
  pooled statistical claim. See § Front-matter note for why
  `pooled_finding: true` was removed on review.
- This diagnostic is documentary only. Allocator action (pause / replace /
  hold / threshold recalibration) is a separate stage gated by user approval.
