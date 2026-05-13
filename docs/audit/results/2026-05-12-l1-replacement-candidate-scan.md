---
audit_type: candidate-surfacing scan
scope: L1 slot (MNQ NYSE_OPEN E2 RR1.5 5min) replacement candidates
mutates: nothing (read-only scan)
companion_to: docs/audit/results/2026-05-12-sr-alarm-3lane-summary.md + PR #271 (allocator pause)
date: 2026-05-12
verdict: NO_QUALIFIED_REPLACEMENT_CANDIDATE
---

# L1 Replacement Candidate Scan — 2026-05-12

## Why this scan exists

The 2026-05-12 SR-alarm diagnostic paused L1
(`MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`) for MECHANISM_FALSIFIED +
ALARM_STILL_LIVE. The diagnostic framework correctly *shrank* the system from 3
deployed lanes to 2. This scan is the offensive companion: **what should
replace L1, if anything?**

Without this companion, every diagnostic monotonically reduces lane count.
That is not sustainable for a 3/7-capacity profile (`topstep_50k_mnq_auto`).

## Scope and constraints (pre-reg-lite)

This is a **candidate-surfacing scan**, not a discovery scan, not a pre-reg.
No `lane_allocation.json` mutation. No `validated_setups` mutation. No
hypothesis registered.

| Criterion | Value |
|---|---|
| Instrument | MNQ |
| Session | NYSE_OPEN |
| Entry model | E2 |
| ORB minutes | 5 (L1 exact slot) |
| RR target | 1.5 (L1 exact slot) |
| Confirm bars | 1 (L1 exact slot) |
| Filter family | **EXCLUDE absolute-threshold filters** (`COST_LT*`, `OVNRNG_*`, `ORB_VOL_*`, `NO_FILTER`, raw-points cutoffs) — see `feedback_absolute_threshold_scale_audit.md` |
| Chordia status | Require `PASS_CHORDIA` or `PASS_PROTOCOL_A` |
| Family status | `WHITELISTED` or `ROBUST` (not `PURGED` / `FRAGILE`) |
| OOS gate | Carver C8 OOS pass with adequate power |
| Sort | metrics.expectancy_r descending |
| Top N | 5 (relax if zero qualify) |

Rationale for excluding absolute-threshold filter family: per
`feedback_absolute_threshold_scale_audit.md`, raw-points thresholds drift with
price; L1 (COST_LT12) was itself MECHANISM_FALSIFIED with 100% fire rate on
2026-05-12. Promoting another absolute-threshold sibling would re-litigate the
same failure class.

## Source corpora consulted (canonical layers only)

1. **786-row deployability batch**:
   `docs/audit/results/2026-05-11-mnq-all-active-deployability.json`
   (786 MNQ strategies with full canonical replay, Carver C8 OOS,
   current FDR, family status, runtime control). This is the corpus the
   2026-05-11 `mnq-profile-candidate-proposal.md` classified into the
   live-deployment taxonomy.

   **Note on plan reference**: the originating plan referenced "the 783-lane
   2026-05-10 batch". The actual canonical corpus is 786 strategies dated
   `2026-05-11` (`mnq-all-active-deployability.json`). The 783 figure does not
   appear in the canonical batch JSON — likely a transcription error in the
   plan. Scan proceeded on the 786 batch.

2. **Allocator lane state**: `docs/runtime/lane_allocation.json` (rebalance
   2026-05-11, 2 lanes + 52 paused + 0 stale, 59 all_scores) — for current
   live status, Chordia verdict, allocator-side fitness.

3. **gold-db `strategy_lookup`** template — for `validated_setups` ranking by
   ExpR (used for non-allocator candidate enumeration).

`bars_1m` / `orb_outcomes` not queried directly — the deployability JSON
already carries canonical replay against orb_outcomes.

## L1-slot candidates from the 786-batch (all 18 rows)

Sorted by `metrics.expectancy_r` descending:

| # | Strategy | ExpR | N | Family | Verdict | Deployable | Filter class |
|---|---|--:|--:|---|---|---|---|
| 1 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70` | 0.197 | 413 | ROBUST | `BLOCKED_OOS_UNDERPOWERED` | FALSE | cross-asset percentile |
| 2 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | 0.132 | 673 | PURGED | `BLOCKED_FAMILY_FRAGILE` | FALSE | cross-asset percentile |
| 3 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K` | 0.116 | 1326 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |
| 4 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_50` | 0.115 | 1185 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |
| 5 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_25` | 0.112 | 1465 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |
| 6 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K` | 0.108 | 1439 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |
| 7 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4` | 0.108 | 1487 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | ORB-size grade (sibling) |
| 8 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER` | 0.107 | 1488 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded — `NO_FILTER` is the unfiltered baseline) |
| 9 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | 0.107 | 1485 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | ORB-size grade (sibling) |
| 10 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G6` | 0.106 | 1484 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | ORB-size grade (sibling) |
| 11 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G8` | 0.106 | 1479 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | ORB-size grade (sibling) |
| 12 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` | 0.105 | 1472 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | **L1 ITSELF — just paused** |
| 13 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT15` | 0.105 | 1478 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |
| 14 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08` | 0.104 | 1459 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |
| 15 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K` | 0.103 | 1476 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |
| 16 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ATR_P30` | 0.103 | 1060 | PURGED | `BLOCKED_FAMILY_FRAGILE` | FALSE | percentile |
| 17 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT10` | 0.102 | 1466 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |
| 18 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K` | 0.102 | 1464 | WHITELISTED | `CONTROLLED_LIVE_PILOT_CANDIDATE` | TRUE | absolute-threshold (excluded) |

## Filter gate

After excluding absolute-threshold and ORB-size-grade siblings of L1
(items 3-15, 17, 18) and item 12 (L1 itself), four rows remain (1, 2, 7-11
ORB_G* family is a sibling-class — included for completeness but commented):

| # | Strategy | Verdict | Block reason |
|---|---|---|---|
| 1 | `X_MGC_ATR70` | `BLOCKED_OOS_UNDERPOWERED` | `c8_oos_status: NO_OOS_DATA`, `replay_expectancy_drift` (stored 0.197 vs recomputed 0.179), `short_history: 4 yr`, `dsr_score: 3.4e-11` |
| 2 | `X_MES_ATR60` | `BLOCKED_FAMILY_FRAGILE` | `family_status: PURGED` |
| 16 | `ATR_P30` | `BLOCKED_FAMILY_FRAGILE` | `family_status: PURGED` |

Cross-asset percentile filters (`X_MGC_ATR70`, `X_MES_ATR60`) and intra-asset
percentile filters (`ATR_P30`) are the only non-absolute-threshold filter
family in the L1-slot. **All three fail the deployment gate.**

ORB-size-grade family (`ORB_G4` / `G5` / `G6` / `G8`) qualifies as
`CONTROLLED_LIVE_PILOT_CANDIDATE` and passes Carver C8, but it is the SAME
filter-class behavior as L1's COST_LT12 (a no-op gate that doesn't filter
trades) — all four pass ~all trades and merely re-index the same edge under
a different filter name. Promoting a sibling without addressing the
mechanism-falsified verdict on L1 would be re-litigation, not replacement.

## Cross-check: Chordia status in current allocator corpus

Only **3 of 54** lanes in the current allocator corpus (2 deployed + 52 paused)
carry a non-`MISSING` Chordia verdict:

| Lane | Status | Chordia | annual_r |
|---|---|---|--:|
| L2 `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | DEPLOY | `PASS_CHORDIA` | 36.20 |
| L3 `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | DEPLOY | `PASS_CHORDIA` | 27.10 |
| L1 `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` | PAUSE | `PASS_PROTOCOL_A` | 25.77 |

The full Chordia-audited universe is 3 strategies, all currently
deployed-or-just-paused. Every other lane in the 52-row paused corpus and
every L1-slot candidate from the 786-batch carries `chordia_verdict: MISSING`
or none-on-record. A new strategy CANNOT be considered for the L1 slot
without first running a Chordia audit on it (separate pre-reg).

## Verdict

**`NO_QUALIFIED_REPLACEMENT_CANDIDATE`**

The Chordia gate plus the filter-family exclusion plus the Carver C8 OOS gate
intersect to zero in the L1 slot. No clean replacement exists in the canonical
corpus today.

## Decision menu for user

| Option | Action | Cost | Risk |
|---|---|---|---|
| (a) | **Skip — accept 2 lanes until next rebalance** | 0 | continued lane-count erosion; allocator at 2/7 capacity not 3/7 |
| (b) | Run Chordia audit on `X_MGC_ATR70` + cure `c8_oos_status: NO_OOS_DATA` (likely requires OOS data window extension or sample shrinkage exemption pre-reg) | medium — separate audit workstream | mechanism is cross-asset percentile (Harris-grounded); strongest in-slot ExpR (0.197) but smallest N (413) and shortest history (4yr) |
| (c) | Widen scope to non-NYSE_OPEN sessions with non-absolute-threshold filter, then run Chordia + C8 on the top candidate | medium-high — adjacency scan + audit | breaks slot-exact constraint; may improve diversification |
| (d) | Promote an absolute-threshold sibling (e.g., `ORB_VOL_16K` ExpR=0.116) via `--bootstrap-runtime-control` | low effort, **HIGH** governance cost | re-litigates the COST_LT12 mechanism-falsification failure class; violates `feedback_absolute_threshold_scale_audit.md`; same governance gap as `feedback_bootstrap_disclosure_not_separation_of_duties.md`. **NOT RECOMMENDED.** |

Recommended: **(a) skip + plan (b) as a follow-up**. (a) preserves capital;
(b) is the disciplined offensive move with the strongest mechanism prior. (d)
is the path to repeating the 2026-05-12 incident.

## Out of scope for this scan

- L2 replacement (`MNQ_COMEX_SETTLE` slot, MECHANISM_FALSIFIED but DEPLOY) —
  defer until user decides on L1 slot.
- Cross-session adjacency scan — covered by option (c) above; would require
  cross-session pre-reg per `feedback_default_cross_session_scope.md`.
- Reconciling the 783 vs 786 batch-size discrepancy in the originating plan —
  flagged for HANDOFF.

## Sources

- `docs/audit/results/2026-05-11-mnq-all-active-deployability.json` (canonical replay corpus)
- `docs/audit/results/2026-05-11-mnq-profile-candidate-proposal.md` (profile-construction taxonomy classification)
- `docs/runtime/lane_allocation.json` (rebalance 2026-05-11, current allocator state)
- `gold-db` MCP `strategy_lookup` template (validated_setups ranking)
- `gold-db` MCP `get_strategy_fitness` (FIT/WATCH/DECAY for X_MGC_ATR70 and X_MES_ATR60 — both `fitness_status: FIT`)
- `pipeline/check_drift.py` — drift status during scan
- `feedback_absolute_threshold_scale_audit.md` (filter-family exclusion rationale)
- `feedback_bootstrap_disclosure_not_separation_of_duties.md` (option d governance objection)
- `feedback_provisional_not_paused_rr_variant_drift.md` (cautions against RR-variant promotion within paused family)

## Verification

- `lane_allocation.json` unchanged: `git diff origin/main -- docs/runtime/lane_allocation.json` shows only commit `d90682eb` pause edit (cherry-picked as commit `2c276c9e` on `feat/allocator-pause-l1-2026-05-12`, PR #271).
- `validated_setups` unchanged: scan was read-only.
- No pre-reg invoked: no `docs/audit/hypotheses/` write.

## Reproduction

To reproduce the scan output:

```bash
# 1. L1-slot candidates from 786-batch with verdicts (Python)
python -c "
import json
with open('docs/audit/results/2026-05-11-mnq-all-active-deployability.json') as f:
    data = json.load(f)
slot = [s for s in data['strategies']
        if 'NYSE_OPEN' in s['strategy_id'] and 'RR1.5' in s['strategy_id']
        and '_O15' not in s['strategy_id'] and '_O30' not in s['strategy_id']
        and '_E2_' in s['strategy_id']]
slot.sort(key=lambda s: s.get('metrics',{}).get('expectancy_r',0) or 0, reverse=True)
for s in slot:
    m = s.get('metrics', {})
    print(f\"{s['strategy_id']:<60} ExpR={m.get('expectancy_r',0):.4f} N={m.get('sample_size',0):5} fam={m.get('family_status','?'):<12} verdict={s.get('verdict')}\")
"

# 2. Chordia-audited universe from lane_allocation.json
python -c "
import json
with open('docs/runtime/lane_allocation.json') as f:
    data = json.load(f)
passed = [l for l in data.get('paused',[]) + data.get('lanes',[])
          if l.get('chordia_verdict') in ('PASS_CHORDIA','PASS_PROTOCOL_A')]
for l in sorted(passed, key=lambda x: x.get('annual_r',0) or 0, reverse=True):
    print(f\"{l['strategy_id']:<60} {l.get('status','?'):<8} {l.get('chordia_verdict','?'):<18} annual_r={l.get('annual_r',0)}\")
"

# 3. Fitness status check (via gold-db MCP get_strategy_fitness)
#    Inputs used: MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70, MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60
#    Both returned fitness_status=FIT at scan time (2026-05-12).
```

Outputs are reproduced inline in the **L1-slot candidates** and
**Cross-check: Chordia status** tables above.

## Caveats and limitations

- **Frozen-time snapshot.** The 786-batch JSON (2026-05-11) is one
  day stale relative to `lane_allocation.json` (rebalance 2026-05-11). Any
  re-rebalance between 2026-05-12 and decision time invalidates the Chordia
  cross-check table; re-run reproduction step 2 before acting.
- **Selection-bias risk on candidate ranking.** Sorting 18 L1-slot rows by
  ExpR and presenting top non-absolute-threshold candidates is a within-slot
  pick. Per `feedback_per_lane_breakdown_required.md`, single-slot ranking
  does not establish a universal pattern; the top ExpR row may simply be the
  one most exposed to selection variance (X_MGC_ATR70 N=413 vs the
  WHITELISTED N=1480 cohort).
- **`X_MGC_ATR70` replay drift not investigated.** The deployability JSON
  flags `replay_expectancy_drift` (stored 0.197 vs recomputed 0.179 on
  trade-day count 500 vs validated_setups N=413). Whether this is a window
  difference, a sample-count semantic discrepancy, or a structural drift
  was NOT investigated. Option (b) Chordia audit on this strategy must
  resolve the drift first.
- **Plan reference correction not back-propagated.** The originating plan
  cites a "783-lane 2026-05-10 batch"; the canonical batch is 786 strategies
  dated 2026-05-11. This document uses the canonical figure but does not
  amend the originating plan.
- **No bootstrap-test of the no-candidate verdict.** The structural finding
  (3 Chordia-audited lanes total) is mechanical, not statistical; no
  paired-p, no Bailey deflation. The verdict NO_QUALIFIED_REPLACEMENT
  describes the corpus, not the universe.
- **Option (b) feasibility is asserted, not verified.** Whether
  `X_MGC_ATR70` can clear Carver C8 with an extended OOS window or
  sample-shrinkage pre-reg is an open question. Option (b) is the strongest
  next-action candidate, not a guaranteed promotion path.
- **Option (c) cross-session adjacency is unscoped.** "Same session, any
  RR/aperture" fallback from the originating plan is named but not
  enumerated; a separate cross-session scan would be required.

## Disconfirming evidence the scan would have surfaced (and did NOT)

- A WHITELISTED non-absolute-threshold L1-slot lane with PASS Chordia and
  passing C8 OOS — none exists in the 786-batch.
- A PROVISIONAL lane in `lane_allocation.json` matching the L1 slot — none
  exists post-L1-pause; the L1 slot is empty.
- An OOS-passing X_MGC_ATR70 cohort — `c8_oos_status: NO_OOS_DATA` (hard
  block).
