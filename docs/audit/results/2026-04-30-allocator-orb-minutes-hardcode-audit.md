# Allocator orb_minutes Hardcode — Audit & Structural Fix

**Date:** 2026-04-30 (audit) / 2026-05-01 (fix landed)
**Severity:** CAPITAL-CLASS BUG (UNSAFE → fixed)
**Triggering review:** Target A in queued bias-defense audit list (live deploy lane EV math)
**Audit method:** Fresh-context evidence-auditor agent (`a1e76860ea5635815` + continuation `ab19fcc9e39814af2`)

## Question / Scope

Does `trading_app/lane_allocator.py` correctly score the strategies it loads
from `validated_setups`? Specifically: do per-strategy queries and ranking
inputs respect the strategy's actual `orb_minutes` aperture, or is the
allocator silently substituting a different aperture and producing
wrong-but-plausible numbers for the live `topstep_50k_mnq_auto` profile?

Capital-risk frame: 6 lanes were DEPLOY in the live JSON; 2 of them are O15.
If the allocator misreads aperture, those lanes' ExpR / DD budget / months_neg
are unreliable, which directly affects sizing decisions on real capital.

## Verdict / Decision

**REFUTED:** the allocator does NOT correctly score O15 strategies. Bug
confirmed at 5 file:line sites (`trading_app/lane_allocator.py` lines 116,
131, 388, 525, 786). Two O15 DEPLOY lanes were scored against O5 data; DD
budget understated 54-59%. Capital-class bug.

**Decision:** ship the structural fix (this PR); land interim mitigation
(PR #188) immediately to pause the 2 O15 lanes while the structural fix
is under review. Kept-hardcode in `_compute_session_regime` (line 388) is
preserved by design — the regime gate is a session-level signal, not
strategy-aperture-specific; rationale comment lives in the code.

## TL;DR

`trading_app/lane_allocator.py` hardcoded `orb_minutes = 5` at every per-strategy query
site. The two O15 strategies in the live `topstep_50k_mnq_auto` profile
(`MNQ_SINGAPORE_OPEN_*_O15`, `MNQ_US_DATA_1000_*_O15`) were therefore scored against
5-minute aperture data instead of their validated 15-minute aperture. Trailing ExpR,
trailing_n, months_negative, and DD-budget P90 ORB sizes were all wrong for these two
lanes. Material capital-safety exposure: DD budgets understated 54-59% on O15 lanes.

## Bug Sites (lane_allocator.py)

| Line | Site | Effect |
|------|------|--------|
| 116 | `_per_month_expr` outcomes query | trailing_expr, trailing_n, months_negative computed from O5 outcomes for O15 strategies |
| 131 | `_per_month_expr` daily_features query | filter eligibility computed from O5 features for O15 strategies |
| 388 | `_compute_session_regime` | regime gate uses O5 (intentional — see "Kept Hardcodes" below) |
| 525 | `compute_pairwise_correlation` | correlation matrix used O5 P&L series for O15 strategies |
| 786 | `compute_orb_size_stats` | per-session ORB size used O5; DD budget uses these values |

## Fix

Threaded the strategy's actual `orb_minutes` from `validated_setups` through every
per-strategy code path:

- Added `orb_minutes: int` field to `LaneScore` dataclass
- `_per_month_expr` now takes `orb_minutes` parameter; queries use it
- `compute_lane_scores` selects `orb_minutes` from `validated_setups` and passes through
- `compute_pairwise_correlation` uses `s.orb_minutes` from `LaneScore`
- `compute_orb_size_stats` GROUP BY now includes `orb_minutes`; key is 3-tuple `(instrument, orb_label, orb_minutes)`
- `save_allocation` lane dict includes `orb_minutes`; uses 3-tuple key for ORB size lookup
- `lane_cap_for` in `scripts/tools/generate_profile_lanes.py` updated to take `orb_minutes` and use 3-tuple key

### Kept Hardcode (deliberate)

`_compute_session_regime` (line 388) keeps `orb_minutes = 5` as a deliberate design
choice with explicit rationale comment. Rationale: the regime gate is a session-health
signal, not strategy-specific. Every strategy in a session sees the same regime
number; O5 is the canonical reference because it's the densest cohort. Changing
this to per-strategy would alter regime semantics, not just fix a bug, and would
break the cross-aperture comparison and the original 2025 backtest design (+630R
regime-only vs -799R per-strategy pause).

## Predicted vs Observed (for audit verification)

The fresh-context auditor predicted specific numerical outcomes from unfiltered
DB queries before the fix was written. The post-fix allocator (filtered, SM-adjusted)
must reproduce these directionally:

| Strategy | Metric | Pre-fix (buggy) | Audit prediction (unfiltered O15) | Post-fix (filtered O15) | ✓/✗ |
|---|---|---|---|---|---|
| SINGAPORE_OPEN_O15 | trailing_expr | 0.2407 | ~0.09 unfiltered | 0.1332 (filtered ATR_P50) | ✓ |
| SINGAPORE_OPEN_O15 | months_negative | 1 | 0 (Apr 2026 +) | 0 | ✓ exact |
| SINGAPORE_OPEN_O15 | p90_orb_pts | 37.8 | ~60.1 (+59%) | 60.1 (+59%) | ✓ exact |
| SINGAPORE_OPEN_O15 | trailing_n | 137 | n/a | 141 | — |
| US_DATA_1000_O15 | trailing_expr | 0.0792 | ~0.0708 unfiltered | dropped from DEPLOY (replaced by VWAP_MID_ALIGNED_O15) | — |
| US_DATA_1000_O15 | p90_orb_pts | 94.9 | ~146.5 (+54%) | 146.5 (correct lookup, on the new replacing lane) | ✓ exact |

The +59% / +54% p90 ratios match exactly because both audit and fix read the same
canonical layer (`orb_outcomes.risk_dollars` per `orb_minutes`).

## Lane Composition Diff (pre-fix vs post-fix on rebalance_date 2026-04-18)

**Pre-fix (buggy — 6 lanes):**
```
[O5 implicit] MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5             ExpR=0.1892 N=264 p90=39.0
[O5 implicit] MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15     ExpR=0.2407 N=137 p90=37.8  ← BUG
[O5 implicit] MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5            ExpR=0.1756 N=253 p90=52.8
[O5 implicit] MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12            ExpR=0.1200 N=262 p90=117.8
[O5 implicit] MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12           ExpR=0.0934 N=236 p90=45.6
[O5 implicit] MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15        ExpR=0.0792 N=257 p90=94.9  ← BUG
```

**Post-fix (corrected — 7 lanes):**
```
[O5 ] MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5                     ExpR=0.1892 N=264 p90=39.0   (no change)
[O5 ] MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5                    ExpR=0.1756 N=253 p90=52.8   (no change)
[O5 ] MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5                       ExpR=0.1200 N=262 p90=117.8  (filter swap COST_LT12→ORB_G5)
[O5 ] MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08                   ExpR=0.1877 N=162 p90=45.6   (filter swap COST_LT12→COST_LT08)
[O15] MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15      ExpR=0.2208 N=124 p90=146.5  (replaces ORB_G5_O15)
[O30] MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30             ExpR=0.1773 N=139 p90=81.4   (new O30 lane fits budget)
[O15] MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15             ExpR=0.1332 N=141 p90=60.1   (corrected — was 0.2407)
```

**Key observations:**
- 6→7 lanes (correct DD budgets allow more lanes per profile)
- O5 lanes that were already correct: ExpR/N/p90 unchanged (no regression)
- SINGAPORE_OPEN_O15 corrected: ExpR -45% (0.2407 → 0.1332), p90 +59% (37.8 → 60.1pts), months_negative reverses (1 → 0)
- US_DATA_1000_O15 (with old ORB_G5 filter, ExpR=0.0708 corrected) drops out because VWAP_MID_ALIGNED_O15 (ExpR=0.2208) is now visible and dominant on the same session
- Aperture mix: 4 O5, 2 O15, 1 O30 (was implicitly 6 O5 even though 2 were validated as O15)

## Why the bug existed

The allocator was written assuming all strategies are O5. When O15 strategies were added
to `validated_setups`, the allocator's strategy load query (line 232) didn't select
`orb_minutes`, so the O15 attribute was lost on the way in, and every downstream query
substituted the hardcoded constant. Fail-quiet path: every query returned non-zero rows
(O5 data exists for every session), so no exception, no warning. The O15 lanes ranked
strongly in the buggy allocator because O5 ExpR was systematically higher than O15 ExpR
on these sessions — directly favoring the bug.

## Why a single-context audit would not have caught this

The bug presents as "ExpR looks right, lane deploys, JSON validates." Same-context
self-review would read `lane_allocator.py` looking for the answer it expects (trailing
window math) and miss the structural hardcode. The fresh-context evidence-auditor
agent was given a tight 4-claim sheet (later expanded to 5 when the auditor surfaced
the hardcode itself) with explicit verification commands and an instruction to verify
against actual file/DB contents not commit messages or doc claims. Output format was
strict: VERIFIED / REFUTED / UNVERIFIABLE per claim. The hardcode was caught on the
auditor's exploration of trailing-stats reproducibility (Claim 1), elevated to its own
Claim 5, and verified at all 5 file:line sites with magnitude estimates per affected
lane.

## Tests

- 39 existing tests still pass (zero regression)
- 1 new regression test added: `test_orb_size_stats_dd_uses_correct_aperture_not_o5`
  - Constructs an O15 lane with budget $200; correct O15 P90 (147pts) gives DD=$220.50
    which doesn't fit; bug regression would use O5 P90 (95pts → DD=$142.50) and incorrectly
    fit the lane. Test fails if the bug returns.
- 1 new aperture-distinguishing test: `test_lane_cap_for_distinguishes_apertures`
  - Verifies `lane_cap_for` returns the right P90 for both O5 and O15 lanes; no
    cross-aperture leakage, falls back correctly when aperture missing.
- Existing `test_orb_size_stats_used_in_dd` and `test_orb_size_stats_budget_constraint`
  updated to use 3-tuple keys (previously they relied on the (100,100) fallback and
  passed for the wrong reason).

47/47 tests pass.

## Drift Check

117 checks PASSED, 0 FAILED, 16 advisory (all pre-existing).

## Live impact

The interim PR (#188) paused the 2 O15 lanes with explicit audit citation while this
structural fix was under review. After this fix lands, the rerun JSON above replaces
the old `lane_allocation.json`. Operator decision required: accept the new 7-lane
composition (which adds VWAP_MID_ALIGNED_O15, ATR_P50_O30, COST_LT08 swaps) or hold
on additional review for the new entrants.

The 6 ORIGINAL DEPLOY lanes are not all preserved in the new allocation — 3 are
replaced (NYSE_OPEN, TOKYO_OPEN, US_DATA_1000_ORB_G5_O15). This is the allocator
correctly responding to better candidates being visible under per-aperture math; it
is not a separate decision needing capital review per se, but operator should glance
at the 3 replacement lanes' validation history before flipping the live JSON.

## Reproduction / Outputs

To reproduce the bug confirmation and the post-fix verification:

```bash
# 1. Bug evidence (run from canompx3 main worktree before this fix)
cd C:/Users/joshd/canompx3
grep -n "orb_minutes = 5\|orb_minutes\": 5" trading_app/lane_allocator.py
# Expected: 5 hits (lines 116, 131, 388, 525, 786 pre-fix)

# 2. Magnitude — unfiltered O5 vs O15 ExpR for SINGAPORE_OPEN
duckdb gold.db <<EOF
SELECT orb_minutes, AVG(pnl_r) AS expr, COUNT(*) AS n
FROM orb_outcomes
WHERE symbol = 'MNQ' AND orb_label = 'SINGAPORE_OPEN'
  AND entry_model = 'E2' AND rr_target = 1.5 AND confirm_bars = 1
  AND outcome IN ('win','loss')
  AND trading_day >= '2025-04-01' AND trading_day < '2026-04-18'
GROUP BY orb_minutes ORDER BY orb_minutes;
EOF
# Expected: O5 expr ~0.13, O15 expr ~0.09 — wrong-aperture inflates ExpR

# 3. Post-fix verification (from canompx3-allocator-fix worktree)
cd C:/Users/joshd/canompx3-allocator-fix
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python scripts/tools/rebalance_lanes.py \
    --date 2026-04-18 --output /tmp/lane_allocation_POST_FIX.json
# Expected lanes (7): aperture mix 4 O5 / 2 O15 / 1 O30
# SINGAPORE_OPEN_ATR_P50_O15 ExpR ~0.1332 (was 0.2407)
# p90_orb_pts 60.1 (was 37.8)

# 4. Tests
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python -m pytest \
    tests/test_trading_app/test_lane_allocator.py \
    tests/test_tools/test_generate_profile_lanes.py -v
# Expected: 47/47 pass

# 5. Drift
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python pipeline/check_drift.py
# Expected: 117 PASSED, 0 FAILED, 16 advisory
```

Pre-fix and post-fix JSON snapshots:
- Pre: `C:/Users/joshd/AppData/Local/Temp/lane_allocation_PRE_FIX.json` (6 lanes, all keyed as O5 implicit)
- Post: `C:/Users/joshd/AppData/Local/Temp/lane_allocation_POST_FIX.json` (7 lanes, explicit `orb_minutes` field on each)

## Caveats / Disconfirming Evidence / Limitations

1. **Filtered vs unfiltered ExpR.** The audit's predicted O15 ExpR (`~0.09`)
   was computed unfiltered. The post-fix allocator output (`0.1332`) is
   filtered (ATR_P50) and SM-adjusted. Direction matches (down from 0.2407)
   and order of magnitude is right, but exact numerical equality is not
   expected and not claimed. The +59% / +54% p90 ratios DO match exactly
   because both audit and fix read `orb_outcomes.risk_dollars` per
   `orb_minutes` directly (no filter, no SM adjustment).

2. **Kept hardcode in `_compute_session_regime` is a design choice, not a
   bug fix omission.** The regime gate is a session-level signal. If a
   future audit argues this should be parameterized on aperture, that's a
   separate discussion with literature grounding (the existing comment cites
   the original 2025 backtest +630R regime-only vs -799R per-strategy
   pause). I am NOT claiming the kept-hardcode is unconditionally correct;
   I am claiming it is consistent with the original design intent and that
   changing it requires a separate research-backed amendment.

3. **Lane composition shift (6 → 7 lanes) is a downstream consequence, not
   a verified-deploy-ready set.** The 3 lane swaps (NYSE_OPEN COST_LT12 →
   ORB_G5; TOKYO_OPEN COST_LT12 → COST_LT08; US_DATA_1000 ORB_G5_O15 →
   VWAP_MID_ALIGNED_O15) and the new O30 lane (SINGAPORE_OPEN_ATR_P50_O30)
   are what the allocator now produces under correct math. Operator should
   review their FDR validation history before flipping live JSON. No claim
   is made here that all 7 new lanes are deploy-ready in isolation; only
   that the allocator's scoring of them is now correct.

4. **OOS power floor for the new lane composition is NOT verified here.**
   Per `feedback_oos_power_floor.md`, any binary OOS gate needs a Cohen's d
   power calc against post-2026-01-01 OOS. The allocator decision rests on
   trailing-window IS stats, not an OOS gate, so the power-floor rule is
   vacuously satisfied — but OOS validation of the corrected lanes is a
   separate prerequisite for real-capital flip. This audit does not clear
   that gate.

5. **Pyright reported transient stale-cache errors during edits.** Runtime
   was verified at every step (47/47 tests pass, drift PASS, allocator
   reruns successfully). Pyright errors are LSP cache, not code defects.
   This is documented for future-me debugging the same false alarm.

6. **The allocator-internal "ranking on _effective_annual_r" path was not
   re-audited under the new aperture math.** This audit covers data-input
   correctness (right ExpR for right strategy). Whether the resulting rank
   ordering produces the optimal portfolio under the corrected inputs is a
   distinct question and is NOT claimed here.

## Related

- Audit agent transcripts: `a1e76860ea5635815` + `ab19fcc9e39814af2`
- Interim mitigation PR: #188 (`fix/allocator-pause-o15-interim`)
- Structural fix PR: TBD (`fix/allocator-orb-minutes-hardcode`)
- Memory entry queued for: `feedback_allocator_orb_minutes_hardcode_2026_04_30.md`
