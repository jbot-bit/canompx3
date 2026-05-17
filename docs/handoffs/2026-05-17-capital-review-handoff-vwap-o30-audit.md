# Capital-Review Handoff — VWAP_MID_ALIGNED_O30 Audit Surfaced 3-Lane Re-Allocation

**Date:** 2026-05-17
**Origin:** Audit-only thread on rank-3 AUDIT_GAP_ONLY VWAP_MID_ALIGNED_O30 pre-reg
**Status:** AUDIT COMPLETE. Re-allocation DEFERRED to capital-review.
**Author thread:** verification audit per plan dated 2026-05-17 (mode: audit-only, no edits)
**Files edited this thread:** zero. **Commits:** zero. **Allocator state mutated:** none.

---

## TL;DR for the capital-review reader

The audit ran the canonical rebalancer (`scripts/tools/rebalance_lanes.py`) into a scratch path to verify the 2026-05-14 `lane_allocation.json` snapshot's freshness. The scratch output reveals that refreshing the canonical JSON today would not be a "verdict-field reconciliation" — it would commit a substantive 3-lane re-allocation. The audit-only thread stopped before any disk mutation. This handoff captures exactly what the dry-run produced, why it exceeds audit scope, and what a capital-review thread needs to evaluate before any rebalance commit.

The original audit target — `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` — does NOT deploy in the dry-run. It clears the chordia gate (`verdict: PASS_CHORDIA`, audit_age_days=3) but is correlation-displaced by `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` at the same session. So F4 of the audit plan (the audit-log's "does not deploy" narrative) is now falsifiable and confirmed correct in direction. The displacement happens at the correlation gate, not the ranking gate as the audit-log note claimed.

A new finding from this thread — F6 — was surfaced: the allocator silently drops correlation-rejected candidates from the JSON entirely (no `correlation_rejected[]` bucket). This is a schema-non-self-describing class issue, not a deployment bug.

---

## The 3 drops, 2 adds (capital-impacting)

**Profile:** `topstep_50k_mnq_auto` | **Rebalance date queried:** 2026-05-17 | **Trailing window:** unchanged

### Dropped lanes (in current canonical JSON, absent from dry-run lanes[])

1. **`MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K`** — DROP, REASON: correlation-displaced
   - Dry-run report line: `[DROP] MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K (DEPLOY: Session HOT (+0.0492), ExpR=+0.1375, N=240)`
   - Still healthy on its own metrics: HOT session regime, positive ExpR, N=240
   - Loses correlation gate to `OVNRNG_100` at the same COMEX_SETTLE session
   - **Capital-review concern:** the canonical JSON keeps this in `lanes[]`. Dropping a healthy currently-deployed lane needs explicit hysteresis review before commit. Memory rule `feedback_provisional_not_paused_rr_variant_drift.md` applies (same-session RR-variant displacement risk).

2. **`MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25`** — DROP, REASON: c8 gate failure (correct, expected)
   - Dry-run report line: `[DROP] MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25 (PAUSE: c8 gate: c8_oos_status='FAILED_RATIO' (Criterion 8 OOS deployment gate))`
   - Known from memory: `feedback_allocator_gate_class_pattern_fail_open.md` (lane-after-column = BUG, locked) and `feedback_pre_impl_git_archaeology_bug_vs_grandfather.md` (OVNRNG_25 deployed 20d after c8 column = BUG, not grandfathered)
   - **Capital-review concern:** none — this drop is the correct fail-closed outcome and matches doctrine. Worth confirming the canonical JSON has been intentionally left stale (lane retained pending audit) rather than reflecting a regression.

3. **`MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`** — DROP, REASON: correlation-displaced by RR1.5 sibling
   - Dry-run report line: `[DROP] MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15 (DEPLOY: Session HOT (+0.0419), ExpR=+0.2052, N=120)`
   - Still healthy: HOT, ExpR=+0.2052, N=120 — strongest by ExpR of the three drops
   - Loses to its own RR1.5 sibling (`...RR1.5_CB1_VWAP_MID_ALIGNED_O15`, annual_r=27.1) at the correlation gate
   - **Capital-review concern:** same-session RR1.0 → RR1.5 swap. Direct analogue to the lifecycle-pause-doesn't-carry incident (`feedback_provisional_not_paused_rr_variant_drift.md`, L1 NYSE_OPEN RR1.0→RR1.5, 2026-05-11). If RR1.5 sibling has not been independently lifecycle-cleared, this swap inherits whatever pause state lived on RR1.0 — verify before commit.

### Added lanes (absent from current canonical, present in dry-run lanes[])

1. **`MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`** — NEW, annual_r=30.9, ExpR=+0.2159, N=155
   - Dry-run rank: **#1**
   - Dry-run report line: `[NEW] MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 (annual_r=30.9)`
   - **Capital-review concern:** this is a NEVER-DEPLOYED lane being promoted directly into rank-1. Full pre-deployment institutional review required: chordia verdict, c8 OOS status, OOS power floor, mechanism prior, lifecycle history. The dry-run shows `chordia_verdict: PASS_CHORDIA` and `c8_oos_status: PASSED` — gates clear — but those are necessary, not sufficient, for first-time deployment.

2. **`MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`** — NEW, annual_r=27.1, ExpR=+0.2416, N=112
   - Dry-run rank: **#2**
   - Dry-run report line: `[NEW] MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 (annual_r=27.1)`
   - **Capital-review concern:** same RR-variant inheritance question as drop #3. The two are paired: this lane is the displacer; that lane is the displaced. Decide them together.

### Kept lane

- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` — KEEP (rank-3 in dry-run, unchanged status)

### Net deployable slot count

- Canonical (2026-05-14): 4 deployed lanes (out of 4 max slots, no profile-side slot constraint exceeded)
- Dry-run (2026-05-17): **3 deployed lanes** — one slot unfilled. The audit thread did not investigate why slot 4 went empty (DD-budget gate? correlation gate? candidates exhausted?). Capital-review must determine whether this is correct fail-closed behavior or a regression.

---

## The original audit target — disposition in dry-run

| Lane | 2026-05-14 canonical | 2026-05-17 dry-run | Reason |
|---|---|---|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` (audit target) | `paused[]` / status=PAUSE / chordia_verdict=MISSING | **ABSENT from all buckets** | Clears chordia (PASS_CHORDIA, age 3d) and c8 (PASSED) gates, then loses correlation gate vs `...RR1.5_CB1_VWAP_MID_ALIGNED_O15` (the new rank-2 addition). Silently dropped — see F6 below. |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30_S075` (sibling) | `paused[]` / MISSING | `paused[]` / MISSING | Unchanged — its own pre-reg does not exist, audit log has no entry, lookup returns None. Field-equality protection intact. |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30` (sibling) | `paused[]` / MISSING | `paused[]` / MISSING | Same as above. |

**F3 protection verified:** both siblings remain MISSING in the dry-run; the audit-log lookup did not flip them by substring match.

---

## F6 — `save_allocation` has no `correlation_rejected[]` bucket

**Class:** schema non-self-describing
**Severity:** MEDIUM
**Locus:** `trading_app/lane_allocator.py:1212-1275` (`save_allocation`), with the silent `continue` at `:1015-1016` inside `build_allocation` (the correlation gate)

**Evidence trace:**
- `build_allocation` runs `apply_chordia_gate` → `apply_c8_gate` → `apply_live_tradeability_gate` (lines 961-963). Each gate explicitly transitions failing candidates to `status="PAUSE"`. PAUSE lanes are persisted to `paused[]` by `save_allocation` (line 1272).
- After the gates, line 966 filters to `status in ("DEPLOY", "RESUME", "PROVISIONAL")` — the deployable pool.
- Lines 1003-1016: greedy selection with a correlation gate. When a candidate fails the gate (rho > `RHO_REJECT_THRESHOLD`), it hits `continue` at line 1016. **No status mutation. No bucket assignment.**
- `save_allocation` (line 1272-1273) writes only `status == "PAUSE"` and `status == "STALE"` rows. Correlation-rejected candidates retain their DEPLOY/RESUME/PROVISIONAL status but are absent from `lanes[]` (never selected) and absent from `paused[]` / `stale[]` (status never changed). They vanish from the JSON.
- DD-budget rejections (line 1031-1032) and hysteresis rejections (line 1044-1048) have the same pattern: silent `continue` without status mutation.

**Why this matters:**
- A future auditor reading `lane_allocation.json` for `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` cannot distinguish between four states:
  1. "Never had a chordia audit" (MISSING in paused)
  2. "Had a passing audit, beaten by correlation"
  3. "Had a passing audit, beaten by DD-budget"
  4. "Had a passing audit, beaten by hysteresis"
- Today's audit only resolved this via running the rebalancer ourselves and reading the human-readable stdout report. The JSON alone is not enough to answer.

**Not a deployment bug** — the gates work correctly and the JSON is internally consistent. It's an auditability gap. Capture as a separate stage if pursued; do not bundle with the rebalance commit.

**Proposed schema fix (out of scope here):** add a `displaced[]` array to `save_allocation` containing every candidate that cleared all hard gates but lost to a soft gate (correlation / DD-budget / hysteresis), with the rejection reason and the winning lane's strategy_id. This makes the JSON fully self-describing and enables future audits to skip the rebalancer-rerun step.

---

## Exact commands run by this audit (for reproducibility)

```bash
# Step 0: confirm canonical state
git log --oneline -10 docs/runtime/chordia_audit_log.yaml
# (output: 2cf2c9cc audit(log): MNQ US_DATA_1000 VWAP_MID_ALIGNED O30 RR1.0 PASS_CHORDIA — present)

python -c "import json; d=json.load(open('docs/runtime/lane_allocation.json')); [print(l['strategy_id'], '|', l.get('status'), '|', l.get('chordia_verdict')) for b in ('lanes','paused','stale') for l in d.get(b,[]) if 'VWAP_MID_ALIGNED_O30' in l.get('strategy_id','')]"
# (output: all three lanes in paused[] with status=PAUSE, chordia_verdict=MISSING)

# Step 1: confirm allocator flags
python scripts/tools/rebalance_lanes.py --help
# (no --dry-run flag; --output is the safe substitute)

# Step 2: scratch-path dry-run
python scripts/tools/rebalance_lanes.py --date 2026-05-17 --output "$TEMP/lane_allocation_dryrun.json"

# Step 3: diff
python -c "
import json, os
canonical = json.load(open('docs/runtime/lane_allocation.json'))
dryrun = json.load(open(os.path.expandvars(r'%TEMP%\\lane_allocation_dryrun.json')))
c = {l['strategy_id'] for l in canonical['lanes']}
d = {l['strategy_id'] for l in dryrun['lanes']}
print('DROP:', sorted(c-d)); print('ADD:', sorted(d-c)); print('KEEP:', sorted(c&d))
"

# Step 4-5 verification: audit-log lookup is exact-match, not substring
python -c "
from datetime import date
from trading_app.chordia import load_chordia_audit_log
log = load_chordia_audit_log()
for sid in [
    'MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30',
    'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30',
    'MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30_S075',
]:
    print(sid, '->', log.verdict(sid), 'age:', log.audit_age_days(sid, date(2026,5,17)))
"
# (output: base=PASS_CHORDIA age 3, RR1.5=None, S075=None — exact-match confirmed)
```

---

## What the capital-review thread needs to decide

In approximate order of dependency:

1. **OVNRNG_25 drop** — confirm this is the correct fail-closed outcome of the c8 gate landing 2026-05-14. Likely no action required beyond accepting the drop on next rebalance.

2. **VWAP_MID_ALIGNED_O15 RR1.0 → RR1.5 swap** — verify the RR1.5 sibling has been independently lifecycle-cleared. If lifecycle pauses on RR1.0 do not carry across (per L1 2026-05-11 NYSE_OPEN incident), an audit-trail check is required before the swap is committed. Mandatory read: `feedback_provisional_not_paused_rr_variant_drift.md`.

3. **ORB_VOL_2K drop** — apply hysteresis review. The lane is HOT, positive ExpR, N=240. Dropping a healthy currently-deployed lane to a never-deployed one (OVNRNG_100) on correlation displacement deserves explicit numeric justification (annual_r delta, correlation rho, capital efficiency).

4. **OVNRNG_100 first-time deployment** — full pre-deployment review: chordia verdict & age, c8 OOS status, OOS power floor, mechanism prior cite, lifecycle history, correlation with the kept lane.

5. **Slot 4 stays empty?** — investigate. If DD-budget is the limiter, surface the numeric gap and the next best candidate's lane_dd. If correlation is the limiter, surface the next-ranked candidate and which lane it correlates with.

6. **F6 schema fix** — optional separate stage. Adding `displaced[]` would have shortened this audit by one step (no rebalancer re-run needed to know the base lane was correlation-displaced). Low capital impact, MEDIUM auditability value.

---

## What this thread did NOT cover

- Re-running pre-reg validator on the original target YAML (F5 of the audit plan — deferred, file unchanged since promotion).
- Authoring sibling pre-regs for RR1.5/S075 variants (out of scope per the base pre-reg's own `out_of_scope` block).
- Touching `chordia_audit_log.yaml` to encode a structured `verdict_override` field (F1 of the audit plan — out of scope, schema change with broad blast radius).
- Touching `lane_allocation.json` (the entire reason for this handoff).

---

## Verification commands for the capital-review reader to start from

```bash
# Confirm the dry-run state hasn't moved underneath you
git status --short docs/runtime/lane_allocation.json docs/runtime/chordia_audit_log.yaml
python pipeline/check_drift.py  # must remain green

# Re-run the dry-run; expect identical output if no upstream changes landed
python scripts/tools/rebalance_lanes.py --date $(date -I) --output "$TEMP/lane_allocation_dryrun.json"

# Confirm the audit-log entry is still trusted
python -c "from datetime import date; from trading_app.chordia import load_chordia_audit_log; l=load_chordia_audit_log(); print(l.verdict('MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30'))"
# Expected: PASS_CHORDIA
```

If any of these diverge from this handoff's recorded state, re-run the audit thread before proceeding to capital-review.
