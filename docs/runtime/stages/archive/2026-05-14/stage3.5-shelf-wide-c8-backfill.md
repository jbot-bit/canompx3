---
task: Shelf-wide C8 OOS backfill via canonical scripts/tools/backfill_deployability_evidence.py. APPLIED 2026-05-11. 844 rows flipped from c8_oos_status=NULL to canonical evaluator result.
mode: IMPLEMENTATION
scope_lock:
  - docs/runtime/stages/stage3.5-shelf-wide-c8-backfill.md
  - docs/audit/results/2026-05-11-stage3.5-shelf-wide-c8-backfill-applied.md
blast_radius:
  DB-WRITE STAGE. Writes to validated_setups.c8_oos_status on 844
  active rows (was NULL, now canonical evaluator result). No code
  changes — invokes existing scripts/tools/backfill_deployability_evidence.py
  with --evidence c8_oos --instrument ALL --write. Reads orb_outcomes
  + daily_features under Mode A holdout (HOLDOUT_SACRED_FROM=2026-01-01).
  Does NOT modify lane_allocation.json, broker state, schema, or
  validated_setups.status / oos_exp_r columns. Capital impact: NONE
  for currently-deployed lanes (all 3 + their siblings PASS C8).
  Unblocks Stage 4 (family_singleton conditional downgrade) per
  Stage 3 decision-3.
---

## Purpose

Stage 3 decision-3 (locked by user) requires shelf-wide C8 evidence
before Stage 4 codifies the family_singleton conditional downgrade.
Pre-backfill: 844 of 847 active rows had `c8_oos_status IS NULL`.
Stage 3.5 runs the canonical backfill tool to fill that evidence.

## Done criteria

1. Dry-run plan inspected; predicted distribution previewed to user. **[DONE]**
2. Sanity check: re-evaluate an existing PASSED row, confirm match. **[DONE — `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` re-evaluates to PASSED]**
3. User authorisation captured. **[DONE — Stage 3 decision-3 + explicit `<bash-input>` invocation]**
4. Write applied to gold.db. **[DONE — exit 0]**
5. Post-write DB state verified: 0 NULL c8_oos_status on active shelf. **[DONE — 847 active rows, 0 NULL]**
6. Capital-impact audit on deployed lanes + siblings. **[DONE — all 3 deployed lanes + all family siblings PASS C8]**
7. Result MD written with Bloomberg-grade provenance. **[DONE — `2026-05-11-stage3.5-shelf-wide-c8-backfill-applied.md`]**
8. Drift check pass + commit + push.

## Verification commands

```bash
# Post-write DB state
python -c "
import duckdb
con = duckdb.connect('C:/Users/joshd/canompx3/gold.db', read_only=True)
for r in con.execute('''
    SELECT instrument, COALESCE(c8_oos_status, 'NULL') AS s, COUNT(*)
    FROM validated_setups WHERE status='active'
    GROUP BY 1,2 ORDER BY 1,2
''').fetchall():
    print(r)
"

# Deployed lane sanity
python -c "
import duckdb
con = duckdb.connect('C:/Users/joshd/canompx3/gold.db', read_only=True)
for r in con.execute('''
    SELECT strategy_id, c8_oos_status FROM validated_setups
    WHERE strategy_id IN (
        'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100',
        'MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12',
        'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15'
    )
''').fetchall():
    print(r)
"
```

## Sibling worktree status

- `stage1/generalize-tbbo-slippage-inference` — PR #258 awaiting merge.
- `stage2/family-singleton-doctrine` — Disposition C + 3 self-audit passes.
- `stage3/c5-doctrine-resolution` — floor spec + 3 decisions locked.
- `stage3.5/shelf-wide-c8-backfill` — THIS BRANCH. APPLIED.
- Next: `stage4/family-singleton-conditional` worktree opens once user
  signals to proceed; consumes Stage 3 § 4 spec + Stage 3.5 evidence.
