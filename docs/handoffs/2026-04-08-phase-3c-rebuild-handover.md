# Handover — Phase 3c Rebuild Ready to Execute (2026-04-08)

**Branch:** `phase-3c-rebuild-orb-outcomes` (fresh branch off main, 0 commits so far)
**Parent:** `main` @ `bc6df12` (Phase 3b merged)
**Author:** Claude Code session, terminal B
**Status:** PAUSED AT PRE-EXECUTE CHECKPOINT — backups created, plan written, NO destructive operations run yet

## Where we are

The canonical-data-redownload plan is at Phase 3c: rebuild
`orb_outcomes` and `daily_features` for MNQ/MES/MGC from the new
real-micro `bars_1m` data that landed in Phase 2 (commit `82e8b60`,
main `73311fe` → now on main).

All Phase 3a/3b code foundation has already merged to main:
- Phase 3a (data_era module + `parent_symbol` field): commits
  `4284465` → `b032a03` → merge `a3b4f32`
- Phase 3b (`requires_micro_data` @property on StrategyFilter):
  commits `853832b` → `cef50f1` → merge `bc6df12`

Phase 3c is a **pure data operation** — no code edits planned. The
existing `pipeline/build_daily_features.py` and
`trading_app/outcome_builder.py` are the tools; Stage 3c just runs them
for the right instruments and date ranges, after a DELETE.

## What's already done (safe checkpoint)

### 1. Backups created in gold.db

Two SQL-level backups were written to the canonical
`gold.db` (path: `pipeline.paths.GOLD_DB_PATH`):

| Backup table | Row count | Contents |
|---|---|---|
| `orb_outcomes_pre_phase_3c_backup` | **9,413,640** | All MNQ/MES/MGC rows from orb_outcomes |
| `daily_features_pre_phase_3c_backup` | **41,478** | All MNQ/MES/MGC rows from daily_features |

Per-instrument breakdown:
```
orb_outcomes: MNQ=3,699,540  MES=2,810,484  MGC=2,903,616   total 9,413,640
daily_features: MNQ=13,830   MES=13,830     MGC=13,818      total 41,478
```

Reversibility (if the rebuild corrupts anything):
```sql
DELETE FROM orb_outcomes  WHERE symbol IN ('MNQ','MES','MGC');
INSERT INTO orb_outcomes  SELECT * FROM orb_outcomes_pre_phase_3c_backup;
DELETE FROM daily_features WHERE symbol IN ('MNQ','MES','MGC');
INSERT INTO daily_features SELECT * FROM daily_features_pre_phase_3c_backup;
```

Drop the backups after successful validation:
```sql
DROP TABLE orb_outcomes_pre_phase_3c_backup;
DROP TABLE daily_features_pre_phase_3c_backup;
```

### 2. Pre-execute ground truth (snapshot)

**DB lock:** gold.db was writable at checkpoint time — no concurrent
writer holding the lock. Live bot was either dormant or in read-only
mode.

**orb_outcomes current state:**
```
MES:  2,810,484 rows  from 2010-06-06 to 2026-04-05
MGC:  2,903,616 rows  from 2010-06-06 to 2026-04-05
MNQ:  3,699,540 rows  from 2010-06-06 to 2026-04-05
```

**Orphaned pre-launch rows** (these reference bars that after Phase 2
are now under symbol='NQ'/'ES'/'GC' — they CANNOT be rebuilt and MUST
be deleted):
```
MNQ < 2019-05-06:  1,595,376 orphaned rows
MES < 2019-05-06:    909,468 orphaned rows
MGC < 2023-09-11:  2,290,860 orphaned rows
                  ─────────
total orphaned:    4,795,704 rows (50.9% of the 9.4M total)
```

**Post-launch rows that need rebuilding from new real-micro bars:**
```
MNQ >= 2019-05-06:  ~2.10M rows  (3,699,540 − 1,595,376)
MES >= 2019-05-06:  ~1.90M rows  (2,810,484 −   909,468)
MGC >= 2023-09-11:  ~0.61M rows  (2,903,616 − 2,290,860)
                    ─────
total to rebuild:   ~4.62M rows
```

**daily_features post-launch:**
```
MNQ (>= 2019-05-06):  6,063 rows, 2,021 trading days, 3 apertures each
MES (>= 2019-05-06):  6,063 rows, 2,021 trading days, 3 apertures each
MGC (>= 2023-09-11):  2,226 rows,   742 trading days, 3 apertures each
```

**`validated_setups` (DO NOT TOUCH in Phase 3c):**
```
MNQ: 101 strategies
MES:  14 strategies
MGC:   9 strategies
```
These are the grandfathered pre-2026-04-07 setups. Stage 3c will leave
them alone — Stage 3d will later audit them against Phase 3a era
discipline.

## What Phase 3c needs to do (the plan)

### Step A — Delete orphaned pre-launch rows (fast, clearly safe)

4.8M rows that reference bars no longer under the corresponding
symbol. Cannot be rebuilt. Definitely stale.

```sql
DELETE FROM orb_outcomes
WHERE symbol = 'MNQ' AND trading_day < '2019-05-06';

DELETE FROM orb_outcomes
WHERE symbol = 'MES' AND trading_day < '2019-05-06';

DELETE FROM orb_outcomes
WHERE symbol = 'MGC' AND trading_day < '2023-09-11';

DELETE FROM daily_features
WHERE symbol = 'MNQ' AND trading_day < '2019-05-06';

DELETE FROM daily_features
WHERE symbol = 'MES' AND trading_day < '2019-05-06';

DELETE FROM daily_features
WHERE symbol = 'MGC' AND trading_day < '2023-09-11';
```

After Step A, query expected state:
```sql
-- Every surviving row must have trading_day >= micro_launch_day
SELECT symbol, MIN(trading_day), MAX(trading_day), COUNT(*)
FROM orb_outcomes WHERE symbol IN ('MNQ','MES','MGC') GROUP BY symbol;
```
Expected: MNQ/MES from 2019-05-06, MGC from 2023-09-11, all ≤ 2026-04-05.

### Step B — Delete post-launch stale rows

These were built pre-Phase-2 from the OLD bars_1m (parent proxy for
pre-2024-02-05, real-micro for 2024-02-05+). Cannot cheaply distinguish
which are valid; cleanest is DELETE all post-launch and rebuild from new
real-micro bars.

```sql
DELETE FROM orb_outcomes WHERE symbol IN ('MNQ','MES','MGC');
DELETE FROM daily_features WHERE symbol IN ('MNQ','MES','MGC');
```

**After Steps A + B, orb_outcomes and daily_features will have ZERO
rows for MNQ/MES/MGC. This is the expected intermediate state.**

### Step C — Rebuild daily_features (must come before orb_outcomes)

daily_features is a prerequisite for outcome_builder.py because outcome
computation reads session features. Rebuild per instrument × aperture
(5m/15m/30m) using existing canonical script:

```bash
# MGC (smallest, do first as smoke test — ~742 days × 3 apertures)
python pipeline/build_daily_features.py --instrument MGC --start 2023-09-11 --end 2026-04-07 --orb-minutes 5
python pipeline/build_daily_features.py --instrument MGC --start 2023-09-11 --end 2026-04-07 --orb-minutes 15
python pipeline/build_daily_features.py --instrument MGC --start 2023-09-11 --end 2026-04-07 --orb-minutes 30

# MES (~2021 days × 3 apertures)
python pipeline/build_daily_features.py --instrument MES --start 2019-05-06 --end 2026-04-07 --orb-minutes 5
python pipeline/build_daily_features.py --instrument MES --start 2019-05-06 --end 2026-04-07 --orb-minutes 15
python pipeline/build_daily_features.py --instrument MES --start 2019-05-06 --end 2026-04-07 --orb-minutes 30

# MNQ (~2021 days × 3 apertures)
python pipeline/build_daily_features.py --instrument MNQ --start 2019-05-06 --end 2026-04-07 --orb-minutes 5
python pipeline/build_daily_features.py --instrument MNQ --start 2019-05-06 --end 2026-04-07 --orb-minutes 15
python pipeline/build_daily_features.py --instrument MNQ --start 2019-05-06 --end 2026-04-07 --orb-minutes 30
```

Each invocation is idempotent (DELETE+INSERT per date). On MGC Apr-7
sample earlier today, one aperture took ~0.2s per day → ~150s for MGC
full range, ~400s for MNQ and MES each. Total Step C: ~20 minutes.

Each script prints integrity check results; confirm PASSED before
moving on.

### Step D — Rebuild orb_outcomes

`trading_app/outcome_builder.py` reads daily_features + bars_1m and
produces orb_outcomes rows. Rebuild per instrument:

```bash
# MGC first (smallest)
python -m trading_app.outcome_builder --instrument MGC --force --start 2023-09-11 --end 2026-04-07 --orb-minutes 5
python -m trading_app.outcome_builder --instrument MGC --force --start 2023-09-11 --end 2026-04-07 --orb-minutes 15
python -m trading_app.outcome_builder --instrument MGC --force --start 2023-09-11 --end 2026-04-07 --orb-minutes 30

# MES
python -m trading_app.outcome_builder --instrument MES --force --start 2019-05-06 --end 2026-04-07 --orb-minutes 5
python -m trading_app.outcome_builder --instrument MES --force --start 2019-05-06 --end 2026-04-07 --orb-minutes 15
python -m trading_app.outcome_builder --instrument MES --force --start 2019-05-06 --end 2026-04-07 --orb-minutes 30

# MNQ
python -m trading_app.outcome_builder --instrument MNQ --force --start 2019-05-06 --end 2026-04-07 --orb-minutes 5
python -m trading_app.outcome_builder --instrument MNQ --force --start 2019-05-06 --end 2026-04-07 --orb-minutes 15
python -m trading_app.outcome_builder --instrument MNQ --force --start 2019-05-06 --end 2026-04-07 --orb-minutes 30
```

**Time estimate:** outcome_builder is slower than daily_features. Past
rebuilds for full MNQ historical (Apr 3 rebuild, 8 years) took ~1-2
hours. For MNQ/MES/MGC combined post-launch, estimate **2-4 hours
total**.

**Suggest running in background** via `nohup` or similar and monitoring
progress. Each script writes to gold.db so they must run SEQUENTIALLY,
not in parallel (duckdb single-writer rule).

### Step E — Validate post-rebuild

1. Row counts match expected:
   ```sql
   SELECT symbol, COUNT(*), MIN(trading_day), MAX(trading_day)
   FROM orb_outcomes WHERE symbol IN ('MNQ','MES','MGC') GROUP BY symbol;

   SELECT symbol, COUNT(*), MIN(trading_day), MAX(trading_day)
   FROM daily_features WHERE symbol IN ('MNQ','MES','MGC') GROUP BY symbol;
   ```
2. No pre-launch rows:
   ```sql
   SELECT COUNT(*) FROM orb_outcomes
   WHERE (symbol='MNQ' AND trading_day<'2019-05-06')
      OR (symbol='MES' AND trading_day<'2019-05-06')
      OR (symbol='MGC' AND trading_day<'2023-09-11');
   -- expect 0
   ```
3. Drift check clean:
   ```bash
   PYTHONPATH=. python pipeline/check_drift.py
   # expect 85 passed, 0 failed, 7 advisory
   ```
4. Behavioral audit clean:
   ```bash
   PYTHONPATH=. python scripts/tools/audit_behavioral.py
   # expect 7/7
   ```
5. Test suite green:
   ```bash
   python -m pytest tests/test_pipeline/ tests/test_trading_app/ -q
   # expect all pass
   ```
6. **Validated setups still exist** (sanity check — Stage 3c MUST NOT
   touch validated_setups):
   ```sql
   SELECT instrument, COUNT(*) FROM validated_setups
   WHERE status='active' GROUP BY instrument;
   -- expect MNQ=101, MES=14, MGC=9 (unchanged from checkpoint)
   ```

### Step F — Commit the rebuild log

Stage 3c doesn't edit any code files, so nothing to commit except this
handover (and possibly a rebuild audit log if you want one). Delete the
backup tables once validation passes:

```sql
DROP TABLE orb_outcomes_pre_phase_3c_backup;
DROP TABLE daily_features_pre_phase_3c_backup;
```

Then commit the handover:
```bash
git add docs/handoffs/2026-04-08-phase-3c-rebuild-handover.md
git commit -m "docs(phase-3c): rebuild handover and execution log"
```

## Risks and guardrails

1. **Concurrent writer (live bot).** gold.db is single-writer. If the
   live bot kicks off a session during the rebuild, it will either
   block or fail. Before starting Step B, confirm:
   ```bash
   ps aux | grep -i 'session_orchestrator\|paper_trader\|live' | grep -v grep
   ```
   No live-bot processes → safe to proceed.

2. **Mid-operation abort.** If any script fails mid-rebuild, the DB is
   in a partial state. Use the backup to restore cleanly:
   ```sql
   DELETE FROM orb_outcomes  WHERE symbol IN ('MNQ','MES','MGC');
   INSERT INTO orb_outcomes  SELECT * FROM orb_outcomes_pre_phase_3c_backup;
   DELETE FROM daily_features WHERE symbol IN ('MNQ','MES','MGC');
   INSERT INTO daily_features SELECT * FROM daily_features_pre_phase_3c_backup;
   ```
   Then investigate the failure and restart.

3. **Mode A holdout integrity.** The rebuild writes to 2026 dates in
   `orb_outcomes` and `daily_features`. This is NOT a holdout
   violation because pipeline data builds are not gated by
   `HOLDOUT_SACRED_FROM` — only discovery (`strategy_discovery.py`)
   and validation (`strategy_validator._check_mode_a_holdout_integrity`)
   are. The new 2026 rows are deterministic transformations of bars_1m,
   not discovery sweeps. `validated_setups` is explicitly untouched.

4. **Broken existing tests.** If any test relies on the pre-rebuild
   row counts, it will fail after the rebuild. Spot-check
   `tests/test_pipeline/test_check_drift_db.py` and any DB-dependent
   test before committing.

5. **strategy_fitness.py reads orb_outcomes** for live monitoring.
   After the rebuild, fitness numbers for deployed lanes will shift
   because the underlying data changed. The 5 deployed
   `topstep_50k_mnq_auto` lanes are not themselves at risk (live_journal.db
   holds actual trade PnL), but the "how is this lane doing" queries
   will reflect new numbers.

## How to resume in a fresh terminal

```bash
cd C:/Users/joshd/canompx3
git checkout phase-3c-rebuild-orb-outcomes   # this branch
git log --oneline -1                         # should be bc6df12 (main parent)
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
print('backups:')
for t in ['orb_outcomes_pre_phase_3c_backup', 'daily_features_pre_phase_3c_backup']:
    try:
        n = con.sql(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        print(f'  {t}: {n} rows — OK')
    except Exception as e:
        print(f'  {t}: MISSING — {e}')
con.close()
"
```

Backups verified → proceed with Step A (DELETE orphaned) → Step B
(DELETE post-launch) → Step C (rebuild daily_features) → Step D
(rebuild orb_outcomes) → Step E (validate) → Step F (commit).

Or pause before any destructive step and get explicit approval for
each — institutional rigor does not require speed.

## Outstanding questions for the user

None blocking. Good to execute when ready.

## Files in working tree at pause time

Committed on this branch: nothing yet (phase-3c-rebuild-orb-outcomes
has 0 commits).

Untracked but mine (this handover — the only artifact to commit before
running the rebuild):
- `docs/handoffs/2026-04-08-phase-3c-rebuild-handover.md` (this file)

Untracked / other-terminal / bot (leave alone):
- M HANDOFF.md, docs/ralph-loop/*, live_journal.db
- ?? docs/runtime/baselines/, gold.db.pre-e2-fix.bak, gold_snap.db

## Branch state

```
* phase-3c-rebuild-orb-outcomes  bc6df12 Merge branch 'phase-3b-requires-micro-data'
  main                           bc6df12 Merge branch 'phase-3b-requires-micro-data'
```

Zero commits ahead of main on this branch (the handover doc is the
first to commit once you're ready to start).
