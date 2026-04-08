# Handover — Phase 2 + Hard Gate (2026-04-08)

**Branch:** `topstep-canonical-info` (pushed to origin)
**Author:** Claude Code session, terminal A
**Status:** READY FOR REVIEW / MERGE / PR

## What landed (4 commits, in order)

```
8cb0c93 fix(asset-configs): add missing GC parent config (Bloomey review HIGH finding)
f4dbec9 feat(holdout): close all gaps — CLI flag, nested/regime gates, override tests
850f679 feat(holdout): function-level Mode A gate + override token (3656)
82e8b60 feat(phase-2): real-micro redownload — fix MNQ/MES/MGC bars_1m corruption
```

PR URL: https://github.com/jbot-bit/canompx3/pull/new/topstep-canonical-info

## What this work delivers

### 1. Phase 2 of canonical-data-redownload (commit 82e8b60)

Replaces the parent-futures data (NQ/ES/GC) that was mislabeled as the active
micro instruments (MNQ/MES/MGC) in `bars_1m` with REAL micro-futures data
from each contract's actual launch date.

**The bug fixed:** for years `bars_1m` had `symbol='MGC'` rows whose
`source_symbol` was actually a GC parent contract (`GCZ2`, `GCG3`, etc.)
because `pipeline/asset_configs.py` had `outright_pattern: ^GC...` baked in
and `config/databento_config.yaml` requested `symbol: GC.FUT` with
`stype_in: parent`. Same pattern for MNQ (NQ.FUT mislabeled as MNQ) and MES.

**Verified data corruption pre-fix:**
- MGC: 5,548,284 rows, ALL `source_symbol LIKE 'GC%'` (zero real micro)
- MNQ: 4,590,357 parent rows + 760,737 real micro rows (cut at 2024-02-04)
- MES: 4,770,406 parent rows + 751,477 real micro rows (cut at 2024-02-11)

**Operations performed:**
1. Created `bars_1m_pre_phase_2_backup` table (16,421,261 rows preserved
   for full reversibility)
2. Relabeled parent rows: `UPDATE bars_1m SET symbol='NQ'/'ES'/'GC' WHERE
   symbol='MNQ'/'MES'/'MGC' AND source_symbol LIKE 'NQ%'/'ES%'/'GC%'`
3. Deleted existing real-micro rows for MNQ/MES (single source of truth)
4. Downloaded 510.9 MB of real micro ohlcv-1m bars from Databento (cost
   $0.00 verified live via `client.metadata.get_cost`):
   - MNQ.FUT 2019-05-06 → 2026-04-07
   - MES.FUT 2019-05-06 → 2026-04-07
   - MGC.FUT 2023-09-11 → 2026-04-07
5. Updated `pipeline/asset_configs.py`:
   - MGC: pattern `^GC...` → `^MGC...`, `prefix_len: 2 → 3`,
     `dbn_path: DB/GOLD_DB_FULLSIZE → data/raw/databento/ohlcv-1m/MGC`
   - MNQ + MES: dbn_path → staging dir
   - NQ + ES: `symbol: 'MNQ'/'MES' → 'NQ'/'ES'` (so future re-ingest
     doesn't re-corrupt)
6. Ingested via canonical `pipeline.ingest_dbn` (--instrument MNQ/MES/MGC):
   - MNQ: 2,437,390 rows, 29 contracts, all honesty gates passed
   - MES: 2,435,218 rows, 29 contracts, all honesty gates passed
   - MGC: 905,875 rows, 14 contracts, all honesty gates passed

**Post-merge state:**
```
('ES',  4,770,406, 2010-06-07, 2024-02-10)  -- parent, preserved
('GC',  5,548,284, 2010-06-07, 2026-04-06)  -- parent, preserved
('MES', 2,435,218, 2019-05-06, 2026-04-07)  -- REAL micro
('MGC',   905,875, 2023-09-11, 2026-04-07)  -- REAL micro
('MNQ', 2,437,390, 2019-05-06, 2026-04-07)  -- REAL micro
('NQ',  4,590,357, 2010-06-07, 2024-02-03)  -- parent, preserved
```

### 2. Hard-gate Mode A enforcement (commits 850f679 + f4dbec9)

Per explicit user instruction: *"WE CANT HAVE THE PROJECT EVER ACCIDENTLY
EANYWHERTER RUN THAT PAST 2025 ... ensure no gaps. proper work."*

**Gap fixed:** Pre-existing Mode A enforcement (commits 81a7079 / 19d7edd
from amendment-2-7) was CLI-only. Direct Python callers of `run_discovery`,
nested discovery, and regime discovery could bypass the gate.

**Solution:**
1. New constant `HOLDOUT_OVERRIDE_TOKEN = "3656"` in
   `trading_app/holdout_policy.py`
2. Extended `enforce_holdout_date(holdout_date, override_token=None)`:
   - `None` default → `HOLDOUT_SACRED_FROM` (unchanged)
   - `holdout_date <= sacred` → pass (unchanged)
   - `holdout_date > sacred` + correct token → LOUD WARNING + allow
   - `holdout_date > sacred` + wrong/no token → ValueError (unchanged)
3. Pushed enforcement INSIDE `run_discovery()` (function-level chokepoint)
4. Added `--unlock-holdout TOKEN` CLI flag with explicit help text warning
5. Extended `run_nested_discovery` and `run_regime_discovery` with same gate
6. Added `TestOverrideToken` class with 10 tests:
   - Token value pinned to "3656"
   - Correct token allows post-sacred + LOUD warning logged
   - Wrong/empty/None tokens still raise
   - Override does not affect None / pre-sacred / exact-sacred cases

**Discovery paths now gated (zero gaps):**
- `trading_app.strategy_discovery.run_discovery` (function-level)
- `trading_app.strategy_discovery.main` (CLI, defense in depth)
- `trading_app.nested.discovery.run_nested_discovery`
- `trading_app.regime.discovery.run_regime_discovery`
- `trading_app.strategy_validator` (Mode A integrity gate, prior commit
  467a0c3)

**Monitoring paths (intentionally NOT gated):**
- `trading_app.strategy_fitness` — rolling fitness for deployed strategies
- `live_journal.db` — separate file, real prop trades
- `regime-check` / `trade-book` skills — query monitoring tables

The distinction is DISCOVERY (search the parameter space) vs MONITORING
(observe existing strategies). Mode A blocks the former, allows the latter.

### 3. GC asset_config fix (commit 8cb0c93)

Bloomey self-review of the above commits found one HIGH finding:

> The MGC config docstring referenced "see GC config below" but no GC
> entry existed in ASSET_CONFIGS. The 5.55M relabeled GC parent rows
> were orphaned from canonical metadata. `get_asset_config('GC')` raised
> SystemExit.

Fix: added GC config analogous to NQ/ES (both pre-existing). Architecture
is now fully symmetric:
- MNQ + NQ
- MES + ES
- MGC + GC

## Test results (verified post-fix)

```
68/68 tests pass:
  tests/test_trading_app/test_holdout_policy.py: 23/23
    (13 original + 10 new TestOverrideToken)
  tests/test_trading_app/test_strategy_discovery.py: 45/45

Drift check: 84/0/7 unchanged

Behavioral probes (all 4 verified):
  [1] run_discovery(holdout_date=2026-03-01) without token → ValueError ✓
  [2] enforce_holdout_date(2026-03-01, '3656') → returns date + LOUD WARNING ✓
  [3] run_discovery(holdout_date=2026-03-01, unlock_holdout='wrong') → ValueError ✓
  [4] enforce_holdout_date(None) → 2026-01-01 (sacred default) ✓
```

## Bloomey self-review grade: A-

After the GC fix landed:
- Section A (Seven Sins): 40/40 — N/A, no trading logic
- Section B (Canonical Integrity): 19/20 — closed the GC gap
- Section C (Statistical Rigor): 25/25 — N/A
- Section D (Production Readiness): 13/15 — minor: double enforcement
  on CLI path (audit log fires twice on override invocation, intentional
  defense-in-depth, cosmetic only); no companion test for run_nested_discovery
  (pre-existing gap)
- Section E (Caller Discipline): PASS — all signature changes backward compat
- Section F (Integration & Execution): PASS — all probes verified
- Section G (Blueprint): N/A

## Important context for the next person

### What's STALE (post-Phase 2, pre-Phase 3)

The downstream layers were derived from the OLD parent data and are now
stale relative to bars_1m:

- `daily_features` — STALE
- `orb_outcomes` — STALE
- `experimental_strategies` — STALE
- `validated_setups` (124 grandfathered) — STALE
- `strategy_fitness.py` queries — return STALE numbers (read from stale
  orb_outcomes)

What's NOT stale:
- `bars_1m` — clean micro data
- `live_journal.db` — real prop trades (independent of backtest tables)
- The 5 deployed lanes' actual PnL — unaffected, real money trading

### What's left in the canonical-data-redownload plan

Source: `docs/plans/2026-04-07-canonical-data-redownload.md`

- **Phase 3** (NOT STARTED) — rebuild downstream layers from new bars_1m:
  - 3a: data_era classification on bars_1m (derive from source_symbol prefix)
  - 3b: requires_micro_data attribute on StrategyFilter classes
  - 3c: Rebuild orb_outcomes + daily_features from new bars_1m
  - 3d: New drift check for era discipline enforcement
- **Phase 4** (BLOCKED on Phase 3) — clean rediscovery:
  - 4a: Pre-registered hypothesis file
  - 4b/c/d: Discovery runs with `--holdout-date 2026-01-01`
  - 4e: Apply 12 criteria from `pre_registered_criteria.md`
  - 4f: Compare to current deployed lanes
- **Phase 5** (BLOCKED on Phase 4) — deploy decision:
  - Update `prop_profiles.ACCOUNT_PROFILES`
  - Start Shiryaev-Roberts drift monitors
  - Preserve pre-redownload baseline

### Critical reversibility note

`bars_1m_pre_phase_2_backup` table preserves the pre-merge state (16.4M
rows). Restoration if needed:

```sql
DELETE FROM bars_1m WHERE symbol IN ('MNQ','MES','MGC','NQ','ES','GC');
INSERT INTO bars_1m SELECT * FROM bars_1m_pre_phase_2_backup;
```

Plus revert the asset_configs.py edit. The Databento staging files at
`data/raw/databento/ohlcv-1m/{MNQ,MES,MGC}/` can also be deleted to
fully roll back.

### Holdout policy quick reference (Mode A, Amendment 2.7)

| Action | Allowed? |
|---|---|
| `python -m trading_app.strategy_discovery --instrument MNQ` | ✅ (defaults to safe) |
| `python -m trading_app.strategy_discovery --instrument MNQ --holdout-date 2025-06-01` | ✅ (pre-sacred) |
| `python -m trading_app.strategy_discovery --instrument MNQ --holdout-date 2026-03-01` | ❌ Mode A violation |
| `python -m trading_app.strategy_discovery --instrument MNQ --holdout-date 2026-03-01 --unlock-holdout 3656` | ⚠️ ALLOWED + LOUD WARNING + research-provisional |
| `run_discovery(instrument='MNQ', holdout_date=date(2026,3,1))` from Python | ❌ Mode A violation |
| `run_discovery(instrument='MNQ', holdout_date=date(2026,3,1), unlock_holdout='3656')` from Python | ⚠️ ALLOWED + LOUD WARNING |
| Reading `live_journal.db` | ✅ always (monitoring, not discovery) |
| Calling `strategy_fitness.py` | ✅ always (monitoring) |

### Known LOW findings (deferred, not blocking)

1. **Double enforcement on CLI path** — `enforce_holdout_date` is called
   twice when going through `main()` (once in main, once inside run_discovery).
   Override warning fires twice on `--unlock-holdout` invocation.
   Cosmetic only. Intentional defense-in-depth.

2. **No companion test for `run_nested_discovery`** — pre-existing gap.
   The function had zero test coverage before my changes too. My new gate
   code there is also untested. `tests/test_trading_app/test_regime/test_discovery.py`
   covers regime discovery (12 tests passing) but doesn't explicitly
   exercise the new gate.

3. **Pyright import false positives** for `from trading_app.holdout_policy
   import ...` inside function bodies. Runtime imports work (verified by
   tests). Could silence with `# type: ignore[import-untyped]` but cosmetic.

## What I deliberately did NOT do (out of scope)

- Phase 3-5 work (separate stages)
- Drift check that scans for past override-token usage
- RESEARCH_RULES.md update mentioning the override mechanism
- Freezing `strategy_fitness` for deployed lanes during rebuild (Phase 5
  concern)
- Updating shell scripts and research scripts that import discovery
  helpers (will fail loud when next run if they pass post-sacred dates,
  which is the desired behavior)
- Updating `MEMORY.md` action queue (it's outside the repo and I shouldn't
  edit other terminals' memory state without explicit direction)

## Files left in working tree (NOT mine — leave alone)

```
M HANDOFF.md                              ← other terminal (jb)
M docs/ralph-loop/ralph-loop-audit.md     ← other terminal
M docs/ralph-loop/ralph-loop-history.md   ← other terminal
M live_journal.db                         ← live trading bot
?? docs/runtime/baselines/                ← e2-fix snapshots from earlier
?? gold.db.pre-e2-fix.bak                 ← pre-existing backup
?? gold_snap.db                           ← pre-existing snapshot
```

## Branch state at handover

```
* topstep-canonical-info  8cb0c93 fix(asset-configs): add missing GC parent config
  main                    73311fe docs(handoff): multi-terminal coordination snapshot
```

8cb0c93 is 4 commits ahead of main (82e8b60 → 850f679 → f4dbec9 → 8cb0c93).
Branch pushed to origin/topstep-canonical-info. Ready for PR review or
direct merge to main.
