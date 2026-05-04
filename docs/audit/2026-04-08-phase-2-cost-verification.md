# Phase 2 — Databento Real Micro Redownload — Cost & Scope Verification

**Date:** 2026-04-08
**Stage:** `docs/runtime/stages/phase-2-databento-real-micro-redownload.md`
**Source plan:** `docs/plans/2026-04-07-canonical-data-redownload.md`
**Action queue:** MEMORY.md item 6, Phase 2 portion

## Live cost verification (2026-04-08)

Verified against Databento Historical API (`client.metadata.get_cost` + `get_billable_size`) — NOT from memory or prior heuristic.

Dataset: `GLBX.MDP3`
Schema: `ohlcv-1m`
stype_in: `parent` (parent-of-symbology root, returns the contract series for the given micro)

| Symbol | Real launch date | End date | Window | Size (MB) | Cost (USD) |
|---|---|---|---|---|---|
| `MNQ.FUT` | 2019-05-06 | 2026-04-07 | ~7 years | 196.4 | $0.0000 |
| `MES.FUT` | 2019-05-06 | 2026-04-07 | ~7 years | 187.0 | $0.0000 |
| `MGC.FUT` | 2023-09-11 | 2026-04-07 | ~31 months | 127.5 | $0.0000 |
| **TOTAL** | | | | **510.9** | **$0.0000** |

### Cost confirm threshold
- Per `config/databento_config.yaml`: `cost_confirm_threshold: 5.00` USD
- Total cost ($0.00) is **below** threshold → no manual confirmation required
- ohlcv-1m is FREE on Standard plan per `databento_subscription.md`

### Dataset availability (queried live)
- `GLBX.MDP3.ohlcv-1m` available range: 2010-06-06 → 2026-04-07T23:50:00Z (per `client.metadata.get_dataset_range`)
- End date 2026-04-07 used (today is 2026-04-08, but dataset only updated through end-of-day yesterday)

## Why this redownload exists

Direct query confirmed in this session:

```sql
SELECT source_symbol, COUNT(*) FROM bars_1m WHERE symbol = 'MGC' GROUP BY source_symbol
```

returns `GCZ2`, `GCG3`, `GCJ3`, `GCM3`, ... — every single source_symbol is a GC parent contract (gold full-size, 100oz, $100/pt). Zero real MGC (Micro Gold, 10oz, $10/pt) contracts.

Cause: `config/databento_config.yaml` line 37 uses `symbol: GC.FUT` with `stype_in: parent`, which returns the GC parent series. The `stored_as: MGC` field re-labels it as MGC. Same pattern applies to MNQ (NQ.FUT → labeled as MNQ) and MES (ES.FUT → labeled as MES).

This Phase 2 redownload uses `MNQ.FUT` / `MES.FUT` / `MGC.FUT` (the real micro roots) so Databento returns the actual micro contracts (`MNQH5`, `MNQM5`, `MGCZ4`, etc.).

## Config changes (already applied)

`config/databento_config.yaml` extended with three new entries:
- `mnq_real_micro_full` (priority must_have)
- `mes_real_micro_full` (priority must_have)
- `mgc_real_micro_full` (priority must_have)

The OLD parent entries (`mnq_1m_extension`, `mes_1m_extension`, `mgc_1m_extension`, etc.) are LEFT IN PLACE — they remain bookkeeping for the parent backfill that produced the existing contaminated bars_1m rows. They will need to be marked deprecated AFTER the merge (Phase 2 step 6), but not before.

## Download invocation (already running)

```bash
python scripts/databento_backfill.py --yes \
  --name mnq_real_micro_full mes_real_micro_full mgc_real_micro_full
```

Note: `--name` uses argparse `nargs='+'` (space-separated, single flag), not multiple `--name` flags. Initial attempt with multi-flag syntax failed silently — only the LAST `--name` was kept. Verified the correct syntax with a `--dry-run` first.

## Database schema constraint (relevant for merge step)

`bars_1m` PRIMARY KEY: `(symbol, ts_utc)`. This means:
- Loading new MGC data with `symbol='MGC'` will COLLIDE with existing `symbol='MGC'` parent rows at the same `ts_utc` for any post-launch period (2023-09-11 onwards for MGC).
- For pre-launch periods, no collision (just inserts).
- The merge cannot be a simple `INSERT OR IGNORE` — must explicitly handle the parent rows.

## Merge strategy options (DEFERRED — user decision needed)

The downloads themselves are safe (write to staging files only — no DB writes). The MERGE into `bars_1m` is a real fork:

### Option A — Additive merge (recommended)
1. `UPDATE bars_1m SET symbol = 'NQ' WHERE symbol = 'MNQ' AND source_symbol LIKE 'NQ%'` (and similar for ES/GC)
2. Load new micro data with `symbol = 'MNQ'` (etc.)
3. Result: parent data preserved under symbol='NQ'/'ES'/'GC', new micro under symbol='MNQ'/'MES'/'MGC'
4. Pro: nothing lost; backtest provenance for the 124 grandfathered validated_setups can still be reproduced via the relabeled symbols
5. Con: pipeline code paths that hard-assume `symbol IN ('MNQ','MES','MGC')` would miss the parent data — verify which queries break

### Option B — Replacement merge
1. `DELETE FROM bars_1m WHERE symbol = 'MNQ' AND source_symbol LIKE 'NQ%'` (and similar)
2. Load new micro data
3. Result: only the post-launch micro data remains; parent history gone
4. Pro: simpler downstream queries; no parent codes to handle
5. Con: 124 grandfathered validated_setups become unreproducible; total sample size shrinks; recovery requires re-download

### Option C — Schema change (additional column)
1. Add `data_era` column to bars_1m (parent/micro)
2. Extend PK to `(symbol, ts_utc, data_era)`
3. Load new data with `data_era='micro'`, mark existing with `data_era='parent'`
4. Pro: cleanest separation; both datasets coexist
5. Con: schema change ripples through all downstream queries; large refactor for limited benefit over Option A

**Recommended:** Option A (additive merge with relabel). Reasoning per the canonical-data-redownload plan and IR rule "no destructive operations as a shortcut".

## Open decisions for the user

1. ✅ **Cost approved** — $0.00, well under $5 threshold (decided automatically)
2. ✅ **Scope approved** — 3 micro symbols from launch dates (decided automatically per source plan)
3. ⏳ **Merge strategy** — Option A vs B vs C (DEFERRED until staging files validated)

## Post-download validation checklist

After downloads complete:
- [ ] All three downloads land in `data/raw/databento/` with non-zero file sizes
- [ ] Manifest JSONs exist and report no chunk errors
- [ ] Sample-decode each download with `databento.DBNStore.from_file(...).to_df().head()` to confirm parseable
- [ ] Verify the source_symbols in the staged files start with MNQ/MES/MGC (not NQ/ES/GC)
- [ ] Total file count matches expected chunk count

## Stale state warning (post-merge)

After Phase 2 merge (whichever option), the following downstream layers are STALE and must be rebuilt by Phase 3 before any further work:
- `daily_features` — derived from old parent bars_1m
- `orb_outcomes` — derived from old parent bars_1m
- `experimental_strategies` — derived from old parent outcomes
- `validated_setups` — derived from old parent strategies

Phase 3 (separate stage, NOT in scope here) handles the rebuild chain.
