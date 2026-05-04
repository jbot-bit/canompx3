---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Three hardening fixes — design (2026-04-20)

## Scope

Three infrastructure gaps surfaced in the RULE 14 retroactive heterogeneity audit (commit aa3399b3) are closed here so the same class of silent failure cannot recur.

1. `ALL_FILTERS` entry registered and routed despite a 0%-populated required column (PIT_MIN today; no barrier against future repeats).
2. `pit_range_atr` backfill never committed — the 2026-04-06 "F5 deployment" commit f776bc13 added schema + ingester + filter + research script but no column-populator. Ghost deployment.
3. "Universal" claims in memory/docs carrying no per-cell breakdown — the pooled-finding class of error documented in `docs/audit/results/2026-04-20-heterogeneity-audit-phase3-results.md`.

## Ground truth (execution-verified before design)

| Fact | Source | Value |
|---|---|---|
| `daily_features.pit_range_atr` populated rows | `SELECT COUNT(pit_range_atr) FROM daily_features` | 0 of 35,112 |
| `exchange_statistics` coverage | `SELECT symbol, COUNT(*) FROM exchange_statistics GROUP BY symbol` | MES 4723 / MGC 4788 / MNQ 4792 rows, 2010-06-06 → 2026-04-03 |
| Column in schema | `pipeline/init_db.py:344` | `pit_range_atr DOUBLE` — exists |
| Filter registered | `trading_app/config.py:3289-3293` | `PIT_MIN` in ALL_FILTERS |
| Filter routed | `trading_app/config.py:3512` | `filters["PIT_MIN"] = ALL_FILTERS["PIT_MIN"]` at CME_REOPEN |
| Filter fail-closed on NULL | `trading_app/config.py:2337-2339` | `if val is None: return False` |
| Original deploy commit | `git show f776bc13 --stat` | 4 files — no backfill script |
| Canonical formula | `scripts/research/exchange_range_t2t8.py:152` | `(prev_cal_date.session_high - session_low) / atr_20` |
| Join semantics | `exchange_range_t2t8.py:98, 142-147` | shift(1) on cal_date-sorted stats within symbol; join `trading_day == cal_date` |
| Look-ahead clean | CME pit closes 21:15 UTC (cal_date T-1); CME_REOPEN starts 23:00 UTC (start of Brisbane trading day T) | confirmed |

Memory file `exchange_range_signal.md` claim "backfill code does not exist in repo" is correct. Historical pass rates (83/82/78%) must have come from an uncommitted one-off SQL.

## Design

### Stage 1 — populate `pit_range_atr` (close ghost deployment)

New module `pipeline/backfill_pit_range_atr.py`, idempotent and re-runnable. Serves both the one-shot backfill AND the forward-flow — `pipeline/build_daily_features.py` calls it at the end of a build to enrich the just-written date range.

Formula (mirrors research script exactly, zero look-ahead):

```
for each (trading_day = T, symbol = S, orb_minutes ∈ {5, 15, 30}) in daily_features
  let prev_cal_date = MAX(es.cal_date) where es.symbol = S AND es.cal_date < T
  let prev_high     = es.session_high at (prev_cal_date, S)
  let prev_low      = es.session_low  at (prev_cal_date, S)
  let atr           = daily_features.atr_20 at (T, S, orb_minutes)
  if prev_high, prev_low, atr all non-NULL and atr > 0:
      pit_range_atr := (prev_high - prev_low) / atr
  else:
      pit_range_atr := NULL  (fail-closed on missing upstream)
```

Scope: `ACTIVE_ORB_INSTRUMENTS` only (MES/MGC/MNQ). Non-active instruments left NULL by design — they are not in the live universe.

All three `orb_minutes` (5, 15, 30) get the same per-day value — pit_range is a day-level feature not aperture-specific.

Expected coverage post-backfill: ≥95% for active instruments (some 2019 Q1 MES history before data launch + US holidays will remain NULL per research file).

Idempotent semantics: single-statement `UPDATE daily_features SET pit_range_atr = (computed value)` — re-run gives identical output; no INSERT path needed since rows already exist.

### Stage 2 — drift check: routed filters require populated columns

New check `check_routed_filter_columns_populated(con)` in `pipeline/check_drift.py`.

Mechanic:
- Collect the ROUTED set: every `filter_type` string that appears in any session's output from `trading_app.config.get_session_filters` across `pipeline.dst.SESSION_CATALOG`. This is the production-contract set.
- For each routed filter, walk `ALL_FILTERS[ft]` through composites to leaves. Call `leaf.describe(sample_row, session, entry_model)` and collect every `atom.feature_column` that names a scalar daily_features column (excludes `orb_<session>_*` per-aperture columns — those are session-conditional by design).
- For each unique scalar column: compute population fraction in `daily_features` scoped to `ACTIVE_ORB_INSTRUMENTS`.
- Flag when fraction < 0.50. Threshold chosen to catch 0%-populated ghost deployments while tolerating warm-up NULLs and early-history gaps. Tighter than 5% first proposed — 50% provides real signal given ≥15 years of clean data.

Delegation: does not re-encode column-requirement logic; uses the canonical `describe()` → `feature_column` surface established by commit b6d2d21d (canonical filter self-description foundation). Per `.claude/rules/research-truth-protocol.md` Canonical Filter Delegation rule.

### Stage 3 — pooled-finding annotation schema

Prevents future RULE 14 violations mechanically.

Three artefacts:

1. `.claude/rules/pooled-finding-rule.md` — locks the rule: any new file under `docs/audit/results/` claiming a pooled-universe finding must carry three YAML front-matter fields: `pooled_finding: true`, `per_cell_breakdown_path: <repo-relative path>`, `flip_rate_pct: <0-100 number>`. 25% threshold reference from aa3399b3.
2. `docs/audit/results/TEMPLATE-pooled-finding.md` — authoring skeleton with the front-matter stub and a minimal body showing where per-cell evidence goes.
3. New check `check_pooled_finding_annotations()` in `pipeline/check_drift.py`. Scans `docs/audit/results/*.md` where file git-add date is after sentinel `2026-04-20`. Parses YAML front-matter. Violation if (a) `pooled_finding: true` and one of the companion fields missing; (b) `flip_rate_pct >= 25` without explicit `heterogeneity_ack: true` field acknowledging the heterogeneity finding. Historical files modified before the sentinel are exempt; opt-in retrofit by editing a file after the sentinel.

## Hardening / future-proofing summary

| Class of regression | Caught by |
|---|---|
| A new filter registered + routed with its column 0%-populated | Stage 2 drift check |
| A backfill that was one-off-SQL and never committed again | Stage 2 drift check (population would drift toward 0% over time without a writer; check catches <50%) |
| A pooled universality claim written without per-cell evidence | Stage 3 drift check |
| A pooled claim with ≥25% cell flip rate missing heterogeneity flag | Stage 3 drift check |

## Blast radius

- `pipeline/backfill_pit_range_atr.py` — new, ~150 LOC
- `pipeline/build_daily_features.py` — 1 call site added at end of `build_daily_features()` (~line 1621)
- `pipeline/ingest_statistics.py` — 2-line docstring correction
- `pipeline/check_drift.py` — 2 new check functions + wire into main registry
- `tests/test_pipeline/test_check_drift_db.py` — 2 new test fns
- `.claude/rules/pooled-finding-rule.md` — new, ~60 LOC
- `docs/audit/results/TEMPLATE-pooled-finding.md` — new, ~30 LOC
- `~/.claude/.../memory/exchange_range_signal.md` — status update to RESOLVED

Total: 3 new files + 4 file edits + 1 memory update. Within sensible stage bounds.

## Acceptance criteria

All of:
- `python pipeline/backfill_pit_range_atr.py --all` exits 0 and reports ≥95% coverage for MES/MGC/MNQ
- `SELECT COUNT(pit_range_atr) / COUNT(*) FROM daily_features WHERE symbol IN ('MES','MGC','MNQ')` returns ≥0.95
- `python pipeline/check_drift.py` exits 0 with both new checks enrolled
- New regression tests pass (stub filter with empty column fires Stage 2; fixture audit file with missing front-matter fires Stage 3)
- Memory file `exchange_range_signal.md` updated

## Rollback

Per-stage.
- Stage 1: DROP/re-NULL the column. `UPDATE daily_features SET pit_range_atr = NULL`. Revert build_daily_features hook.
- Stage 2: delete the check function.
- Stage 3: delete the check + rule + template.

## Literature grounding

No new thresholds introduced. Stage 2 threshold (50% populated) is pragmatic, not literature-derived. Stage 3 25% cell-flip threshold pins to the 2026-04-20 RULE 14 commit which derives from the heterogeneity audit — itself pinned to Harvey-Liu 2015 cross-section.
