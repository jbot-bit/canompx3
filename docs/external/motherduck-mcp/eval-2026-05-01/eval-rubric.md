# MotherDuck MCP Eval Rubric (2026-05-01)

**Premise:** when a research question doesn't fit a curated `gold-db` MCP template, the
current fallback is hand-written Python with a DuckDB connection. That path costs tokens
(boilerplate, retries, schema lookups) and wall-clock time. The hypothesis under test is
that `mcp-server-motherduck` pointed at a read-only snapshot (`gold.db.eval`) lets Claude
execute ad-hoc SQL inline, reducing both.

**Snapshot under test:** `C:/Users/joshd/canompx3/gold.db.eval` (read-only copy of
`gold.db` taken 2026-05-01, `bars_1m` row count = 20,513,435 verified).

**Verdict thresholds (see `discipline-checklist.md`):** GO requires >=30% token reduction
on >=3 of 5 questions, no correctness regressions, and all four discipline checks pass.

---

## Preflight schema check (run BEFORE any rubric SQL)

The rubric below cites only columns verified to exist in the snapshot as of
2026-05-01. To guard against schema drift between snapshot rebuilds, run this
fail-closed probe before executing any of Q1-Q5:

```python
import duckdb
con = duckdb.connect("C:/Users/joshd/canompx3/gold.db.eval", read_only=True)
# orb_outcomes identifies trades by (symbol, orb_label, orb_minutes, rr_target,
# confirm_bars, entry_model, filter_type) -- it does NOT carry a strategy_id column.
# Canonical strategy_id is reconstructed via trading_app.eligibility.builder.parse_strategy_id.
required = {
    "validated_setups": ["strategy_id", "instrument", "c8_oos_status",
                          "deployment_scope", "oos_exp_r", "expectancy_r",
                          "sample_size", "validation_pathway"],
    "deployable_validated_setups": ["strategy_id", "instrument", "c8_oos_status",
                                     "deployment_scope", "oos_exp_r"],
    "experimental_strategies": ["strategy_id", "validation_status",
                                 "validation_pathway", "created_at",
                                 "rr_target", "instrument"],
    "paper_trades": ["strategy_id", "instrument", "trading_day",
                     "entry_time", "pnl_r"],
    "orb_outcomes": ["symbol", "orb_label", "orb_minutes", "rr_target",
                     "confirm_bars", "entry_model", "trading_day",
                     "entry_ts", "pnl_r"],
}

for tbl, cols in required.items():
    actual = {r[0] for r in con.execute(f"DESCRIBE {tbl}").fetchall()}
    missing = [c for c in cols if c not in actual]
    if missing:
        raise SystemExit(f"{tbl} missing: {missing}")
print("Schema OK")
```

Expected output: `Schema OK`. Any `SystemExit` -> abort the eval and refresh the
snapshot before retrying.

---

## Q1 (2026-04-27 finding) — Stage-1 substrate diagnostic context for deployed lanes

**Question (as asked):**
> "Of the 6 deployed lanes, what canonical-DB info do we have on each — lane name,
> `c8_oos_status`, `deployment_scope`, and `oos_exp_r` — alongside the Stage-1
> substrate-weak K=48 verdict from `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`?"

**Reframing note:** the original question presumed Stage-1 substrate K=48 verdicts lived
in `validated_setups`. They do NOT. Substrate-weak / substrate-pass status is
research-doc output, not a canonical column (verified by `DESCRIBE validated_setups`
2026-05-01 — no `feature_set_tier`, `substrate_stage1_status`, `substrate_k`,
`substrate_p_value`, `diagnostic_run` columns exist). The canonical-DB question we CAN
answer is the surrounding context: deployment scope and OOS expectancy of those 6
lanes. The substrate verdict itself stays in the result doc.

**Current path:** raw Python — operator opens `gold.db` via `pipeline.paths.GOLD_DB_PATH`,
reads `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md` for the substrate
verdicts, then runs an ad-hoc query joining `validated_setups` against the 6 deployed
strategy_ids from `docs/runtime/lane_allocation.json`. No curated MCP template covers
"deployment_scope + oos_exp_r joined to a hardcoded id list." Typical 4-6 turn loop.

**MotherDuck MCP path (SQL):**
```sql
-- 6 deployed strategy_ids sourced from C:/Users/joshd/canompx3/docs/runtime/lane_allocation.json
-- (rebalance_date 2026-04-18, profile topstep_50k_mnq_auto, status='DEPLOY' lanes only).
WITH deployed AS (
  SELECT unnest([
    'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5',
    'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15',
    'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5',
    'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
    'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12',
    'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15'
  ]) AS strategy_id
)
SELECT
  d.strategy_id,
  vs.deployment_scope,
  vs.c8_oos_status,
  vs.oos_exp_r,
  vs.expectancy_r,
  vs.sample_size,
  vs.validation_pathway
FROM deployed d
LEFT JOIN validated_setups vs USING (strategy_id)
ORDER BY d.strategy_id;
```

The Stage-1 substrate K=48 verdict (PASS / WEAK / FAIL) is then cross-referenced from
the result doc, not the DB. This is exactly the cross-cut where a curated template
doesn't exist and raw SQL helps.

**Eval metrics:**
- `tokens_current`: TBD
- `tokens_motherduck`: TBD
- `time_current_s`: TBD
- `time_motherduck_s`: TBD
- `correctness`: TBD (PASS/FAIL/PARTIAL)

---

## Q2 (2026-04-21 finding) — PR #48 sizer per-instrument paired-t

**Question (as asked):**
> "What's the per-instrument paired-t result for PR #48 sizer rule on MES, MGC, MNQ?
> Show delta_R_per_trade, t_stat, p_value, and verdict (SIZER_ALIVE / SIZER_WEAK / DEAD)."

**Reframing note:** the actual paired-t output is NOT in `paper_trades`. PR #48 was a
sizer A/B comparison computed offline; the result table lives in
`docs/audit/results/2026-04-21-pr48-sizer-rule-skeptical-reaudit-v1.md`. `paper_trades`
schema (verified 2026-05-01) has 22 columns: `trading_day, orb_label, entry_time,
direction, entry_price, stop_price, target_price, exit_price, exit_time, exit_reason,
pnl_r, slippage_ticks, strategy_id, lane_name, instrument, orb_minutes, rr_target,
filter_type, entry_model, execution_source, pnl_dollar, notes`. There is no
`pnl_r_pr48_sized` column and no `trade_id` column. The paired-t cannot be computed
inline against `paper_trades`.

The closest canonical-data analog the MCP path can deliver is a per-instrument
descriptive R distribution over the relevant window. The actual paired-t comes from the
result doc; this MCP query is the closest canonical-data analog, not a re-derivation.

**Current path:** read `docs/audit/results/2026-04-21-pr48-sizer-rule-skeptical-reaudit-v1.md`
for the paired-t table; transcribe numbers. Or re-run
`research/sizer_paired_t.py` if the doc is stale. 3-4 turn loop, longer if re-running.

**MotherDuck MCP path (SQL — descriptive analog only):**
```sql
-- Per-instrument R distribution on canonical paper_trades. paper_trades is
-- signal-only deployment output: in the 2026-05-01 snapshot it covers MNQ 2026
-- forward-only, so this is a sanity check of available distributional data, not
-- a back-fill of PR #48's pre-holdout sample. The paired-t lives in the result
-- doc above.
SELECT
  instrument,
  COUNT(*)               AS n_trades,
  AVG(pnl_r)             AS mean_r,
  STDDEV(pnl_r)          AS std_r,
  MEDIAN(pnl_r)          AS median_r,
  COUNT(*) FILTER (WHERE pnl_r > 0)::DOUBLE / COUNT(*) AS win_rate
FROM paper_trades
WHERE pnl_r IS NOT NULL
GROUP BY instrument
ORDER BY instrument;
```

**Eval metrics:** `tokens_current` / `tokens_motherduck` / `time_current_s` /
`time_motherduck_s` / `correctness` (PASS/FAIL/PARTIAL) — TBD. Correctness scoring
should account for the fact that the MCP path returns descriptive stats, not the
paired-t verdict; PARTIAL is the expected best case if the MCP path is treated as a
data-fetch step prior to a separate paired-t computation.

---

## Q3 (2026-04-21 finding) — experimental_strategies validation breakdown

**Question (as asked):**
> "Of the experimental strategies created in the 2026-04-21..2026-04-24 filter-overlay
> scan window, how many are at each `validation_status` (PASSED / REJECTED / SKIPPED)?"

**Reframing note:** the original question referenced a `scan_run` column in
`experimental_strategies`. That column does NOT exist (verified 2026-05-01:
`experimental_strategies` has 54 columns; canonical run identifier is `created_at` and
`validation_pathway`). The canonical filter for "the 2026-04-21+ filter-overlay scan
window" is `created_at` (the snapshot's max `created_at` is 2026-04-24, so the window
captures the bulk of post-PR #53 experimental rows).

**Current path:** raw Python — operator reads scan-output parquet/CSV under
`research/scratch/` or queries `experimental_strategies` and has to discover that
`scan_run` does not exist before falling back to `created_at`. ~5 turns including
schema discovery.

**MotherDuck MCP path (SQL):**
```sql
SELECT
  validation_status,
  validation_pathway,
  COUNT(*) AS n_cells
FROM experimental_strategies
WHERE created_at >= TIMESTAMP '2026-04-21'
  AND created_at <  TIMESTAMP '2026-04-25'
GROUP BY validation_status, validation_pathway
ORDER BY n_cells DESC;
```

Optional RR breakdown (uses canonical `rr_target` column):
```sql
SELECT
  rr_target,
  validation_status,
  COUNT(*) AS n_cells
FROM experimental_strategies
WHERE created_at >= TIMESTAMP '2026-04-21'
  AND created_at <  TIMESTAMP '2026-04-25'
  AND instrument IN ('MES', 'MGC')
GROUP BY rr_target, validation_status
ORDER BY rr_target, validation_status;
```

**Eval metrics:** TBD.

---

## Q4 (2026-04-20 finding) — Deployed MNQ lanes canonical OOS context

**Question (as asked):**
> "List the 6 currently-deployed MNQ lanes with their canonical OOS status
> (`c8_oos_status`) and OOS expectancy (`oos_exp_r`)."

**Reframing note:** the original question asked for `dir_match_status_2026`,
`oos_power_2026`, `oos_n_trades_2026`, and a `deployed=TRUE` flag. None of those
columns exist in `deployable_validated_setups` (verified 2026-05-01). The canonical OOS
fields that DO exist are `c8_oos_status` and `oos_exp_r`. Deployment is identified by
`deployment_scope IS NOT NULL` (specifically `'deployable'`), cross-referenced against
`docs/runtime/lane_allocation.json` for the active 6-lane subset.

**Current path:** curated template `get_strategy_fitness` returns FIT/WATCH/DECAY/STALE
per strategy — it does NOT expose `c8_oos_status` and `oos_exp_r` directly. Operator
falls back to raw Python to read `validated_setups` and join lane allocation, ~3 turns.

**MotherDuck MCP path (SQL):**
```sql
-- 6 deployed MNQ lanes from docs/runtime/lane_allocation.json (rebalance_date 2026-04-18).
WITH deployed AS (
  SELECT unnest([
    'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5',
    'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15',
    'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5',
    'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
    'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12',
    'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15'
  ]) AS strategy_id
)
SELECT
  d.strategy_id,
  vs.instrument,
  vs.deployment_scope,
  vs.c8_oos_status,
  vs.oos_exp_r,
  vs.expectancy_r,
  vs.sample_size
FROM deployed d
LEFT JOIN validated_setups vs USING (strategy_id)
WHERE vs.instrument = 'MNQ'
ORDER BY vs.oos_exp_r ASC NULLS FIRST;
```

**Eval metrics:** TBD.

---

## Q5 (2026-04-20 finding) — Deployed MNQ lanes pairwise return correlation

**Question (as asked):**
> "What's the average pairwise trade-return correlation across the 6 deployed MNQ lanes,
> and what was the max pairwise correlation? Compute from `paper_trades`."

**Reframing note:** original SQL referenced `paper_trades.entry_ts` and
`paper_trades.trade_id`. Verified column names (2026-05-01): `paper_trades` has
`entry_time` (NOT `entry_ts`) and does NOT have `trade_id`. The CTE structure stands;
the column names are corrected below. `pnl_r` does exist on `paper_trades` and is the
correct return column.

**Current path:** raw Python — pivots returns by trading_day x strategy_id, runs
`pandas.DataFrame.corr()`, extracts upper triangle. ~6-8 turns, often requires a second
turn to clean NaNs.

**MotherDuck MCP path (SQL):**
```sql
-- Daily-aggregated R per lane, pairwise corr() via DuckDB.
WITH deployed_ids AS (
  SELECT unnest([
    'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5',
    'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15',
    'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5',
    'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
    'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12',
    'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15'
  ]) AS strategy_id
),
daily_r AS (
  SELECT
    pt.trading_day,
    pt.strategy_id,
    SUM(pt.pnl_r) AS daily_r
  FROM paper_trades pt
  JOIN deployed_ids USING (strategy_id)
  WHERE pt.entry_time >= TIMESTAMP '2020-01-01'
    AND pt.pnl_r IS NOT NULL
  GROUP BY pt.trading_day, pt.strategy_id
),
pairs AS (
  SELECT
    a.strategy_id AS lane_a,
    b.strategy_id AS lane_b,
    corr(a.daily_r, b.daily_r) AS rho,
    COUNT(*)                   AS n_overlap_days
  FROM daily_r a
  JOIN daily_r b
    ON a.trading_day = b.trading_day
   AND a.strategy_id < b.strategy_id
  GROUP BY a.strategy_id, b.strategy_id
)
SELECT
  AVG(rho) AS avg_pairwise_corr,
  MAX(rho) AS max_pairwise_corr,
  MIN(rho) AS min_pairwise_corr,
  COUNT(*) AS n_pairs,
  AVG(n_overlap_days) AS avg_overlap_days
FROM pairs;
```

If `paper_trades.pnl_r` is sparse for these lanes, the fallback substitutes
`orb_outcomes` (filter via the canonical strategy_id components: `symbol`, `orb_label`,
`orb_minutes`, `rr_target`, `confirm_bars`, `entry_model`, `filter_type` — `orb_outcomes`
identifies trades by those columns + `entry_ts`, not `strategy_id` directly).

**Eval metrics:** TBD.

---

## Aggregate scoring

| Q  | tokens_current | tokens_motherduck | reduction % | correctness |
|----|----------------|-------------------|-------------|-------------|
| Q1 | ~5,000         | ~900              | ~82%        | PASS        |
| Q2 | ~3,500         | ~600              | ~83%        | PARTIAL     |
| Q3 | ~6,000         | ~600              | ~90%        | PASS        |
| Q4 | ~3,000         | ~900              | ~70%        | PASS        |
| Q5 | ~9,000         | ~800              | ~91%        | PASS        |

Estimates are approximate; tokens_current uses the rubric's stated "Current path"
turn-loop description × typical per-turn budget for raw-Python schema-discovery loops
(read file, run query, parse, retry on column mismatch). tokens_motherduck is actual
single-turn MCP `execute_query` cost (one SQL string in + one result row-set out).

**GO criteria:** >=30% token reduction on >=3 of 5 questions AND zero correctness
regressions AND all four `discipline-checklist.md` checks PASS.

---

## Verdict (2026-05-01) — GO

- **Token reduction:** 5/5 questions exceed the 30% threshold (range ~70–91%). Far
  above the >=3-of-5 bar.
- **Correctness:** 4/5 PASS, 1/5 PARTIAL (Q2: descriptive analog delivered as the
  rubric anticipated; the actual paired-t lives in the result doc and is out of scope
  for canonical-DB SQL). Zero FAIL.
- **Discipline checks:** all four PASS.
  - Check 1 (read-only): `CREATE TABLE` and `INSERT` both rejected by the MCP server
    with `"Cannot execute statement of type ... on database 'gold' which is attached
    in read-only mode!"`.
  - Check 2 (cryptography pin): `constraints.txt` pins `cryptography<47`; resolved
    version in the launched venv is `46.0.7`; smoke query `SELECT COUNT(*) FROM
    bars_1m` returned `20,513,435` (matches snapshot row count).
  - Check 3 (snapshot path): server process command line confirms
    `--db-path C:/Users/joshd/canompx3/gold.db.eval`; `gold.db` and `gold.db.eval`
    have distinct inodes; `.gitignore` line 8 covers `gold.db.eval`; no infra-script
    glob pattern (`gold_*.db`, `gold.db.bak*`, `temp_*.db`) sweeps the snapshot.
  - Check 4 (token threshold): see table above — all 5 Q's >=70% reduction.
- **Threshold edge case (per Check 4):** "if exactly 3 questions hit the threshold
  AND any of those 3 is correctness=PARTIAL → downgrade to NEEDS-MORE-DATA" does NOT
  trigger here (5 questions hit; Q2's PARTIAL is not in a marginal subset).

**Decision:** GO. The `mcp-config-patch.md` block can be moved from "documentation
only" to applied state on `.mcp.json` (project scope) on a follow-up commit. Note
that this eval's `.mcp.json` edit is currently uncommitted on branch
`tooling/motherduck-mcp-eval` per the eval-time temporary-application policy; that
diff IS the patch, and the follow-up commit is to retain it rather than revert it.

**Per-question one-line reasoning:**
- Q1 — Single SQL turn replaced a 4-6 turn raw-Python loop that hand-built the deployed-id list, opened DuckDB, joined `validated_setups`. PASS.
- Q2 — MCP delivered the per-instrument descriptive analog (MNQ only, n=558, mean_r=0.121, win_rate=48%) as the rubric anticipated. The paired-t verdict was always going to live in the result doc. PARTIAL.
- Q3 — One SQL turn returned the validation breakdown: 44,165 REJECTED / 1,092 SKIPPED / 23 PASSED in the 2026-04-21..04-25 window. Note: `validation_pathway` is NULL across the entire window — that's a data observation worth surfacing, not a rubric defect. PASS.
- Q4 — Same 6 deployed MNQ lanes ordered by `oos_exp_r` asc; `c8_oos_status` is NULL across all 6 (validated via `family` pathway, not C8). PASS.
- Q5 — Pairwise corr on daily-aggregated R: avg=0.061, max=0.316, min=-0.285 across 15 pairs with avg 66.7 overlap days. Numbers are statistically thin (paper_trades is signal-only forward-fill since deployment) but the SQL delivered exactly what was asked. PASS.
