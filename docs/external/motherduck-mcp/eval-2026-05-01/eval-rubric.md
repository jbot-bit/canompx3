# MotherDuck MCP Eval Rubric (2026-05-01)

**Premise:** when a research question doesn't fit a curated `gold-db` MCP template, the
current fallback is hand-written Python with a DuckDB connection. That path costs tokens
(boilerplate, retries, schema lookups) and wall-clock time. The hypothesis under test is
that `mcp-server-motherduck` pointed at a read-only snapshot (`gold.db.eval`) lets Claude
execute ad-hoc SQL inline, reducing both.

**Snapshot under test:** `C:/Users/joshd/canompx3/gold.db.eval` (read-only copy of
`gold.db` taken 2026-05-01, `bars_1m` row count = 20,513,435 verified).

**Verdict thresholds (see `discipline-checklist.md`):** GO requires ≥30% token reduction
on ≥3 of 5 questions, no correctness regressions, and all four discipline checks pass.

---

## Q1 (2026-04-27 finding) — Stage-1 substrate-weak diagnostic

**Question (as asked):**
> "Of the 6 deployed lanes, which had Stage-1 substrate-weak pass results vs fail in the
> sizing-substrate K=48 diagnostic? Show lane name, feature_set tier (a/b), and pass/fail
> status."

**Current path:** raw Python — no curated template covers Stage-1 substrate diagnostics.
Operator opens `gold.db` via `pipeline.paths.GOLD_DB_PATH`, reads
`docs/audit/results/2026-04-27-*.md` to recall column names, then runs an ad-hoc query
joining `validated_setups` (or the diagnostic results table — operator must scan schema)
against the lane allocator JSON. Typical 4–6 turn loop.

**MotherDuck MCP path (SQL):**
```sql
-- Resolve deployed lane strategy_ids from lane_allocation.json (passed inline by Claude).
WITH deployed AS (
  SELECT unnest([
    /* 6 strategy_ids from trading_app/lane_allocation.json */
    'MNQ_NYSE_OPEN_15_E2_C1_F_OVNRNG_50_FAST10_RR2',
    'MNQ_NYSE_OPEN_15_E2_C1_F_BULLDAY_RR2',
    'MNQ_LONDON_15_E2_C1_F_NONE_RR2',
    'MNQ_ASIA_15_E2_C1_F_NONE_RR2',
    'MNQ_SYDNEY_15_E2_C1_F_NONE_RR2',
    'MNQ_CME_REOPEN_15_E2_C1_F_NONE_RR2'
  ]) AS strategy_id
)
SELECT
  d.strategy_id,
  vs.feature_set_tier,         -- 'a' or 'b'
  vs.substrate_stage1_status,  -- 'PASS' / 'FAIL' / 'WEAK'
  vs.substrate_k,
  vs.substrate_p_value
FROM deployed d
LEFT JOIN validated_setups vs USING (strategy_id)
WHERE vs.diagnostic_run = 'sizing_substrate_k48_2026_04_27'
ORDER BY vs.substrate_stage1_status, d.strategy_id;
```

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

**Current path:** raw Python. No curated template. Operator typically opens
`docs/audit/results/2026-04-21-pr48-*.md`, finds the result table, transcribes the
numbers. Or re-runs the analysis script `research/sizer_paired_t.py` if the doc is
stale. 3–4 turn loop, longer if re-running.

**MotherDuck MCP path (SQL):**
```sql
-- Paired-t per instrument: baseline R vs PR#48-sized R per trade, joined by trade_id.
WITH paired AS (
  SELECT
    pt.instrument,
    pt.trade_id,
    pt.pnl_r            AS r_baseline,
    pt.pnl_r_pr48_sized AS r_sized,
    pt.pnl_r_pr48_sized - pt.pnl_r AS delta_r
  FROM paper_trades pt
  WHERE pt.entry_ts >= '2020-01-01'  -- pre-holdout sample for PR #48 lock
    AND pt.entry_ts <  '2026-01-01'
    AND pt.pnl_r_pr48_sized IS NOT NULL
)
SELECT
  instrument,
  COUNT(*)                          AS n_trades,
  AVG(delta_r)                      AS delta_r_per_trade,
  AVG(delta_r) / (STDDEV(delta_r) / SQRT(COUNT(*))) AS t_stat,
  -- p_value computed in client; SQL exposes degrees of freedom
  COUNT(*) - 1                      AS df,
  CASE
    WHEN AVG(delta_r) > 0 AND ABS(AVG(delta_r) / (STDDEV(delta_r)/SQRT(COUNT(*)))) > 2.0 THEN 'SIZER_ALIVE'
    WHEN AVG(delta_r) > 0 THEN 'SIZER_WEAK'
    ELSE 'DEAD'
  END AS verdict
FROM paired
GROUP BY instrument
ORDER BY instrument;
```

**Eval metrics:** `tokens_current` / `tokens_motherduck` / `time_current_s` /
`time_motherduck_s` / `correctness` (PASS/FAIL/PARTIAL) — TBD.

---

## Q3 (2026-04-21 finding) — MES+MGC filter-overlay family scan cell breakdown

**Question (as asked):**
> "How many cells of the MES+MGC filter-overlay family scan (PR-something post-#53) hit
> CANDIDATE_READY vs RESEARCH_SURVIVOR vs KILL_IS, broken down by RR target?"

**Current path:** raw Python. Operator reads the scan output parquet/CSV under
`research/scratch/` or queries `experimental_strategies` if landed. ~5 turns including
schema discovery.

**MotherDuck MCP path (SQL):**
```sql
SELECT
  regexp_extract(strategy_id, '_RR([0-9]+)$', 1)::INT AS rr_target,
  fitness_status,
  COUNT(*) AS n_cells
FROM experimental_strategies
WHERE scan_run = 'mes_mgc_filter_overlay_post_pr53_2026_04_21'
  AND instrument IN ('MES', 'MGC')
  AND fitness_status IN ('CANDIDATE_READY', 'RESEARCH_SURVIVOR', 'KILL_IS')
GROUP BY rr_target, fitness_status
ORDER BY rr_target, fitness_status;
```

**Eval metrics:** TBD (see Q1 schema).

---

## Q4 (2026-04-20 finding) — Deployed MNQ lanes 2026 OOS dir_match + power

**Question (as asked):**
> "List the 6 currently-deployed MNQ lanes with their 2026 OOS dir_match status and OOS
> power. Sort by power ascending."

**Current path:** curated template `get_strategy_fitness` partially covers this, BUT the
template returns FIT/WATCH/DECAY/STALE per strategy — it does NOT expose
`dir_match_status` and `oos_power` columns directly. So operator falls back to raw Python
to read `validated_setups` or `deployable_validated_setups`, ~3 turns.

**MotherDuck MCP path (SQL):**
```sql
SELECT
  strategy_id,
  dir_match_status_2026,    -- e.g. 'CONFIRMED' / 'UNVERIFIED' / 'FLIPPED'
  oos_power_2026,           -- e.g. 0.12, 0.08
  oos_n_trades_2026
FROM deployable_validated_setups
WHERE instrument = 'MNQ'
  AND deployed = TRUE
ORDER BY oos_power_2026 ASC;
```

**Eval metrics:** TBD.

---

## Q5 (2026-04-20 finding) — Deployed MNQ lanes pairwise return correlation

**Question (as asked):**
> "What's the average pairwise trade-return correlation across the 6 deployed MNQ lanes,
> and what was the max pairwise correlation? Compute from `paper_trades` if it exists,
> else from `orb_outcomes` filtered to those lanes' strategy_ids."

**Current path:** raw Python — pivots returns by trading_day × strategy_id, runs
`pandas.DataFrame.corr()`, extracts upper triangle. ~6–8 turns, often requires a second
turn to clean NaNs.

**MotherDuck MCP path (SQL):**
```sql
-- Daily-aggregated R per lane, pivoted, then corr() pairwise via DuckDB.
WITH deployed_ids AS (
  SELECT unnest([
    'MNQ_NYSE_OPEN_15_E2_C1_F_OVNRNG_50_FAST10_RR2',
    'MNQ_NYSE_OPEN_15_E2_C1_F_BULLDAY_RR2',
    'MNQ_LONDON_15_E2_C1_F_NONE_RR2',
    'MNQ_ASIA_15_E2_C1_F_NONE_RR2',
    'MNQ_SYDNEY_15_E2_C1_F_NONE_RR2',
    'MNQ_CME_REOPEN_15_E2_C1_F_NONE_RR2'
  ]) AS strategy_id
),
daily_r AS (
  SELECT
    pt.trading_day,
    pt.strategy_id,
    SUM(pt.pnl_r) AS daily_r
  FROM paper_trades pt
  JOIN deployed_ids USING (strategy_id)
  WHERE pt.entry_ts >= '2020-01-01'
  GROUP BY pt.trading_day, pt.strategy_id
),
pairs AS (
  SELECT
    a.strategy_id AS lane_a,
    b.strategy_id AS lane_b,
    corr(a.daily_r, b.daily_r) AS rho
  FROM daily_r a
  JOIN daily_r b
    ON a.trading_day = b.trading_day
   AND a.strategy_id < b.strategy_id
  GROUP BY a.strategy_id, b.strategy_id
)
SELECT
  AVG(rho) AS avg_pairwise_corr,
  MAX(rho) AS max_pairwise_corr,
  COUNT(*) AS n_pairs
FROM pairs;
```

If `paper_trades.pnl_r` is sparse for these lanes, the fallback substitutes
`orb_outcomes` filtered to the 6 strategy_ids — same shape, swap the source CTE.

**Eval metrics:** TBD.

---

## Aggregate scoring

| Q  | tokens_current | tokens_motherduck | reduction % | correctness |
|----|----------------|-------------------|-------------|-------------|
| Q1 | TBD            | TBD               | TBD         | TBD         |
| Q2 | TBD            | TBD               | TBD         | TBD         |
| Q3 | TBD            | TBD               | TBD         | TBD         |
| Q4 | TBD            | TBD               | TBD         | TBD         |
| Q5 | TBD            | TBD               | TBD         | TBD         |

**GO criteria:** ≥30% token reduction on ≥3 of 5 questions AND zero correctness
regressions AND all four `discipline-checklist.md` checks PASS.
