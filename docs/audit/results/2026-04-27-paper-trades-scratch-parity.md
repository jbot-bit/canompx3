# paper_trades scratch parity — Stage 7 audit

**Plan stage:** Stage 7 of `docs/runtime/stages/scratch-eod-mtm-canonical-fix.md`
**Policy spec:** `docs/specs/paper_trades_scratch_policy.md`
**Class bug:** `memory/feedback_scratch_pnl_null_class_bug.md`

**Scope:** verify that `paper_trades.pnl_r` is non-NULL for `exit_reason='scratch'` rows on currently-deployed lanes, post Stage 5b rebuild and paper_trade_logger re-run.

**Outcome (verdict):** **PASSED.** Currently-deployed-lane paper_trades scratch population is 100% (14/14). Retired-strategy historical rows (22 NULL) are explicitly carved out by `docs/specs/paper_trades_scratch_policy.md` per the audit trail / no-mutation-of-historical-record principle.

## Reproduction

```sql
-- Pre-Stage-7 baseline (post Stage 5b rebuild but before paper_trade_logger re-run)
SELECT exit_reason, COUNT(*) AS n,
       SUM(CASE WHEN pnl_r IS NULL THEN 1 ELSE 0 END) AS n_null
FROM paper_trades GROUP BY exit_reason ORDER BY n DESC;
-- → loss=278/0_null, win=249/0_null, scratch=36/36_null

-- Re-run paper_trade_logger to ingest rebuilt orb_outcomes for current 6 deployed lanes:
python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto

-- Post Stage-7 state:
SELECT
  CASE WHEN strategy_id IN ('MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5',
                            'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15',
                            'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5',
                            'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
                            'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12',
                            'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15')
    THEN 'DEPLOYED' ELSE 'RETIRED' END AS status,
  exit_reason, COUNT(*) AS n,
  SUM(CASE WHEN pnl_r IS NULL THEN 1 ELSE 0 END) AS n_null
FROM paper_trades GROUP BY 1, 2 ORDER BY 1, 2;
```

| status | exit_reason | n | n_null | % populated |
|---|---|---:|---:|---:|
| DEPLOYED | loss | 201 | 0 | 100% |
| DEPLOYED | scratch | 14 | 0 | **100%** |
| DEPLOYED | win | 204 | 0 | 100% |
| RETIRED | loss | 83 | 0 | 100% |
| RETIRED | scratch | 22 | 22 | **0% (carved out)** |
| RETIRED | win | 56 | 0 | 100% |

## Verdict

Stage 7 **passes** under the `OR` clause of plan acceptance criterion: "≥99% of scratch rows non-NULL **OR** explicit policy doc exists." The policy doc exists at `docs/specs/paper_trades_scratch_policy.md`. The deployed-lane-only metric (100%) exceeds the 99% threshold by a large margin.

## What was NOT done

- No code change to `trading_app/paper_trade_logger.py` — module is already correct given upstream `orb_outcomes` is correct.
- No mutation of legacy retired-strategy NULL rows — preserved as historical audit-trail record per Backtesting Rule 11.
- No new drift check — implicit via `check_orb_outcomes_scratch_pnl` upstream.
- No audit of `trading_app/paper_trader.py` (live streaming-mode writer) — out of scope; live execution forces flat at session end via `risk_manager.py::F-1` and there is no reported live scratch NULL bug.

## Limitations

- The 22 retired-strategy NULL rows are an immutable historical record. If a future analyst pulls aggregate `paper_trades` stats without filtering on active strategies, they may see the NULLs. The policy doc is the documented contract.
- Future re-deployment of any retired strategy auto-heals via the paper_trade_logger idempotent re-run.

## Cross-references

- Stage 5 fix: `trading_app/outcome_builder.py` canonical scratch-EOD-MTM (commit `68ee35f8`).
- Stage 5b rebuild: 9 instrument-aperture combos rebuilt 2026-04-28 in ~36 minutes.
- Stage 6 downstream impact: `docs/audit/results/2026-04-27-canonical-scratch-fix-downstream-impact.md`.
- Policy: `docs/specs/paper_trades_scratch_policy.md`.
