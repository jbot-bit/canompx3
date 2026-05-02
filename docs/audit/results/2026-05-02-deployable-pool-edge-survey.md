# Deployable-pool edge survey — 2026-05-02

**Type:** Read-only canonical-truth survey. No new audits, no new claims.
**Source:** `validated_setups` joined to `lane_allocation.json` rebalance dry-run output.
**Authority:** RESEARCH_RULES.md § Validated Universe Rule; pre_registered_criteria.md.

## Scope

Question: which validated strategies have edge (OOS_ExpR ≥ deployed-book floor)
but are not deployed, and what's blocking each?

Single-pass, K=1, no FDR claim — this is a triage scope, not a discovery scan.

## Verdict

`active + deployment_scope=deployable + fdr_significant=TRUE` returns 56 rows
in `validated_setups`. Live book is 3 (DEPLOY) of those.

**Strict Chordia (t≥3.79 AND all_years_positive=TRUE):** exactly 5 rows.
Three deployed; two not:

| strategy_id | derived_t | IS_ExpR | OOS_ExpR | DSR | dep |
|---|---:|---:|---:|---:|---|
| MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | 4.58 | 0.210 | 0.206 | 0.69 | DEPLOY |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | 4.32 | 0.151 | 0.163 | 0.07 | — |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15 | 4.27 | 0.149 | 0.151 | 0.52 | — |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | 4.32 | 0.172 | 0.149 | 0.22 | DEPLOY |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 3.51 | 0.087 | 0.100 | 0.00 | DEPLOY |

(NYSE_OPEN_COST_LT12 has theory grant Chan Ch7, hurdle 3.00 not 3.79, hence DEPLOY at derived_t=3.51.)

## Higher-OOS candidates blocked by t-hurdle (top 9 by OOS_ExpR)

These all have OOS_ExpR > the lowest deployed lane's OOS (0.100), but
fail strict Chordia. Each row reports the operative blocker.

| strategy_id | N | derT | OOS | yrs_pos | DSR | likely blocker |
|---|---:|---:|---:|:-:|---:|---|
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K | 138 | 3.30 | 0.357 | T | 0.24 | t<3.79; profile (MES not allowed) |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG | 211 | 3.57 | 0.293 | F | 0.86 | t<3.79; era-instable; needs theory grant |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_S075 | 152 | 3.08 | 0.282 | F | 0.13 | t<3.79; era-instable; profile (MES) |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG | 321 | 3.32 | 0.255 | F | 0.62 | t<3.79; era-instable; needs theory grant |
| MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 | 86 | 3.09 | 0.230 | T | 0.50 | t<3.79; profile (MGC not allowed) |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 | 427 | 3.57 | 0.223 | F | 0.51 | t<3.79; era-instable |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG | 303 | 3.65 | 0.221 | F | 0.74 | t<3.79; era-instable; needs theory grant |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | 532 | 3.40 | 0.218 | T | 0.04 | t<3.79; DSR≈0 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG | 324 | 3.74 | 0.218 | F | 0.75 | t<3.79; era-instable; needs theory grant |

## Reproduction

```
.venv/Scripts/python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
print(con.execute('''
SELECT strategy_id, sample_size,
       sharpe_ratio*sqrt(sample_size) AS derived_t,
       expectancy_r, oos_exp_r, all_years_positive, dsr_score
FROM validated_setups
WHERE status='active' AND deployment_scope='deployable' AND fdr_significant=TRUE
ORDER BY oos_exp_r DESC NULLS LAST
LIMIT 25
''').fetchall())"
```

## Caveats

- **Not a multiple-comparison-corrected claim.** This is a single-cell triage
  view across already-validated strategies. The strategies were discovered
  under the pre-Phase-0 brute-force regime (`n_trials_at_discovery` ~30k–37k);
  Bailey-LdP MinBTL violation is documented in `MEMORY.md` and applies to
  every row here.
- **OOS values from `validated_setups.oos_exp_r`** — these were computed at
  promotion time, possibly under Mode B grandfathered windows
  (`research-truth-protocol.md` Mode B warning). Recompute via canonical
  replay before any reopen-deploy decision.
- **DSR ≈ 0 on most candidates** is the structural block from the brute-force
  search. Adding theory grants does NOT fix DSR; only narrower pre-registered
  re-derivation does. Treat OOS lift in this list as "where to look first,"
  not "what to deploy."
- **Profile filter (`topstep_50k_mnq_auto.allowed_instruments={MNQ}`)** keeps
  MES and MGC candidates out regardless of statistics. Reopening MES/MGC
  profiles is a separate decision tracked in `memory/topstep_scaling_corrected_apr15.md`.
- **`all_years_positive=False`** is the per-year stability gate (pre-reg
  Criterion 9). Several rows above are era-dependent — OOS lift may come
  from a single late-window year. Need per-year R distribution before
  treating as deployable.
- **`PD_CLEAR_LONG` / `PD_GO_LONG` / `PD_DISPLACE_LONG`** appear repeatedly
  with high OOS but no theory grant. These are prior-day-context filters;
  `mechanism_priors.md` has `prior_day` listed as R1-FILTER class but no
  literature extract has been written. That's the path to a 3.00 hurdle.

## Higher-EV next threads (ranked)

1. **Theory-grant feasibility for prior-day-context filters.** Read
   `resources/Algorithmic_Trading_Chan.pdf` Ch7 + Carver Ch9-10 for any
   extract grounding prior-day high/low/close/range as session-momentum
   anchors. If found, write the literature extract; the 3.00 hurdle then
   admits 4 of the top-9 candidates above.
2. **Per-year R stratification on the 6 era-instable candidates** to rule
   out single-year drivers per Criterion 9. Cheap; canonical query.
3. **Profile expansion to MES/MGC** — separate scaling track, but unlocks
   the 3 MES/MGC rows in the top-9.
4. (Lowest EV) Continue chordia_audit_unlock siblings — same-session
   correlation-pruned, ranking-pruned even before correlation gate.

## Decision

Status of `chordia_audit_unlock_pass_chordia_strategies` action-queue item:
**half-done is the right place to leave it.** 5 of 8 audited (4 PASS_CHORDIA, 1 PARK).
The 3 unaudited names are not the highest-EV next move — the prior-day-context
literature-grant track is.

This survey is read-only and writes nothing to canonical layers. No verdicts
issued; no audit_log rows added.
