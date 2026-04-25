# MES E2 TBBO slippage pilot v1

**Script:** `research/mes_e2_tbbo_slippage_pilot.py`
**Result CSV:** `research/data/tbbo_mes_pilot/slippage_results.csv`
**Scope:** MES | E2 | O5 | deployable MES sessions | TBBO stop-market repricing
**Sessions:** CME_PRECLOSE, COMEX_SETTLE, SINGAPORE_OPEN, US_DATA_830

## Verdict: **PASS**

median is modeled-conservative and p95 stays within 2x modeled slippage

## Action Taken

- Closed the MES routine-central-tendency portion of `cost-realism-slippage-pilot` in `docs/runtime/debt-ledger.md`.
- Left the MES modeled slippage in `pipeline/cost_model.py` unchanged because measured routine slippage is inside the existing modeled 1-tick assumption.
- Preserved event-tail uncertainty as open debt; this sample does not prove behavior on rare degraded/gap/liquidity-shock days.
- No live routing, sizing, validator, schema, or deployment behavior changed.

## Summary

- Valid repriced samples: `40`
- Error rows: `0`
- Modeled slippage: `1` ticks
- Median slippage: `0.00` ticks
- Mean slippage: `-0.25` ticks
- p95 slippage: `0.00` ticks
- Max slippage: `0.00` ticks
- Percent <= 1 tick: `100.0%`
- Percent <= 2 ticks: `100.0%`

## Integrity

- This measures fill-quality / cost realism only; it is not alpha discovery.
- Repricing delegates to `research.databento_microstructure.reprice_e2_entry`.
- ORB timing delegates to `pipeline.build_daily_features._orb_utc_window`.
- Cost assumptions come from `pipeline.cost_model.get_cost_spec("MES")`.
- Databento emitted a degraded-day warning for `2025-11-28`; the row repriced successfully and remains in the sample.

## Limitations

- This is a 40-window routine-liquidity pilot, not an event-tail stress sample.
- The paid TBBO cache and CSV are intentionally local ignored data under `research/data/tbbo_mes_pilot/`; the committed evidence is the bounded script, tests, and this result summary.
- The result supports leaving modeled MES routine slippage unchanged; it does not justify reducing the cost model or changing deployment.

## Reproduction / Outputs

No-spend verification from the existing local cache:

```bash
./.venv-wsl/bin/python -m research.mes_e2_tbbo_slippage_pilot --reprice-cache
```

Primary outputs:

- `research/data/tbbo_mes_pilot/manifest.json` local ignored Databento window manifest.
- `research/data/tbbo_mes_pilot/slippage_results.csv` local ignored repricing result CSV.
- `docs/audit/results/2026-04-24-mes-e2-slippage-pilot-v1.md` committed result summary and action record.

## Follow-up rule

- `PASS`: close the MES portion of `cost-realism-slippage-pilot`.
- `WARN`: keep the debt open for session/tail review before cost-model changes.
- `FAIL`: keep the debt open and review MES cost assumptions before trusting MES ExpR.
