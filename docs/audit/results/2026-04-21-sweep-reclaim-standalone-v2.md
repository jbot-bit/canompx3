# Sweep Reclaim Standalone V2

Pre-reg: `docs/audit/hypotheses/2026-04-21-sweep-reclaim-standalone-v2.yaml`.

## Scope

- Instruments: MES, MNQ
- Sessions: EUROPE_FLOW, NYSE_OPEN
- Levels: prev_day_high / prev_day_low only
- Entry: reclaim close
- Stop: one tick beyond the maximum/minimum sweep excursion known by the reclaim close
- Target: fixed 1.5R
- Entry window: first 60 minutes of the session
- Outcome path: stop/target on bars after entry, otherwise end-of-day close
- Selection uses pre-2026 only; 2026 is diagnostic OOS only

## Family Verdict

- Locked family K: 8
- Trades collected: 417 total (397 IS, 20 OOS)
- CANDIDATE_READY: 0
- RESEARCH_SURVIVOR: 0
- DEAD: 8

## Per-Cell Table

| Instrument | Session | Level | Dir | N_IS | ExpR_IS | WR_IS | t | p(1t) | q(BH) | WFE | N_OOS | ExpR_OOS | OOS/IS | Worst Era | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MNQ | EUROPE_FLOW | prev_day_low | long | 26 | +0.1349 | 53.8% | +0.636 | 0.2653 | - | - | 1 | +1.2713 | - | - | **DEAD** |
| MNQ | NYSE_OPEN | prev_day_low | long | 63 | -0.0130 | 42.9% | -0.089 | 0.5355 | - | - | 3 | -0.1903 | - | - | **DEAD** |
| MNQ | NYSE_OPEN | prev_day_high | short | 80 | -0.0658 | 42.5% | -0.535 | 0.7028 | - | - | 6 | +0.9609 | - | - | **DEAD** |
| MES | NYSE_OPEN | prev_day_low | long | 63 | -0.0852 | 42.9% | -0.632 | 0.7352 | - | - | 3 | +0.4456 | - | - | **DEAD** |
| MES | EUROPE_FLOW | prev_day_high | short | 31 | -0.1475 | 48.4% | -0.900 | 0.8123 | - | - | 0 | - | - | - | **DEAD** |
| MES | NYSE_OPEN | prev_day_high | short | 70 | -0.1631 | 41.4% | -1.347 | 0.9089 | - | - | 7 | -0.0261 | - | - | **DEAD** |
| MNQ | EUROPE_FLOW | prev_day_high | short | 47 | -0.2536 | 40.4% | -1.835 | 0.9635 | - | - | 0 | - | - | - | **DEAD** |
| MES | EUROPE_FLOW | prev_day_low | long | 17 | -0.5527 | 23.5% | -2.715 | 0.9923 | - | - | 0 | - | - | - | **DEAD** |

## Interpretation

- No cell survived H1 at the locked K=8 scope. This standalone family is dead under the current geometry.
- C8 is only counted when N_OOS >= 20. Thin OOS cannot promote a cell.
- The stop is explicitly the known sweep extreme by reclaim close, plus/minus one tick, to avoid discretionary stop placement.

## Not Done

- No write to validated_setups, edge_families, live_config, or account profiles.
- No deployment recommendation beyond the verdict classes above.
- No widening to overnight levels, first-retest entries, or next-liquidity targets.