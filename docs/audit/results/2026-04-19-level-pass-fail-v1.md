# Level Pass/Fail V1

Research-only event study locked by `docs/audit/hypotheses/2026-04-19-level-pass-fail-v1.yaml`.

## Scope

- Instruments: MES, MGC, MNQ
- Sessions: CME_PRECLOSE, COMEX_SETTLE, EUROPE_FLOW, NYSE_OPEN, TOKYO_OPEN, US_DATA_1000
- Levels: prev_day_high / prev_day_low only
- Interaction kinds: close_through / wick_fail
- Event window: first 30 minutes of the session
- Response metric: signed next-2-bar close-to-close return normalized by ATR20
- Selection uses pre-2026 only; 2026 is diagnostic OOS only

## Family Verdict

- Locked family K: 72
- Primary survivors (BH + N>=100 + avg_is>0): 0

No primary survivors.

The family remains useful as infrastructure proof, but this first narrow pass did not surface a clear positive cell under the locked standards.

## Caveats

- This is not a trade strategy or deployability result.
- No costs or trade geometry are applied; this is a short-horizon directional event study only.
- Protocol B skepticism applies because exact level-interaction theory is not separately grounded in local literature resources.