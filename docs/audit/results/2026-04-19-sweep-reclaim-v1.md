# Sweep Reclaim V1

Research-only event study locked by `docs/audit/hypotheses/2026-04-19-sweep-reclaim-v1.yaml`.

## Scope

- Instruments: MES, MGC, MNQ
- Sessions: CME_PRECLOSE, COMEX_SETTLE, EUROPE_FLOW, NYSE_OPEN, TOKYO_OPEN, US_DATA_1000
- Levels: prev_day_high / prev_day_low only
- Event definition: swept close-through that reclaims back inside within 2 bars
- Event window: first 30 minutes of the session
- Response metric: signed next-2-bar close-to-close return from reclaim close normalized by ATR20
- Selection uses pre-2026 only; 2026 is diagnostic OOS only

## Family Verdict

- Locked family K: 36
- Primary survivors (BH + N>=50 + avg_is>0): 0

No primary survivors.

The family remains useful as a cleaner trapped-side re-test, but this narrow sweep-reclaim pass did not surface a clear positive cell under the locked standards.

## Warm Cells (Informational Only)

- MNQ NYSE_OPEN prev_day_low: IS n=47, avg=+0.0251, WR=55.3%, t=2.11, p=0.0405, BH_survivor=False
- MES COMEX_SETTLE prev_day_low: IS n=26, avg=+0.0223, WR=53.8%, t=1.78, p=0.0873, BH_survivor=False
- MNQ NYSE_OPEN prev_day_high: IS n=64, avg=+0.0174, WR=54.7%, t=1.70, p=0.0949, BH_survivor=False
- MNQ US_DATA_1000 prev_day_high: IS n=39, avg=+0.0113, WR=51.3%, t=1.34, p=0.1888, BH_survivor=False
- MNQ CME_PRECLOSE prev_day_low: IS n=26, avg=+0.0151, WR=46.2%, t=1.30, p=0.2060, BH_survivor=False

## Caveats

- This is not a trade strategy or deployability result.
- No costs or trade geometry are applied; this is a short-horizon reversal event study only.
- Protocol B skepticism applies because exact sweep-reclaim theory is not separately grounded in local literature resources.