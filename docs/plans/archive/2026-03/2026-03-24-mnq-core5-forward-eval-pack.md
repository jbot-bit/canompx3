---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# MNQ Unfiltered CORE 5 — Pre-Registered Forward Evaluation Pack

**Registered:** 2026-03-24
**Evaluation window:** 2026-03-25 through minimum 2026-05-19 (8 weeks)
**Instrument:** MNQ | **Entry:** E2 | **Aperture:** O5 | **RR:** 1.0 | **Filter:** NO_FILTER
**Sessions:** CME_PRECLOSE, COMEX_SETTLE, NYSE_OPEN, US_DATA_1000, EUROPE_FLOW

**Authority:** Paper governance memo (2026-03-24). This pack does NOT authorize live promotion, portfolio edits, or new discovery.

---

## 1. Daily Operator Checklist

Run after market close (after 09:00 Brisbane next day, when all sessions are complete).

```
[ ] 1. Refresh bars:    python pipeline/ingest_dbn.py --instrument MNQ --resume
[ ] 2. Build features:  python pipeline/build_daily_features.py --instrument MNQ --start <TODAY> --end <TODAY>
[ ] 3. Build outcomes:  python -m trading_app.outcome_builder --instrument MNQ --start <TODAY> --end <TODAY>
[ ] 4. Run paper replay:
      python -m trading_app.paper_trader \
        --instrument MNQ --raw-baseline --rr-target 1.0 --entry-model E2 \
        --exclude-sessions "NYSE_CLOSE,TOKYO_OPEN,CME_REOPEN,LONDON_METALS,US_DATA_830,SINGAPORE_OPEN,BRISBANE_1025" \
        --start 2026-01-01 --end <TODAY> \
        --output data/paper_mnq_core5_replay.csv --quiet
[ ] 5. Record today's trades in forward scoreboard (Section 3)
[ ] 6. Check kill triggers — if any fire, log in incident log (Section 5) and STOP
[ ] 7. If Friday: complete weekly risk memo (Section 2)
```

**If any step fails:** Log the failure in the incident log. Do NOT skip steps. Do NOT improvise fixes. Resume next trading day after root cause is identified.

---

## 2. Weekly Risk Memo Template

Complete every Friday. File as `data/weekly_risk/week_YYYY-MM-DD.md`.

```
# Weekly Risk Memo — MNQ CORE 5 Paper Book
Week ending: YYYY-MM-DD
Operator: _______________

## Forward Trades This Week
| Session | Trades | Wins | Losses | Week PnL (R) |
|---------|--------|------|--------|--------------|
| CME_PRECLOSE | | | | |
| COMEX_SETTLE | | | | |
| NYSE_OPEN | | | | |
| US_DATA_1000 | | | | |
| EUROPE_FLOW | | | | |
| **TOTAL** | | | | |

## Cumulative Since Registration (2026-03-25)
| Session | Total Trades | Cumulative PnL (R) | Consecutive Negative Months |
|---------|-------------|--------------------|-----------------------------|
| CME_PRECLOSE | | | |
| COMEX_SETTLE | | | |
| NYSE_OPEN | | | |
| US_DATA_1000 | | | |
| EUROPE_FLOW | | | |

## Kill Trigger Status
- [ ] Any session at 3 consecutive negative months? YES / NO
- [ ] Any session cumulative PnL <= -10R? YES / NO
- [ ] Slippage > 2x cost model on 20+ fills? YES / NO
- [ ] Any data pipeline failure unresolved > 2 business days? YES / NO

## Regime Context (monitoring only — not a gate)
- Median cost/risk % this week: _____ (ref: edge lives below ~10%)
- Average ATR_20 this week: _____ (ref: edge amplified above ~200)

## Anomalies / Notes
(Anything unusual: missed days, data gaps, cost model concerns, market regime observations)

## Decision
- [ ] CONTINUE monitoring — no triggers fired
- [ ] KILL session: _______________ (reason: 3 months negative / cumulative <= -10R / slippage)
- [ ] ESCALATE to live promotion review (requires Section 4 gate card)
```

---

## 3. Forward Scoreboard Fields

Maintain in `data/paper_mnq_core5_scoreboard.csv`. One row per trading day per session.

| Field | Type | Description |
|-------|------|-------------|
| `date` | DATE | Trading day |
| `session` | TEXT | Session name |
| `had_break` | BOOL | Did an ORB break occur? |
| `direction` | TEXT | LONG / SHORT / NO_BREAK |
| `outcome` | TEXT | win / loss / scratch / no_break |
| `pnl_r` | FLOAT | P&L in R-multiples (NULL if no break) |
| `entry_price` | FLOAT | From paper trader output |
| `stop_price` | FLOAT | From paper trader output |
| `target_price` | FLOAT | From paper trader output |
| `exit_price` | FLOAT | From paper trader output |
| `orb_size_pts` | FLOAT | ORB high - low in points |
| `risk_dollars` | FLOAT | orb_size * $2/pt |
| `pnl_dollars` | FLOAT | pnl_r * risk_dollars |
| `notes` | TEXT | Anomalies, data issues, or blank |

**Derived fields (computed weekly, not per-trade):**
- Cumulative PnL (R) per session
- Cumulative PnL ($) per session
- Rolling 30-trade ExpR per session
- Monthly PnL (R) per session (for consecutive-month kill trigger)

---

## 4. Live-Promotion Gate Card

**No session may be promoted to LIVE_PORTFOLIO without ALL gates passing AND human sign-off.**

| # | Gate | Criterion | Fail = |
|---|------|-----------|--------|
| 1 | Forward sample | Minimum 8 weeks of paper data collected (ops minimum; promotion requires materially larger forward sample — exact N determined at review) | WAIT |
| 2 | No kill triggers | Zero kill triggers fired during evaluation | REJECT if any fired |
| 3 | Forward ExpR | Session forward ExpR > 0 (point estimate) | REJECT |
| 4 | BH FDR | Session survives BH FDR at honest instrument K (K_MNQ) on combined IS+forward data | REJECT |
| 5 | Walk-forward | WFE > 50% on IS+forward combined window | REJECT |
| 6 | Sensitivity | ExpR survives ±20% RR perturbation (test RR 0.8 and RR 1.2) | REJECT |
| 7 | Cost stress | ExpR remains positive at 1.5x current cost model | REJECT |
| 8 | Human sign-off | Explicit written approval after reviewing all above evidence | REJECT without sign-off |

**Fail-closed:** Any gate returning REJECT or WAIT blocks promotion. No exceptions. No override without re-registration.

**What this gate card does NOT cover:**
- Position sizing (separate decision)
- Which prop firm / account (separate decision)
- Portfolio-level correlation / concentration limits (separate decision)
- Stop multiplier choice (0.75x vs 1.0x — separate decision)

---

## 5. Incident Log Template

Maintain in `data/paper_mnq_core5_incidents.md`. Append-only.

```
## YYYY-MM-DD — [CATEGORY]

**Category:** KILL_TRIGGER / DATA_GAP / PIPELINE_FAILURE / COST_ANOMALY / OTHER
**Session(s) affected:** _______________
**Description:** _______________
**Action taken:** _______________
**Resolution:** RESOLVED / OPEN
**Impact on evaluation:** (e.g., "1 day missing from CME_PRECLOSE forward sample")
```

---

## Registration Integrity

This document was written before the forward evaluation period begins. Any changes to evaluation criteria, kill triggers, or promotion gates after 2026-03-25 must be logged as amendments with date and rationale. Post-hoc changes to rescue or kill a result are prohibited under the Research Truth Protocol.

**Registered by:** Claude Code session 2026-03-24
**Countersigned by:** _______________________ (operator)
