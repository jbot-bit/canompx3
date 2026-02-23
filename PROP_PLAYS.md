# PROP PLAYS — Manual Trading Playbook
_$2K trailing DD constraint. 1 trade per session window. No automation needed._
_Last updated: 2026-02-23_

---

## The Rules

- **Max trailing DD: $2,000** (treat floor as $0 until $3K+ ahead)
- **1 micro contract per trade** — no scaling yet
- **1 instrument per session window** — when multiple slots fire at the same time, pick ONE
- Stop for the day if down 3R. Stop trading if portfolio DD hits $1,800.

---

## Manual Session Schedule (Brisbane time)

Slots sorted by time. When there's a clash — pick the highest ExpR.

### Morning (Brisbane)

| Time (BRIS) | Trade | ExpR | WR% | RR | Note |
|---|---|---|---|---|---|
| **9:00am** | **MGC_1800** | **+0.333** | **49%** | **2.0** | Gold ORB. Solo — no clash. Set alarm. |

### Afternoon (Brisbane)

| Time (BRIS) | Trade | ExpR | WR% | RR | Note |
|---|---|---|---|---|---|
| **11:30am** | **MNQ_0030** | **+0.201** | **42%** | **2.0** | NQ ORB. Solo. Arvo session. |

### Night / US Session (Brisbane — main cluster)

| Time (BRIS) | Trade | ExpR | WR% | RR | Note |
|---|---|---|---|---|---|
| 11:30pm | MNQ_CME_OPEN | +0.130 | 61% | 1.0 | NQ pre-CME. Solo. OK to skip if tired. |
| **12:00am** | **MGC_0900** | **+0.443** | **46%** | **2.5** | **Gold NY open ORB. Solo. Best risk-adj.** |
| **12:30am** | **MES_US_EQUITY_OPEN** | **+0.585** | **43%** | **1.5** | **A0 — only validated slot. Skip MNQ/M2K.** |
| **1:00am** | **MGC_1000** | **+0.311** | **50%** | **2.0** | Gold 10am ET. Pick over MES_1000/MNQ_1000. |
| 1:30am | ~~MES_US_POST_EQUITY~~ | +0.077 | 59% | 1.0 | Weakest slot. Skip for now. |
| **2:00am** | **MNQ_1100** | **+0.249** | **47%** | **2.0** | NQ 11am ET. Solo — easy to watch. |
| **3:00am** | **MES_CME_CLOSE** | **+0.206** | **70%** | **1.0** | CME afternoon close. Pick over MNQ_CME_CLOSE. |

### Evening (Brisbane — optional)

| Time (BRIS) | Trade | ExpR | WR% | RR | Note |
|---|---|---|---|---|---|
| 6:00pm | MNQ_LONDON_OPEN | +0.092 | 59% | 1.0 | Weakest. Skip unless bored. |

---

## Realistic Daily Plans

You don't need to trade every session. Pick what fits your schedule.

### Option A — Morning Person (light)
Trade 2–3 sessions, done by noon.
- ✅ 9:00am MGC_1800
- ✅ 11:30am MNQ_0030
- Done. ~2 trades/day average.

### Option B — Night Owl (US session only)
Sit the US open cluster. 12:00am–3:00am.
- ✅ 12:00am MGC_0900
- ✅ 12:30am MES_US_EQUITY_OPEN (A0)
- ✅ 1:00am MGC_1000
- ✅ 2:00am MNQ_1100
- ✅ 3:00am MES_CME_CLOSE
- ~3–4 trades/night. This is the high-value cluster.

### Option C — Hybrid (most realistic)
Morning session + US open.
- ✅ 9:00am MGC_1800
- ✅ 11:30am MNQ_0030
- ✅ 12:00am MGC_0900
- ✅ 12:30am MES_US_EQUITY_OPEN (A0)
- ~3–4 trades/day. Best bang for time.

---

## Income Expectation by Plan

All figures backtested. Apply ~35% haircut for live reality.

| Plan | Slots active | Backtest $/mo | Realistic $/mo |
|---|---|---|---|
| Option A (morning only) | 2 | ~$180 | ~$115 |
| Option B (US session) | 5 | ~$370 | ~$240 |
| Option C (hybrid) | 4–5 | ~$320 | ~$210 |
| Full 15 slots (automated) | 15 | $465 | ~$300 |

Max DD stays under $1,100 for all configs (46% safety margin on $2K limit).

---

## DD Rules

| Situation | Action |
|---|---|
| Normal | Trade your plan |
| Down 3R in a day | Stop for the day |
| Portfolio DD hits $1,000 | Go half-size next week, review |
| Portfolio DD hits $1,500 | Stop, reassess — something's wrong |
| Portfolio DD hits $2,000 | Account blown / eval failed |

---

## Validation Status (be honest about this)

Only one slot is formally validated:

| Slot | Status |
|---|---|
| **MES_US_EQUITY_OPEN (A0)** | ✅ PROMOTE — passed no-leakage WF + stress + falsification |
| All other slots | ⚠️ Backtested — in forward gate, need 60+ live trades each |

**What this means:** The income projections assume all slots perform as in backtest. Some may not. A0 is the one you can trust most. The others are best-effort based on solid backtesting but not yet falsification-verified.

---

## How to Execute an ORB Trade (reminder)

1. **Set alarm** for the session start time
2. **Identify the ORB range** — high and low of the opening period (e.g., 5-min ORB = first 5 candles)
3. **Wait for the break** — price closes outside the range
4. **Check filter** — volume/gap condition must be met (see filter column above)
5. **Enter on confirmation bar** — next candle after break, in the direction of break
6. **Stop** at opposite side of ORB range (= 1R)
7. **Target** at RR × stop distance (e.g., RR 2.0 = 2× the stop)
8. **Log the trade** — date, slot, entry, stop, target, outcome

---

## What to Add Next

As more slots pass the forward gate (60+ live trades each), add them to this playbook. Check `research/output/forward_gate_status_latest.md` for current status.

Priority for validation:
1. MGC_0900 — highest ExpR after A0, worth fast-tracking
2. MGC_1000 — solid, solo slot
3. MNQ_1100 — good ExpR, no clashes
