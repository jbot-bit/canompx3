# Trading Playbook — Brisbane

**Version:** V2 — Combined Claude + Codex, March 16 2026
**Goal:** Prove edge live → scale to $100K/year via Apex EOD copy-traded stack

---

## The Plan in One Sentence

Phase 1: trade manually to prove the edge is real. Phase 2: copy-trade 20 Apex EOD accounts on the best overnight pair. Phase 3: add automation + self-funded capital.

---

## Phase 1: Manual Proof (Months 1-3)

Trade 5 sessions manually using G-filters you can eyeball. No automation needed. Purpose is to build reps, confirm backtest-to-live match, and earn first payouts.

**Firm:** Apex 50K EOD (primary) | TopStep $50K (MGC morning lane)
**Contracts:** 1 micro per signal
**Stop:** 0.75x on prop | 1.0x on self-funded

### The 5-Session Plan (3 sit-downs)

All times Brisbane local (AEST, UTC+10). Brisbane has no DST — everything else shifts around you.

**Morning Block (10:00–11:15) — coffee + charts**

| Session | Time | Instrument | Filter | How to Check | ORB | RR | Entry |
|---------|------|------------|--------|-------------|-----|-----|-------|
| **TOKYO_OPEN** | 10:00 (fixed) | MNQ | ORB_G5_CONT | ORB >= 5 MNQ pts + break bar closes beyond ORB edge | 5m | 1.5 | E2 CB1 |
| **SINGAPORE_OPEN** | 11:00 (fixed) | MNQ | ORB_G8 | ORB >= 8 MNQ pts | 15m | 1.5 | E2 CB1 |

- TOKYO: ORB forms by 10:05. Check size. Set bracket or walk away.
- SINGAPORE: ORB forms by 11:15. Check size. Set bracket or walk away.
- Both Asia sessions — no DST, same time every day, year-round.

**Evening Block (17:00–18:15) — after work**

| Session | Time | Instrument | Filter | How to Check | ORB | RR | Entry |
|---------|------|------------|--------|-------------|-----|-----|-------|
| **EUROPE_FLOW** | 17:00 (winter) / 18:00 (summer) | MNQ | ORB_G8 | ORB >= 8 MNQ pts | 5m | 1.5 | E2 CB1 |
| **LONDON_METALS** | 18:00 (winter) / 17:00 (summer) | MNQ | ORB_G6_NOMON | ORB >= 6 MNQ pts + not Monday | 15m | 2.0 | E2 CB1 |

- These two always swap order with DST but are always 1 hour apart.
- Winter: EUROPE_FLOW 17:00 first, LONDON_METALS 18:00 second.
- Summer: LONDON_METALS 17:00 first, EUROPE_FLOW 18:00 second.
- One sit-down, two sessions, back-to-back.
- LONDON_METALS NOMON = skip Mondays (Mondays are statistically negative).

**Night Block (23:30 or 00:30) — last thing before bed**

| Session | Time | Instrument | Filter | How to Check | ORB | RR | Entry |
|---------|------|------------|--------|-------------|-----|-----|-------|
| **NYSE_OPEN** | 23:30 (summer) / 00:30 (winter) | MNQ | ORB_G4 | ORB >= 4 MNQ pts | 5m | 1.5 | E2 CB1 |

- Lowest threshold (G4) so it fires often — NYSE_OPEN has big ORBs.

### Optional: TopStep MGC Morning Lane

If you also open a TopStep $50K account (Apex bans metals), add:

| Session | Time | Instrument | Filter | ORB | RR | Entry | Note |
|---------|------|------------|--------|-----|-----|-------|------|
| **CME_REOPEN** | 08:00 (fixed) | MGC | ORB_G4_FAST10 | 5m | 1.5 | E2 CB1 | ORB gate: skip if ORB > 26 pts |

- MGC = $10/point (5x MNQ). Higher per-trade value but more volatile.
- ORB gate is mandatory on TopStep: if MGC ORB > 26 points, risk exceeds 10% of $2K DD. Skip.
- This adds a 4th sit-down at 08:00 but on a different firm + instrument = diversification.

### Phase 1 Expected Stats

| Session | Filter | N (backtest) | ExpR | Sharpe | Est. Trades/Mo |
|---------|--------|-------------|------|--------|----------------|
| TOKYO_OPEN | ORB_G5_CONT | ~200+ | 0.17+ | 1.0+ | ~5-8 |
| SINGAPORE_OPEN | ORB_G8 | 1,117 | 0.091 | 0.77 | ~8-10 |
| EUROPE_FLOW | ORB_G8 | 1,117 | 0.133 | 1.46 | ~9-11 |
| LONDON_METALS | ORB_G6_NOMON | 1,050 | 0.098 | 0.77 | ~8-10 |
| NYSE_OPEN | ORB_G4 | ~500+ | 0.10+ | 0.70+ | ~8-12 |
| **TOTAL** | | | | | **~38-51** |

**~2 trades/day average.** Some days 0, some days 4. That's normal — filters exist to skip bad setups.

---

## Pre-Session Routine (same for every session)

1. **Session starts** → wait for ORB to form (5m or 15m depending on session)
2. **Measure ORB range** in points (high minus low)
3. **Check filter:**
   - Size >= threshold? (G4=4pts, G5=5pts, G6=6pts, G8=8pts)
   - CONT variant? Break bar must close beyond the ORB edge, not just wick through.
   - NOMON? Skip if today is Monday.
   - FAST10? Break must occur within 10 bars of ORB formation.
   - MGC ORB gate? Skip if ORB > 26 points (TopStep only).
4. **If filter passes → set bracket:**
   - **Entry:** stop-market at ORB high (long side) AND ORB low (short side) — whichever breaks first
   - **Stop:** opposite ORB edge × 0.75 (prop) or × 1.0 (self-funded)
   - **Target:** entry + (stop distance × RR)
5. **If filter fails → no trade.** Close chart. Done.
6. **After bracket set → walk away.** The bracket does the work.
7. **If no fill within session → cancel.** No chasing into the next session.

### How to Set the Bracket (E2 Stop-Market)

E2 means you place TWO stop-market orders — one above the ORB high (long), one below the ORB low (short). Whichever triggers first is your trade. Cancel the other immediately.

```
Example: MNQ EUROPE_FLOW, ORB high = 20,150, ORB low = 20,140 (10pt range, passes G8)

Stop multiplier: 0.75x prop
Risk = 10 pts x 0.75 = 7.5 pts
RR target = 1.5

LONG bracket:
  Entry:  buy stop at 20,150
  Stop:   20,150 - 7.5 = 20,142.50
  Target: 20,150 + (7.5 x 1.5) = 20,161.25

SHORT bracket:
  Entry:  sell stop at 20,140
  Stop:   20,140 + 7.5 = 20,147.50
  Target: 20,140 - (7.5 x 1.5) = 20,128.75

→ Place both. First one fills = your trade. Cancel the other.
```

---

## Decision Tree: What to Trade Today

```
START
  |
  +-- Full energy, normal day?
  |     -> All 3 blocks: Morning (10:00-11:15) + Evening (17:00-18:15) + Night (23:30/00:30)
  |     -> ~2-4 signals depending on filter pass
  |
  +-- Good energy, but want an easy day?
  |     -> Morning + Evening only (skip Night)
  |     -> Still 4 sessions, still ~2 signals
  |
  +-- Tired but functional?
  |     -> Evening block only (17:00-18:15)
  |     -> 2 sessions, ~1 signal, no late night
  |
  +-- Morning only available?
  |     -> TOKYO + SINGAPORE (10:00-11:15)
  |     -> Best human hours, no DST nonsense
  |
  +-- Wrecked / sick / distracted?
  |     -> Trade NOTHING. Zero trades is a valid day.
  |     -> The edge is there tomorrow.
  |
  +-- Monday?
        -> Skip LONDON_METALS (NOMON filter)
        -> Everything else runs normal
```

---

## DST Calendar: When Do Times Shift?

Only the Evening and Night blocks shift. Morning is fixed year-round.

| Period | EUROPE_FLOW | LONDON_METALS | NYSE_OPEN |
|--------|-------------|---------------|-----------|
| **Jan–mid Mar** (both winter) | 17:00 | 18:00 | 00:30 |
| **Mid Mar** (US springs forward, UK hasn't) | 17:00 | 18:00 | 23:30 |
| **Late Mar–late Oct** (both summer) | 18:00 | 17:00 | 23:30 |
| **Late Oct–early Nov** (UK falls back, US hasn't) | 17:00 | 18:00 | 23:30 |
| **Nov–Dec** (both winter) | 17:00 | 18:00 | 00:30 |

**Morning block: TOKYO 10:00, SINGAPORE 11:00 — always. No exceptions.**

---

## Phase 2: The $100K/Year Plan (Months 3-12)

Once Phase 1 proves the edge (30+ live trades, positive P&L, 2+ payouts), transition to the scaling stack.

### The Deployed Pair (from codex's proper firm-aware sim pass)

| Strategy ID | Session | Time (Brisbane) | Inst | ORB | RR | Filter |
|------------|---------|-----------------|------|-----|-----|--------|
| `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5` | CME_PRECLOSE | 05:45/06:45 | MNQ | 5m | 1.0 | ORB_G5 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_VOL_RV12_N20` | COMEX_SETTLE | 03:30/04:30 | MNQ | 5m | 1.0 | VOL_RV12_N20 |

**Vehicle:** Apex 50K EOD
**Method:** Copy-trade from 1 lead account to all follower accounts

### Per-Account Economics (Apex 50K EOD, simulated)

| Micros | Survival | Median $/mo | P5 $/mo |
|--------|----------|-------------|---------|
| 1 | 100.0% | ~$210 | ~$85 |
| **2** | **99.2%** | **~$430** | **~$167** |
| 3 | 94.9% | ~$638 | — |
| 4 | 89.3% | ~$877 | — |

**2 micros/account is the conservative serious sizing.** 3 micros is aggressive. 4 is too hot.

### Stack Math

| Target | Accounts | Micros | Monthly | Annual | Notes |
|--------|----------|--------|---------|--------|-------|
| **$100K/year** | 20 | 2/acct | ~$8,600 | **~$103K** | Conservative path |
| **$150K/year** | 20 | 3/acct | ~$12,760 | **~$153K** | Aggressive, 94.9% survival |
| $200K/year | 20 | 4/acct | ~$17,540 | ~$210K | Too hot (89.3% survival) |

### Ramp Schedule

| Step | What | Gate Before Proceeding |
|------|------|----------------------|
| 1 | 1 funded account, 1 micro | — |
| 2 | 5 funded accounts, 1 micro | Zero rule breaches, zero copier errors |
| 3 | 20 funded accounts, 1 micro | Live DD not worse than modeled P95 |
| 4 | 20 funded accounts, 2 micros | Payout qualification actually achieved |
| 5 | 20 funded accounts, 3 micros | 2+ months clean at 2 micros |

### Abort Scaling If

- Two execution errors in a month
- Live DD materially exceeds modeled P95
- Missed sessions become recurrent
- Payout qualification stalls
- You start manually interfering with bracket logic

### Apex EOD Payout Rules (Current March 2026)

- 5 qualifying days required (each day must be +$250 minimum)
- Safety net: $52,100 (trailing stops)
- Minimum balance to request payout: $52,600
- Max 6 payouts
- 50% consistency rule (no single day > 50% of total profits)
- **Caveat:** Broader compliance pages mention stricter 30% windfall language. Underwrite to the stricter interpretation until confirmed.

Source: https://support.apextraderfunding.com/hc/en-us/articles/47205823183003-EOD-Payouts

### Lifestyle Consideration

The deployed pair requires early Brisbane starts:
- COMEX_SETTLE: ~03:30/04:30
- CME_PRECLOSE: ~05:45/06:45

**Options:**
- Wake at 03:15, place brackets for both sessions, go back to sleep
- Set an alarm, place the order in 2 minutes, back to bed
- Once auto-execution is ready (Phase 3), sleep through entirely

The question is not whether the edge exists. It does. The question is whether you will reliably run that schedule.

---

## Phase 3: Automation + Self-Funded (Month 12+)

### What Automation Unlocks

Switch from G-filters (eyeball) to **VOL_RV12_N20** (volume-based, needs code):
- Better edge: higher Sharpe (2.28 vs 1.46 on EUROPE_FLOW)
- Higher frequency (~3 trades/day vs ~2)
- Unlocks sleep-impossible sessions (03:30, 05:45, 00:00)

| Session | VOL Filter | Trades/Mo | ExpR | Sharpe |
|---------|-----------|-----------|------|--------|
| TOKYO_OPEN | VOL_RV12_N20 | 14.8 | 0.167 | 1.65 |
| SINGAPORE_OPEN | VOL_RV12_N20 | 16.7 | 0.140 | 1.15 |
| EUROPE_FLOW | VOL_RV12_N20 | 12.7 | 0.144 | 2.28 |
| LONDON_METALS | VOL_RV12_N20 | 14.1 | 0.126 | 1.20 |
| NYSE_OPEN | VOL_RV12_N20 | 11.9 | 0.158 | 1.94 |
| US_DATA_1000 | VOL_RV12_N20 | ~15 | 0.149 | 2.24 |
| COMEX_SETTLE | VOL_RV12_N20 | ~14 | 0.181 | 2.08 |
| CME_PRECLOSE | VOL_RV12_N20 | ~13 | 0.266 | 2.29 |

### Self-Funded IBKR (when prop edge is proven)

No prop constraints. DD is temporary, not death. Scale by contracts, not accounts.

| Capital | Sessions | Contracts | Annual | DD% Med |
|---------|----------|-----------|--------|---------|
| $50K | 9 (all) | 5c | ~$56K | 20% |
| $50K | 9 (all) | 10c | ~$112K | 40% |
| $100K | 9 (all) | 10c | ~$112K | **20%** |
| $100K | 9 (all) | 15c | ~$169K | 29% |
| $100K | 9 (all) | 20c | ~$225K | 39% |

IBKR integration not built yet. Architecture ready (broker ABC pattern, ~700 lines to add).

---

## Fatigue Protocol

**Rule:** If you traded the Night block past midnight and slept less than 6 hours, skip the Morning block. Resume Evening.

**Rule:** If you skip 3+ consecutive days, do NOT compensate by adding extra sessions or larger size. Just resume normal.

**Rule:** If you find yourself staring at the chart after the bracket is set — close the chart. Watching doesn't help. The bracket does the work.

---

## When to Add a Second Contract

**NOT before all three conditions are met:**

1. **30+ live trades executed** (proves you can follow the plan mechanically)
2. **Prop account is funded** (passed evaluation, trading live capital)
3. **Positive P&L over 2+ months** (not just "I think this works" — show the numbers)

**Scaling order:**
- First: 1c MNQ on all sessions (current plan)
- Second: 2c on highest-confidence session (start with the deployed pair)
- Third: 2c across all sessions
- Note: Apex bans MGC — MNQ scales on Apex, MGC only on TopStep/Tradeify

---

## What Not to Do

- **Do not override the filter.** ORB too small? No trade. That IS the system working.
- **Do not chase.** If you missed the ORB window, it's gone. Next session.
- **Do not add sessions outside the plan.** Random sessions at 3:30 AM are for Phase 2 copy-trade, not manual heroics.
- **Do not watch the trade after setting the bracket.** Set it, walk away, check later.
- **Do not trade on tilt.** After a loss, the next session runs exactly the same. No revenge sizing, no "I need to make it back."
- **Do not skip the filter "just this once."** Every blown trade in the backtest came from unfiltered days.
- **Do not run 20 accounts before proving 1.** The ramp schedule exists for a reason.

---

## Quick Reference Card

Print this. Tape it next to your screen.

```
+---------------------------------------------------------------+
|  PHASE 1 MANUAL TRADING — DAILY CARD                          |
|                                                                |
|  MORNING (10:00-11:15 Brisbane — fixed, no DST)                |
|    TOKYO_OPEN     10:00  MNQ  G5_CONT  5m ORB   RR 1.5        |
|    SINGAPORE_OPEN 11:00  MNQ  G8       15m ORB  RR 1.5        |
|                                                                |
|  EVENING (17:00-18:15 Brisbane — shifts with UK/summer DST)    |
|    First session   17:00  (see DST table for which is which)   |
|    Second session  18:00                                       |
|    EUROPE_FLOW     MNQ  G8       5m ORB   RR 1.5              |
|    LONDON_METALS   MNQ  G6_NOMON 15m ORB  RR 2.0  skip Mon   |
|                                                                |
|  NIGHT (23:30 summer / 00:30 winter Brisbane)                  |
|    NYSE_OPEN       MNQ  G4       5m ORB   RR 1.5              |
|                                                                |
|  EVERY SESSION:                                                |
|    1. Wait for ORB to form                                     |
|    2. Measure range in points                                  |
|    3. Range >= filter? Set bracket. Range < filter? Walk away. |
|    4. E2: stop-market both sides. First fill = trade.          |
|    5. Stop = opposite ORB edge x 0.75 (prop)                  |
|    6. Target = risk x RR                                       |
|    7. Walk away. Check result later.                           |
|                                                                |
|  RULES:                                                        |
|    * Filter fail = no trade. Period.                           |
|    * No fill = cancel. No chasing.                             |
|    * Monday = skip LONDON_METALS.                              |
|    * Tired? Morning block only. Or skip entirely.              |
|    * Never override. Never chase. Never revenge trade.         |
+---------------------------------------------------------------+
```

---

## Monthly Review

1. **Count trades taken** vs filter-passing days you were present. Execution rate > 80%.
2. **Win rate tracking:** compare your results to backtest expected rates.
3. **Run `/regime-check`** — if any strategy shows DECAY, pause that session.
4. **Journal:** any overrides? Any chases? Any "just this once"? Fix the pattern.
5. **Sleep check:** if Night block is wrecking you, drop it. 4 sessions is plenty.
6. **Phase 2 readiness:** after 30+ trades with consistent execution, start scaling accounts.
7. **Payout check:** are you qualifying for Apex payouts? If not, diagnose why.
