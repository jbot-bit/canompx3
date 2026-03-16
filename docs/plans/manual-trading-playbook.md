# Trading Playbook — Brisbane

**Status:** CANONICAL prop trading playbook
**Version:** V3 — consolidated manual + allocation memo, March 16 2026
**Goal:** Prove edge live → scale to $100K/year via Apex 50K EOD copy-traded stack
This file is the single working prop memo for this repo.
Use it for:
- current firm selection and sizing
- manual bridge trading
- the copy-traded scale plan
- operating gates and abort rules

Rule hygiene:
- Official firm rules drift. Re-check the firm's support docs before paying for or scaling any account.
- This file separates verified official rules from local modeled conclusions.
- If a firm doc conflicts with older local notes, underwrite to the stricter interpretation until confirmed.

### Official Rule Snapshot To Underwrite Right Now

**Apex 50K EOD**
- Treat as the current base vehicle for the main scaling lane.
- Official docs checked in this cycle support: `20` active PA accounts max, `4 mini / 40 micro` cap on `50K EOD`, and EOD payout gating with `5` qualifying days at `+$250` minimum, safety net `52,100`, minimum balance `52,600`, and `6` payouts max.
- Official compliance docs do **not** support unattended automation on PA / Live accounts. Copy trading is allowed, and semi-automated software can be used only while actively monitored by the trader.
- Important caveat: Apex documentation is not perfectly clean. The EOD payout page uses a `50%` consistency framing, while broader compliance pages still contain stricter language around windfall concentration / compliance behavior. Underwrite to the stricter interpretation until the dashboard and support match.
- Sources:
  - `https://support.apextraderfunding.com/hc/en-us/articles/4406804554779-How-Many-Paid-Funded-Accounts-Am-I-Allowed-to-Have`
  - `https://support.apextraderfunding.com/hc/en-us/articles/47204516592795-EOD-Performance-Accounts-PA`
  - `https://support.apextraderfunding.com/hc/en-us/articles/47205823183003-EOD-Payouts`
  - `https://support.apextraderfunding.com/hc/en-us/articles/31519788944411-Performance-Account-PA-and-Compliance`
  - `https://support.apextraderfunding.com/hc/en-us/articles/4404875002139-What-are-the-Consistency-Rules-For-PA-and-Funded-Accounts`

**Topstep 50K Express Funded**
- Treat as the cleanest MGC / morning alternative when you want a second firm.
- Current local underwriting assumptions remain: `5 Express + 1 Live` account cap, `50 micro` cap on `50K`, flat by `3:10 PM CT`, and `90/10` payout structure on current accounts.
- Use Topstep for the morning MGC lane and not as the main overnight scaling vehicle.
- Sources used in local planning:
  - `https://help.topstep.com/en/articles/10799569-xfa-faq`
  - `https://help.topstep.com/en/articles/9208217-topstep-pricing`

**Tradeify 50K**
- Treat as the best non-Apex secondary firm in current local modeling.
- Current local planning assumptions still favor it as the clean external diversification lane, but this turn did not re-verify every payout / copier detail from Tradeify support.
- Use the current local conclusion operationally only after checking the live Tradeify help pages you are actually signing up under.
- Source used in local planning:
  - `https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns`

**MFFU**
- Do not use MFFU as a primary lane in the current plan.
- Older local notes that framed this as a simple "`$129/month forever`" rejection are not clean enough to keep treating as canonical without a fresh re-check.
- Current honest position: deprioritized, not chosen, and not fully re-verified in this turn.

---

## The Plan in One Sentence

Phase 1: trade manually to prove the edge is real. Phase 2: copy-trade 20 Apex EOD accounts on the best compliant pair. Phase 3: add automation only on firms / venues that officially allow it, or move to self-funded capital.

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

If you also open a TopStep $50K account for the MGC morning lane, add:

| Session | Time | Instrument | Filter | ORB | RR | Entry | Note |
|---------|------|------------|--------|-----|-----|-------|------|
| **CME_REOPEN** | 08:00 (fixed) | MGC | ORB_G4_FAST10 | 5m | 1.5 | E2 CB1 | ORB gate: skip if ORB > 26 pts |

- MGC = $10/point (5x MNQ). Higher per-trade value but more volatile.
- ORB gate is mandatory on TopStep: if MGC ORB > 26 points, risk exceeds 10% of $2K DD. Skip.
- This adds a 4th sit-down at 08:00 but on a different firm + instrument = diversification.
- Planning note: the current Apex operating lane in this file is equity-only. Treat MGC-on-Apex as unavailable unless you re-confirm product eligibility directly in current Apex support/dashboard.

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

### Manual-First Variant (No 03:30 / 04:30 Brisbane)

If manual trading will not include `COMEX_SETTLE`, use a separate manual lane and keep the ugly-hours pair as a modeled benchmark only.

**Best strict no-3am manual pair on Apex EOD**

| Strategy ID | Session | Time (Brisbane) | Inst | ORB | RR | Filter |
|------------|---------|-----------------|------|-----|-----|--------|
| `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5` | CME_PRECLOSE | 05:45 / 06:45 | MNQ | 5m | 1.0 | ORB_G5 |
| `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20` | NYSE_CLOSE | 06:00 / 07:00 | MNQ | 5m | 1.0 | VOL_RV12_N20 |

Modeled at `1 micro/account`, this pair was about `100.0%` survival with roughly `$166/month` median on Apex EOD.

**Higher-income manual variant if midnight is acceptable**

| Strategy ID | Session | Time (Brisbane) | Inst | ORB | RR | Filter |
|------------|---------|-----------------|------|-----|-----|--------|
| `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5` | CME_PRECLOSE | 05:45 / 06:45 | MNQ | 5m | 1.0 | ORB_G5 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6` | US_DATA_1000 | 00:00 / 01:00 | MNQ | 5m | 1.0 | ORB_G6 |

Modeled at `1 micro/account`, this pair was roughly `98.8%` survival with about `$264/month` median on Apex EOD.

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

These numbers are for the max-EV benchmark pair, not the manual-safe pair.

### Ramp Schedule

| Step | What | Gate Before Proceeding |
|------|------|----------------------|
| 1 | 1 funded account, 1 micro | — |
| 2 | 5 funded accounts, 1 micro | Zero rule breaches, zero copier errors |
| 3 | 20 funded accounts, 1 micro | Live DD not worse than modeled P95 |
| 4 | 20 funded accounts, 2 micros | Payout qualification actually achieved |
| 5 | 20 funded accounts, 3 micros | 2+ months clean at 2 micros |

### Apex Manual-First Ladder (`5am+`, No `3am`)

Use this if manual means:
- `05:00+` starts are fine
- `03:30 / 04:30` is not
- unattended Apex automation is off the table

**Manual base pair**
- `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5`
- `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20`

**Modeled per-account economics on Apex EOD**

| Micros | Survival | Median $/mo | P5 $/mo |
|--------|----------|-------------|---------|
| 1 | 100.0% | ~$169 | ~$64 |
| **2** | **99.3%** | **~$336** | **~$122** |
| 3 | 96.0% | ~$515 | ~$193 |
| 4 | 90.8% | ~$696 | ~$301 |

**Institutional use**
- `2 micros/account` is the manual serious-size lane
- `3 micros/account` is the aggressive manual lane
- `4 micros/account` is too hot for the base plan

**Manual-first stack math**

| Target | Accounts | Micros | Monthly | Annual | Notes |
|--------|----------|--------|---------|--------|-------|
| Base proof | 5 | 1/acct | ~$845 | ~$10K | Process proof, not income replacement |
| Serious manual | 20 | 2/acct | ~$6,700 | ~$80K | Clean first manual target |
| Aggressive manual | 20 | 3/acct | ~$10,300 | ~$124K | Requires better discipline and tolerance |

If you want a manual-first path to `100K/year`, the realistic answer is:
- start with the `5am+` pair
- get to `20 x 2 micros`
- only then decide whether the jump to `3 micros` is justified

### Midnight + `5am` Manual Ladder

If midnight is acceptable manually, the higher-income pair is:
- `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5`
- `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6`

But the risk profile is materially worse:

| Micros | Survival | Median $/mo | P5 $/mo |
|--------|----------|-------------|---------|
| 1 | 98.5% | ~$257 | ~$63 |
| 2 | 81.8% | ~$564 | ~$219 |
| 3 | 66.7% | ~$877 | ~$332 |

Interpretation:
- this is a viable **small-account manual supplement**
- it is **not** a clean primary scaling lane on Apex EOD
- the extra income is not worth the survival collapse once you size it

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

The max-EV deployed pair requires early Brisbane starts:
- COMEX_SETTLE: ~03:30/04:30
- CME_PRECLOSE: ~05:45/06:45

If you will not do `03:30 / 04:30` manually, do **not** force this pair into the manual playbook. Use the manual-first variant above and keep the full pair as a benchmark for either:
- a future automation-capable venue, or
- a future self-funded broker lane

Important policy note:
- Apex PA / Live accounts should **not** be underwritten for unattended automation.
- So "sleep through entirely" is **not** a compliant Apex operating assumption under current official docs.

The question is not whether the edge exists. It does. The question is whether you will reliably run that schedule.

---

## Phase 2b: Multi-Strategy Portfolio (Month 6+)

Phase 2 is one pair on Apex. Phase 2b adds sessions, firms, and instruments. This is how you go from "one trick" to a real portfolio without losing track of what's working.

### The Portfolio Grid

Each account gets ONE assignment. Track P&L per account, not per trade.

```
+------------------------------------------------------------------+
|  ACCOUNT GRID (example — edit as plans change)                    |
|                                                                   |
|  Account   Firm         Strategy Pair         Micros  Status      |
|  -------   ----         --------------         ------  ------     |
|  AX-01     Apex EOD     PRECLOSE + CLOSE       2c     LEAD        |
|  AX-02     Apex EOD     PRECLOSE + CLOSE       2c     COPY        |
|  AX-03     Apex EOD     PRECLOSE + CLOSE       2c     COPY        |
|  ...       ...          ...                    ...    COPY        |
|  AX-20     Apex EOD     PRECLOSE + CLOSE       2c     COPY        |
|                                                                   |
|  TS-01     TopStep      CME_REOPEN MGC         1c     LEAD        |
|  TS-02     TopStep      CME_REOPEN MGC         1c     COPY        |
|                                                                   |
|  TF-01     Tradeify     TOKYO + SINGAPORE      1c     LEAD        |
|  TF-02     Tradeify     EUROPE + LONDON        1c     LEAD        |
|                                                                   |
|  IBKR      Self-funded  ALL sessions           5-10c  MASTER      |
+------------------------------------------------------------------+
```

### Rules for the Grid

**Adding a strategy to the grid:**
1. Strategy must be in `validated_setups` with status `active` and NOT `PURGED`/`DECAY`
2. Must pass sim at >95% survival on the target firm's DD + trailing type
3. Start at 1 micro on ONE account. Run 30+ trades. Then copy.
4. Update this grid when you add it.

**Removing a strategy from the grid:**
1. Run `/regime-check`. If status = `DECAY` → pause ALL accounts running that strategy.
2. Don't close the accounts immediately — DECAY can reverse. Pause for 1 month.
3. If still DECAY after 1 month → close accounts, remove from grid.

**Changing firms:**
1. If a firm changes rules (DD, fees, account limits) → re-run the sim with new rules before continuing.
2. If survival drops below 95% → reduce micros or move strategy to a different firm.
3. This grid is the single source of truth — update it, don't keep it in your head.

**Changing strategy parameters:**
1. If you want to try a different RR or filter on a session → open a NEW account for it.
2. Don't change parameters on a running account mid-stream. That breaks tracking.
3. Run both old and new in parallel for 30+ trades, then compare.

### Which Sessions Go on Which Firms

This is driven by firm constraints, not preference:

| Constraint | Effect |
|-----------|--------|
| Apex planning lane is equity-only | MGC strategies default to TopStep or Tradeify unless Apex product eligibility is re-verified |
| TopStep close by 3:10 PM CT | No CME_PRECLOSE (5:45am Bris) on TopStep |
| TopStep max 5+1 accounts | Not the main scaling vehicle |
| Tradeify max 5 accounts | Secondary diversification, not scaling |
| Apex max 20 accounts | Primary scaling vehicle |
| MFFU unresolved / not re-verified | Do not use as a primary lane without a fresh rules pass |

**Practical assignments:**

| Session | Best Firm | Why | Fallback |
|---------|-----------|-----|----------|
| CME_PRECLOSE | Apex EOD | Main scaling pair, 20 accts | Tradeify |
| COMEX_SETTLE | Apex EOD | Main scaling pair | Tradeify |
| NYSE_OPEN | Apex EOD | High E$/mo, no restrictions | Tradeify |
| US_DATA_1000 | Apex EOD | No news restriction | Tradeify |
| EUROPE_FLOW | Apex EOD or Tradeify | Either works | TopStep |
| LONDON_METALS | Apex EOD or Tradeify | Either works | TopStep |
| TOKYO_OPEN | Any firm | No restrictions | — |
| SINGAPORE_OPEN | Any firm | No restrictions | — |
| CME_REOPEN | **TopStep first** (if MGC) | Current MGC planning lane | Tradeify (if MGC) |
| CME_REOPEN | Apex EOD (if MNQ/MES) | Fine if equity variant | — |

### Diversification Tiers

Don't diversify for the sake of it. Add complexity only when it earns more or reduces risk.

**Tier 1 (start here):** One pair, one firm, copy-traded.
- Manual-first route: Apex EOD × up to 20 accounts × `PRECLOSE + CLOSE`
- Max-EV benchmark route: Apex EOD × 20 accounts × `PRECLOSE + COMEX`
- Use the benchmark route only as a modeled ceiling unless the operating venue and policy allow it cleanly

**Tier 2 (after Tier 1 is stable):** Add a second firm for a different session cluster.
- TopStep × 3-5 accounts × MGC morning lane or other automation-capable session cluster
- Tradeify × 3-5 accounts × evening / ugly-hours cluster if current live rules still permit the operating style you want
- Now you have manual Apex income plus automation-capable or time-diversified expansion

**Tier 3 (after Tier 2 is stable):** Add different pairs on Apex.
- Some Apex accounts run NYSE + US_DATA instead of PRECLOSE + COMEX
- Reduces single-pair concentration risk
- Only do this AFTER the primary pair has 3+ months of live data

**Tier 4 (mature):** Self-funded IBKR running all sessions at higher contracts.
- No prop constraints, no account caps, no trailing DD death
- This is the graduation — prop was the proof, IBKR is the engine

### Automation-Capable Expansion Ladder

This is separate from the Apex manual-first ladder.

Use it when:
- the ugly-hour sessions still model best,
- you want execution without human wakeups,
- and the venue officially allows the automation style you plan to use.

Current institutional framing:
- `Apex`: main scaling vehicle, but do **not** underwrite unattended automation on PA / Live
- `Topstep`: cleanest officially automation-capable lane in the current doc set
- `Tradeify`: likely usable as a secondary automation-capable lane, but re-check the exact live rule pages before committing
- `IBKR / self-funded`: end-state for fully controlled automation

Suggested sequence:
1. Prove the `Apex` manual-first lane live
2. Add `Topstep` or `Tradeify` only for the sessions you do not want to handle manually
3. Keep the strategy-family tracking separate by venue
4. Promote the ugly-hours lane to self-funded infrastructure once the live edge is proven and broker integration is production-grade

### Code Integration

The portfolio grid is NOT just a markdown table. It must stay aligned with the codebase.

**Canonical sources (always query, never hardcode):**
- Active strategies: `from trading_app.live_config import LIVE_PORTFOLIO`
- Strategy fitness: `edge_families.robustness_status` (FIT/WATCH/DECAY/PURGED)
- Session times: `from pipeline.dst import SESSION_CATALOG` (resolves DST automatically)
- Cost specs: `from pipeline.cost_model import COST_SPECS`
- Active instruments: `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`

**Before adding a strategy to the grid, verify:**
```bash
# Check it exists and is active
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
print(con.execute('''
    SELECT strategy_id, status, sample_size, win_rate, expectancy_r
    FROM validated_setups
    WHERE strategy_id = 'YOUR_STRATEGY_ID'
''').fetchdf())
con.close()
"

# Check fitness (not PURGED or DECAY)
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
print(con.execute('''
    SELECT head_strategy_id, robustness_status, trade_tier
    FROM edge_families
    WHERE head_strategy_id = 'YOUR_STRATEGY_ID'
''').fetchdf())
con.close()
"
```

**Trade sheet integration:**
- Run `python scripts/tools/generate_trade_sheet.py` to see tonight's sessions with correct Brisbane times
- The trade sheet already filters to LIVE_PORTFOLIO and checks fitness
- If a strategy shows PURGED in the trade sheet, it should NOT be in your grid

**Sim validation before deploying on a firm:**
- Use `scripts/tmp_prop_sim_v2.py` or `scripts/tmp_prop_firm_proper_pass.py`
- Must test with the EXACT firm's DD amount, trailing type, and contract limits
- Survival must be >95% at your planned micro count before deploying

**Regime monitoring (run weekly or on-demand):**
```bash
# Quick fitness check
/regime-check

# Full trade book with session times
/trade-book
```

If `/regime-check` shows DECAY on a strategy you're actively trading → pause that account immediately. Don't wait for the monthly review.

### When Rules Change

Prop firms change rules regularly. When this happens:

1. **Check this file first** for the current local underwriting assumptions.
2. **Re-run the sim** with updated DD/trailing/fees. The sim scripts are in `scripts/tmp_prop_sim_v2.py` and `scripts/tmp_prop_firm_proper_pass.py`.
3. **If survival drops below 95%** → reduce micros or move to a different firm.
4. **Update the account grid** above.
5. **Update this file** if the rules change materially.

Things that change often:
- DD amounts and trailing type (Apex changed Mar 1, 2026)
- Payout splits and consistency rules
- Account count limits
- Monthly fees / activation fees
- Instrument eligibility / product availability

Things that don't change:
- Your strategies (validated in gold.db, re-checked by `/regime-check`)
- The bracket mechanics (E2, stop-market, ORB-based)
- The principle: prove at 1 micro → copy → scale micros

### Monthly Portfolio Review

In addition to the per-session monthly review (below), do a portfolio-level check:

1. **Account grid audit:** Is every account's assignment still current? Any closed/paused?
2. **Firm rules check:** Any emails from Apex/TopStep/Tradeify about rule changes?
3. **Cross-account P&L:** Is the primary pair still the best? Compare to alternatives.
4. **Concentration risk:** What % of income comes from the primary pair? If >80%, consider Tier 2/3.
5. **Payout health:** Are all accounts meeting payout qualification? If any are stalling, diagnose.
6. **Run `/regime-check`** across all strategies in the grid. DECAY = pause.

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
- Note: keep Apex scaling equity-only unless MGC eligibility is re-confirmed in current Apex support/dashboard

---

## Operational Edge Cases

These situations WILL happen. Know the answer before they do.

### Both Sides Fill (Dual Fill)

E2 places stop-market on both sides of the ORB. In fast markets (NFP, CPI, gap opens), both can trigger within seconds. This is rare but real.

**If both sides fill:**
1. You are now hedged (long + short on the same instrument). Net exposure = zero.
2. **Close both immediately.** This is a scratch, not a disaster.
3. Total cost = 2x commission + any slippage between fills. Typically < $5 on MNQ.
4. Do NOT try to "manage" one side. Close both, move on, wait for the next session.
5. Log it in your journal as "dual fill / scratch."

**Prevention:** If your platform supports OCO (one-cancels-other) brackets, use them. Tradovate supports OCO. TopstepX supports OCO. This auto-cancels the unfilled side when one triggers.

### Overlapping Positions

TOKYO fires at 10:05, still running at SINGAPORE 11:00. Do you take both?

**Rule: max 1 open position per instrument per account.**

- If TOKYO MNQ is still open when SINGAPORE MNQ fires → **skip SINGAPORE**. Do not stack.
- If TOKYO MGC is open and SINGAPORE MNQ fires → fine, different instruments.
- If the TOKYO trade exits during SINGAPORE's ORB window and SINGAPORE still qualifies → you can enter SINGAPORE. But do NOT re-measure the ORB — use the original measurement from 11:00/11:15.

This applies to ALL session pairs on the same instrument:
- EUROPE_FLOW + LONDON_METALS (both MNQ, 1 hour apart) — same rule
- CME_PRECLOSE + NYSE_CLOSE (both MNQ, 15 minutes apart) — same rule

### Cancel the Unfilled Side

After one side of the bracket fills:

**If using OCO:** Automatic. The other side cancels. Nothing to do.

**If NOT using OCO:**
1. Cancel the other stop-market order **immediately after fill notification**.
2. If you walked away: set a fill alert on your phone. When the alert fires, open the platform and cancel the other side.
3. If you forgot and both filled: see "Both Sides Fill" above.

**Platform-specific OCO setup:**
- **Tradovate (Apex):** Order entry → Advanced → OCO group. Link the two stop-market orders.
- **TopstepX:** Bracket order type. Both sides linked by default.
- **NinjaTrader:** OCO order type in the order entry window.

### Partial Fills at 2+ Contracts

At 2+ micros, what if only 1 micro fills?

**Rule: manage what fills. Same stop, same target.**

- MNQ micro futures are extremely liquid. Partial fills are very rare (< 0.1% of orders).
- If it happens: manage the 1 micro that filled with the same bracket (stop + target).
- Do NOT re-enter to "complete" the position. That's chasing.
- Log it in your journal as a partial fill.

---

## Trade Journal

Every trade must be recorded. You cannot track edge decay, diagnose problems, or prove Phase 1 readiness without records.

### Minimum Fields Per Trade

| Field | Example | Notes |
|-------|---------|-------|
| Date | 2026-03-17 | Trading day |
| Session | EUROPE_FLOW | Which session |
| Time (Brisbane) | 17:05 | When bracket was set |
| Instrument | MNQ | |
| Direction | Long | Which side of the ORB broke |
| ORB Size (pts) | 12.5 | High minus low |
| Filter | ORB_G8 PASS | Which filter, did it pass |
| Entry Price | 20,150.00 | Fill price |
| Stop Price | 20,140.63 | 0.75x from opposite edge |
| Target Price | 20,164.06 | Entry + (risk x RR) |
| Outcome | WIN / LOSS / SCRATCH | |
| P&L ($) | +$28.12 | Actual dollars after commission |
| Account | AX-01 | Which account |
| Notes | Clean fill, walked away | Any observations |

### Where to Journal

Options (pick one, stick with it):
1. **Spreadsheet** — Google Sheets or Excel. Simple, always accessible.
2. **`live_journal.db`** — the codebase has a trade journal module (`trading_app/live/trade_journal.py`). Auto-logs if using the live trading engine.
3. **Tradovate built-in** — trade history export. Less structured but zero effort.

**Minimum viable:** a Google Sheet with the fields above. Fill it out within 1 hour of each session.

### Weekly Journal Review

Every Sunday:
1. Count trades taken vs opportunities (execution rate should be > 80%)
2. Count filter overrides (should be ZERO)
3. Compare win rate to backtest expectation
4. Flag any patterns (e.g., "losing every Monday evening" → investigate)
5. If 5+ consecutive losses: mandatory — see Losing Streak Protocol below

---

## Losing Streak Protocol

At 40-55% win rate (our strategies), losing streaks are **mathematically guaranteed**:

| Win Rate | 5-Loss Streak | 8-Loss Streak | 10-Loss Streak |
|----------|--------------|---------------|----------------|
| 55% | Once per ~55 trades (~1/month) | Once per ~790 trades (~1.5 years) | Once per ~5,900 trades (rare) |
| 45% | Once per ~18 trades (~every 2 weeks) | Once per ~110 trades (~2 months) | Once per ~400 trades (~8 months) |
| 40% | Once per ~13 trades (~every 8 days) | Once per ~58 trades (~1 month) | Once per ~170 trades (~3 months) |

**A 5-loss streak at 45% win rate happens every 2 weeks on average.** This is not a bug. This is math.

### Protocol

**After 5 consecutive losses:**
1. Stop trading for the rest of the day (not the week — just today).
2. Review journal: did you follow the system on all 5 trades? (Filter check, bracket correct, no overrides?)
3. If YES (system followed correctly): **resume tomorrow, same plan, same size.** The system is working. Variance is variance.
4. If NO (overrides detected): identify which rule you broke. Write it down. Resume only after committing to follow the plan.

**After 8 consecutive losses:**
1. Take 2 days off.
2. Run `/regime-check` on the affected strategies. If DECAY → pause that session.
3. If strategies are still FIT: resume at **half size** for 10 trades, then back to full.

**After 10 consecutive losses:**
1. Full stop. Take a week off.
2. Run full `/health-check`. Review backtest-to-live comparison.
3. Do NOT resume until you understand whether this is variance or edge decay.

**NEVER DO:**
- Increase size after losses ("I need to make it back")
- Add extra sessions ("I'll trade CME_PRECLOSE at 5:45am to recover")
- Skip filters ("the ORB was 7.5 points, close enough to G8")
- Blame the system after 5 losses. Blame the system after STATISTICAL EVIDENCE of edge decay (run the test).

---

## Platform Setup Guide

### Apex 50K EOD (Primary)

**Platform:** Tradovate (web, desktop, or mobile)
1. Sign up at apextraderfunding.com → select 50K EOD evaluation
2. Receive Tradovate credentials after purchase
3. Log in to Tradovate → set to **Simulation** mode (eval account)
4. Configure order entry:
   - Default order type: **Stop Market**
   - Default TIF: **DAY** (auto-cancels at session close)
   - Enable OCO (one-cancels-other) for bracket pairs
5. Set up contract: search **MNQM6** (or current front month) → Micro E-mini Nasdaq
6. After passing eval: activate PA → select **Lifetime fee** ($160, not monthly)

**Copy trading setup (Phase 2):**
- Tradovate → Group Trading → create group with lead + follower accounts
- All followers mirror the lead account's orders automatically
- Test with 2 accounts first before scaling to 20

### TopStep $50K (MGC Morning Lane)

**Platform:** TopstepX
1. Sign up at topstep.com → select $50K Trading Combine ($49/month standard)
2. Download TopstepX platform
3. Pass the combine (profit target $3K, max loss $2K, no daily loss limit required)
4. Activate Express Funded Account ($149 one-time)
5. Configure for MGC: search **MGCM6** (or current front month)
6. Set ORB gate alert: if MGC ORB > 26 points → skip

**API access (for future auto):**
- TopstepX API via ProjectX ($14.50/month with TopStep discount)
- Live trading infra already built: `trading_app/live/projectx/`

### Tradeify $50K (Secondary)

**Platform:** Tradovate (same as Apex)
1. Sign up at tradeify.co → select 50K Select evaluation
2. Same Tradovate setup as Apex
3. After passing: select **Select Flex** payout policy (no consistency rule, 5-day payout cadence)

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
