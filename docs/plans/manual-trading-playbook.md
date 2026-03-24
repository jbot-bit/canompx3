# Trading Playbook — Brisbane

**Status:** CANONICAL prop trading playbook
**Version:** V4 — Phase 1 trimmed to CORE 5 tradeable sessions, March 24 2026
**Goal:** Prove edge live on props → scale via Tradeify/TopStep automation → self-funded IBKR for $100K/year
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

### Evidence Classes Used In This File

- `OFFICIAL RULE`: directly supported by a current firm help / compliance / pricing page listed in this file
- `LOCAL MODEL`: conclusion from local sims, backtests, or operational judgment inside this repo
- `UNRESOLVED`: plausible local belief that is not verified enough to underwrite as fact

Maintenance rule:
- Never rewrite a `LOCAL MODEL` as if it were an `OFFICIAL RULE`.
- Never promote an `UNRESOLVED` claim without adding or checking a current source.
- If a sentence affects account selection, payout expectations, automation legality, or scaling, it must be mentally classified into one of these three buckets before editing.

### Firm Rule Snapshot (mixed evidence — read labels per entry)

**Apex 50K EOD** `OFFICIAL RULE`
- Manual proof lane only (1 account). NOT the scaling vehicle.
- Official docs checked in this cycle support: `20` active PA accounts max, `4 mini / 40 micro` cap on `50K EOD`, and EOD payout gating with `5` qualifying days at `+$250` minimum, safety net `52,100`, minimum balance `52,600`, and `6` payouts max.
- Official compliance docs **prohibit** automation AND copy trading on PA / Live accounts. "PA and Live Prop Accounts must be traded by the actual individual listed on the account and not by any other party, person, system, automated trading bot, copy, or trade mirror service." Manual only.
- Important caveat: Apex documentation is not perfectly clean. The EOD payout page uses a `50%` consistency framing, while broader compliance pages still contain stricter language around windfall concentration / compliance behavior. Underwrite to the stricter interpretation until the dashboard and support match.
- Sources:
  - `https://support.apextraderfunding.com/hc/en-us/articles/4406804554779-How-Many-Paid-Funded-Accounts-Am-I-Allowed-to-Have`
  - `https://support.apextraderfunding.com/hc/en-us/articles/47204516592795-EOD-Performance-Accounts-PA`
  - `https://support.apextraderfunding.com/hc/en-us/articles/47205823183003-EOD-Payouts`
  - `https://support.apextraderfunding.com/hc/en-us/articles/31519788944411-Performance-Account-PA-and-Compliance`
  - `https://support.apextraderfunding.com/hc/en-us/articles/4404875002139-What-are-the-Consistency-Rules-For-PA-and-Funded-Accounts`

**Topstep 50K Express Funded** `OFFICIAL RULE` — verified from support pages Mar 16 2026
- Local use in this plan: MGC morning automation lane. `LOCAL MODEL`
- Automation: ALLOWED on Combine + Funded. `OFFICIAL RULE`
- Trade copier: Express only, NOT Live Funded. `OFFICIAL RULE`
- Account cap: 5 Express + 1 Live. Live kills all Express. `OFFICIAL RULE`
- Payout structure: `90/10` `UNRESOLVED` — believed correct, not re-sourced this cycle.
- Flat by `3:10 PM CT`: `UNRESOLVED` — believed correct, not re-sourced.
- Sources:
  - https://help.topstep.com/en/articles/8284097-can-automated-strategies-be-used-in-the-trading-combine-and-funded-account
  - https://help.topstep.com/en/articles/8284140-what-is-a-trade-copier
  - https://help.topstep.com/en/articles/8284218-multiple-express-funded-accounts
  - https://help.topstep.com/en/articles/8284219-can-i-have-more-than-one-funded-level-account

**Tradeify 50K** `OFFICIAL RULE` — verified from support pages Mar 16 2026
- Local use in this plan: primary MNQ automation scaling lane. `LOCAL MODEL`
- Automation: ALLOWED (exclusive ownership, no cross-firm use, no HFT). `OFFICIAL RULE`
- Copy trading: same-owner accounts only. Tradovate Group Trading does NOT support brackets. `OFFICIAL RULE`
- Account cap: 5 Simulated Funded per user/household. `OFFICIAL RULE`
- Bot exclusivity: bot must not be shared or used across multiple firms. `OFFICIAL RULE`
- Microscalping: >50% of trades must be held >10 seconds. `OFFICIAL RULE`
- Payout policy / trailing DD details: `UNRESOLVED` — not re-sourced this cycle.
- Sources:
  - https://help.tradeify.co/en/articles/10468318-guidelines-for-traders
  - https://help.tradeify.co/en/articles/10468299-group-trading-copy-trading
  - https://help.tradeify.co/en/articles/10468251-how-many-simulated-funded-accounts-can-i-have-at-once

**MFFU**
- Do not use MFFU as a primary lane in the current plan.
- Older local notes that framed this as a simple "`$129/month forever`" rejection are not clean enough to keep treating as canonical without a fresh re-check.
- Current honest position: deprioritized, not chosen, and not fully re-verified in this turn.

---

### Current Local Model Conclusions

These are the operating conclusions from local research and sim work in this repo. They are useful, but they are not firm-policy facts.

- `Apex 50K EOD` is manual proof only (1 account). Automation and copy trading are prohibited.
- `Tradeify 50K` is the primary MNQ automation scaling lane (5 accounts, Tradovate API).
- `TopStep 50K Express` is the MGC automation lane (5 Express accounts, ProjectX API).
- The overnight deployed sessions are: CME_PRECLOSE, COMEX_SETTLE, NYSE_OPEN (all MNQ on Tradeify). NYSE_CLOSE removed in V4 (not CORE 5).
- The morning deployed session is: TOKYO_OPEN (MGC on TopStep, conditional — 1 contract only until N=250). CME_REOPEN is DEAD (no validated survivors).
- `2 micros/account` is the conservative serious sizing on the current model.
- `0.75x` stop sizing remains the preferred prop-risk setting in the current local modeling.
- Prop ceiling: ~$60K/year. $100K/year target lives in self-funded IBKR (Phase 3).

Ambiguity kill:
- A separate local sim result also exists for a stricter `Apex manual ladder` built around `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5 + MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20`.
- Treat that result as `LOCAL MODEL ONLY`, not as the canonical repo plan.
- The canonical repo plan in this file is still: `Apex manual proof -> Tradeify + TopStep scaling -> self-funded IBKR`.
- If memory files or older notes imply "Apex remains the main scaling vehicle," those notes are superseded by this playbook.

If any of these change, update them as `LOCAL MODEL` conclusions, not as firm rules.

---

### Pre-Trade Checklist (Run Before Every Session)

Hard stops. If ANY line fails, do NOT trade that session.

```
[ ] Account is funded and active (not in eval, not suspended)
[ ] No open position on this instrument on this account
[ ] Checked firm's status page — no maintenance, no rule changes
[ ] News-rule check passed for this firm (Apex: `OFFICIAL RULE` — no current restriction on cited compliance page; Tradeify: `OFFICIAL RULE` — guidelines state no rules against major-news trading; TopStep: verify current help/account docs before assuming no restriction)
[ ] ORB has fully formed (waited full 5m or 15m — no early entries)
[ ] Filter passes (size >= threshold, CONT/NOMON/FAST10 if applicable)
[ ] ORB gate passes (MGC on TopStep: ORB <= 26 points)
[ ] Stop and target calculated BEFORE placing any order
[ ] Both bracket orders placed (long stop + short stop) OR OCO bracket set
[ ] Journal entry started (date, session, instrument, ORB size, filter result)
```

If you skip ANY of these, you are not trading the system. You are gambling.

### Capital Exposure Limits

**Per-account hard limits (enforce in platform risk settings where possible):**

| Limit | Value | Why |
|-------|-------|-----|
| Max open positions | 1 per instrument per account | Prevents stacking on same breakout |
| Max daily loss | $500 per account (Phase 1) | Circuit breaker — stop trading for the day |
| Max contracts | 1 micro (Phase 1), 2 micro (Phase 2 after proof) | Scale only after gates pass |
| Max accounts | 1 (Phase 1), 5+5 across two firms (Phase 2 ramp) | Follow the ramp schedule exactly |

**Portfolio-level hard limits:**

| Limit | Value | Why |
|-------|-------|-----|
| Max total open exposure | 3 positions across all accounts | Prevents correlated blowups |
| Max firm concentration | 80% of accounts on one firm | If Apex changes rules, you don't lose everything |
| Max daily portfolio loss | $1,000 (Phase 1), scales with ramp | Stop ALL trading for the day |

Set these in each platform's risk settings on day 1. Don't rely on discipline — let the platform enforce them.

### Change Control

This playbook is a living document. But changes must follow a process:

**Who can change what:**
- **Strategy parameters** (RR, filter, session) → requires sim re-validation at >95% survival before deploying
- **Firm selection** → requires official rule verification from firm's current support pages
- **Contract sizing** → requires ramp gate passage (see Phase 2 ramp schedule)
- **Adding a new session** → requires strategy in `validated_setups` with status `active`, fitness NOT PURGED/DECAY

**Change log format** (append to bottom of this file):
```
## Change Log
| Date | Change | Evidence Class | Verified By |
|------|--------|---------------|-------------|
| 2026-03-16 | Initial V3 | LOCAL MODEL + OFFICIAL RULE | Claude + Codex |
```

**What triggers a mandatory review of this playbook:**
- Email/notification from any prop firm about rule changes
- `/regime-check` shows DECAY on any active strategy
- Any account blown (even if expected — review what happened)
- Monthly portfolio review (scheduled, not reactive)
- Any sim re-run showing survival < 95% on current config

### Red Lines — Immediate Full Stop

If ANY of these happen, stop ALL trading immediately. Do not resume until resolved.

1. **Rule violation notice from a prop firm** — you may have misunderstood a rule. Stop, read, clarify with support before placing another trade.
2. **Two accounts blown in the same week** — possible regime shift or execution error. Run full `/health-check` and `/regime-check`.
3. **You override a filter** — even once. Stop for the day. Journal why. Recommit to the system.
4. **Copy trading error** — follower account takes wrong trade, double entry, or missed entry. Stop all followers, debug the copy setup.
5. **Platform outage during open position** — if you can't manage the trade, flatten everything via phone/mobile app. Don't hope.
6. **You trade while impaired** (drunk, sleep-deprived, emotional crisis) — close everything. Come back when clear-headed.

These aren't suggestions. They're circuit breakers. The system makes money over time. Any single session is irrelevant. Protecting the system IS the edge.

---

## The Plan in One Sentence

Phase 1: trade manually on Apex to prove the edge is real. Phase 2: scale via automated Tradeify (5 accts MNQ) + TopStep (5 accts MGC) for ~$60K/year. Phase 3: self-funded IBKR with $50K own capital for $100K+/year.

---

## Phase 1: Manual Proof (Months 1-3)

Trade 5 sessions manually using G-filters you can eyeball. No automation needed. Purpose is to build reps, confirm backtest-to-live match, and earn first payouts.

**Firm:** Apex 50K EOD (primary) | TopStep $50K (MGC morning lane)
**Contracts:** 1 micro per signal
**Stop:** 0.75x on prop | 1.0x on self-funded

### The 2-Session Plan (2 sit-downs)

**V4 change (2026-03-24):** Trimmed from 5 sessions to 2, per CORE 5 unfiltered baseline from 10-year regime audit. TOKYO_OPEN (p=0.010, marginal), SINGAPORE_OPEN (p=0.154, not significant), and LONDON_METALS (p=0.058, not significant) dropped from default Phase 1. Paper book runs all 5 CORE sessions in parallel — see `docs/plans/2026-03-24-mnq-core5-forward-eval-pack.md`.

All times Brisbane local (AEST, UTC+10). Brisbane has no DST — everything else shifts around you.

**Evening Block (17:00 or 18:00) — after work**

| Session | Time | Instrument | Filter | How to Check | ORB | RR | Entry |
|---------|------|------------|--------|-------------|-----|-----|-------|
| **EUROPE_FLOW** | 17:00 (winter) / 18:00 (summer) | MNQ | ORB_G8 | ORB >= 8 MNQ pts | 5m | 1.0 | E2 CB1 |

- Single session, single sit-down. 5 minutes to check, set bracket, walk away.
- G8 filter at current MNQ prices (~20K) passes most days (ORBs are large). Functions as a minimum-size cost screen.

**Night Block (23:30 or 00:30) — last thing before bed**

| Session | Time | Instrument | Filter | How to Check | ORB | RR | Entry |
|---------|------|------------|--------|-------------|-----|-----|-------|
| **NYSE_OPEN** | 23:30 (summer) / 00:30 (winter) | MNQ | ORB_G4 | ORB >= 4 MNQ pts | 5m | 1.0 | E2 CB1 |

- Lowest threshold (G4) so it fires often — NYSE_OPEN has big ORBs.
- CORE 5 session — structural edge (positive both halves of 10yr data).

### Optional: TOKYO_OPEN (regime-only, not default)

| Session | Time | Instrument | Filter | How to Check | ORB | RR | Entry |
|---------|------|------------|--------|-------------|-----|-----|-------|
| **TOKYO_OPEN** | 10:00 (fixed) | MNQ | NO_FILTER | Any break | 5m | 1.0 | E2 CB1 |

- **Marginal statistical basis:** p=0.010 unfiltered, WFE=0.53, regime-class only. NOT in CORE 5.
- Include at your discretion for morning practice — not a default Phase 1 session.
- Monitor via forward eval pack. Kill if 3 consecutive months negative.

### Optional: TopStep MGC Morning Lane (CONDITIONAL)

| Session | Time | Instrument | Filter | ORB | RR | Entry | Note |
|---------|------|------------|--------|-----|-----|-------|------|
| **TOKYO_OPEN** | 10:00 AM Brisbane (fixed, no DST) | MGC | ORB_G4_CONT | 5m | 2.0 | E2 CB1 | 1 contract only. ORB gate: skip if ORB > 26 pts |

- **CONDITIONAL edge** — per-session null P95=0.153 cleared, P99=0.364 not cleared. Between P95 and P99.
- N=125 trades (REGIME-class). **1 contract only until N=250** (~2 years at current frequency).
- MGC = $10/point (5x MNQ). Higher per-trade value but more volatile.
- ORB gate is mandatory on TopStep: if MGC ORB > 26 points, risk exceeds 10% of $2K DD. Skip.
- **Invalidation criteria:** 3 consecutive losing months OR forward ExpR < 0.10. If triggered, remove lane.
- Evidence: commits c236c57 (pooled null, P95=0.305), 3850efa (per-session null, P95=0.153).
- CME_REOPEN is DEAD (no validated survivors). Do not trade.

### Phase 1 Expected Stats `LOCAL MODEL` (from gold.db backtest)

| Session | Filter | N (10yr backtest) | ExpR | p-value | Est. Trades/Mo |
|---------|--------|-------------------|------|---------|----------------|
| EUROPE_FLOW | ORB_G8 (5m) | 2,613 | +0.101 (5yr) / -0.014 (10yr unfilt) | 3e-5 (5yr) | ~9-11 |
| NYSE_OPEN | ORB_G4 (5m) | 2,581 | +0.117 (5yr) / +0.079 (10yr unfilt) | 1e-5 (5yr) | ~8-12 |
| **TOTAL** | | | | | **~17-23** |

**~1 trade/day average.** Some days 0, some days 2.

**Note on EUROPE_FLOW:** 10-year unfiltered is negative (-0.014, p=0.41). 5-year (2021-2025) is positive (+0.101, p=3e-5). The regime audit (2026-03-24) classified this as ERA-DEPENDENT. G8 filter provides some protection at current prices. Monitor closely. Kill if 3 consecutive months negative.

**Note on filters:** G-filters are cost screens (same family as friction <10% gate). At current MNQ prices (~20K), G4/G8 pass most days. They function as minimum trade size gates, not edge predictors. See regime audit: `research/output/2026-03-24-combined-gate-stress-test.md`.

Paper book runs all 5 CORE sessions in parallel: `docs/plans/2026-03-24-mnq-core5-forward-eval-pack.md`.

### Phase 1 → Phase 2 Transition Note

Phase 1 and Phase 2 trade **different sessions**. Phase 1 validates your ability to execute brackets, follow filters, and maintain discipline — NOT the specific overnight sessions that Phase 2 scales.

Before scaling Phase 2, you MUST:
1. Complete Phase 1 gates (30+ trades, positive P&L, 2+ payouts)
2. Do 10+ manual trades on the Phase 2 sessions (CME_PRECLOSE at 05:45/06:45) on a SINGLE Apex account
3. Confirm schedule adherence — can you reliably wake at 05:30 and execute?
4. Only THEN deploy automation to additional accounts

Phase 1 proves process. The Phase 2 mini-trial proves the specific pair works for your lifestyle.

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
   - **Stop:** entry ± (ORB range × 0.75) toward the opposite edge (prop) or × 1.0 (self-funded)
   - **Target:** entry ± (ORB range × 0.75 × RR) in the breakout direction
5. **If filter fails → no trade.** Close chart. Done.
6. **After bracket set → walk away.** The bracket does the work.
7. **If no fill within session → cancel.** No chasing into the next session.

### How to Set the Bracket (E2 Stop-Market)

E2 means you place TWO stop-market orders — one above the ORB high (long), one below the ORB low (short). Whichever triggers first is your trade. Cancel the other immediately.

```
Example: MNQ EUROPE_FLOW, ORB high = 20,150, ORB low = 20,140 (10pt range, passes G8)

Stop multiplier: 0.75x prop
Risk = 10 pts x 0.75 = 7.5 pts
RR target = 1.0

LONG bracket:
  Entry:  buy stop at 20,150
  Stop:   20,150 - 7.5 = 20,142.50
  Target: 20,150 + (7.5 x 1.0) = 20,157.50

SHORT bracket:
  Entry:  sell stop at 20,140
  Stop:   20,140 + 7.5 = 20,147.50
  Target: 20,140 - (7.5 x 1.0) = 20,132.50

→ Place both. First one fills = your trade. Cancel the other.
```

---

## Decision Tree: What to Trade Today

```
START
  |
  +-- Full energy, normal day?
  |     -> Evening (17:00/18:00) + Night (23:30/00:30)
  |     -> ~1-2 signals
  |
  +-- Good energy, skip the late night?
  |     -> Evening only (EUROPE_FLOW)
  |     -> ~1 signal, no late night
  |
  +-- Night owl?
  |     -> Night only (NYSE_OPEN at 23:30/00:30)
  |     -> Strongest CORE 5 session at tradeable hours
  |
  +-- Want morning practice? (optional)
  |     -> TOKYO_OPEN at 10:00 (regime-only, marginal)
  |     -> Not default. Best human hours.
  |
  +-- Wrecked / sick / distracted?
        -> Trade NOTHING. Zero trades is a valid day.
        -> The edge is there tomorrow.
```

---

## DST Calendar: When Do Times Shift?

Both Phase 1 sessions shift with DST. Brisbane is fixed — the world moves around you.

| Period | EUROPE_FLOW | NYSE_OPEN |
|--------|-------------|-----------|
| **Jan–mid Mar** (both winter) | 17:00 | 00:30 |
| **Mid Mar** (US springs forward, UK hasn't) | 17:00 | 23:30 |
| **Late Mar–late Oct** (both summer) | 18:00 | 23:30 |
| **Late Oct–early Nov** (UK falls back, US hasn't) | 17:00 | 23:30 |
| **Nov–Dec** (both winter) | 17:00 | 00:30 |

**Optional TOKYO_OPEN: 10:00 always. No DST. No exceptions.**

---

## Phase 2: Prop Scaling (Months 3-12) `OFFICIAL RULE` + `LOCAL MODEL`

Once Phase 1 proves the edge (30+ live trades, positive P&L, 2+ payouts), scale across firms using automation.

### Firm Reality Check (grounded from official docs, Mar 16 2026)

| Firm | Automation | Copy Trading | Max Accounts | Our Role |
|------|-----------|-------------|-------------|----------|
| **Apex** | PROHIBITED | PROHIBITED | 20 (irrelevant) | Manual proof only (1 account) |
| **Tradeify** | ALLOWED (exclusive ownership, no cross-firm) | ALLOWED (same-owner, API only — GUI brackets broken in Group Trading) | 5 per household | Automation scaling lane |
| **TopStep** | ALLOWED | Express only (NOT Live) | 5 Express + 1 Live | MGC automation lane |
| **MFFU** | `UNRESOLVED` | `UNRESOLVED` | `UNRESOLVED` | Not in plan until verified |

Sources: `resources/prop-firm-official-rules.md`

**Key constraint:** Tradeify Group Trading does NOT support bracket orders (ATM). Must place brackets per-account via Tradovate API. Our automation code already does this — `TradovateOrderRouter.submit()` places orders per-account, no GUI copier needed.

### The Scaling Architecture

```
Phase 2 = Automation handles overnight sessions you can't trade manually

  Tradeify (5 accounts × MNQ via Tradovate API):
    - CME_PRECLOSE (05:45/06:45 Brisbane) — CORE 5, strongest session
    - COMEX_SETTLE (03:30/04:30 Brisbane) — CORE 5 (marginal at 10yr unfiltered, monitor)
    - NYSE_OPEN (23:30/00:30 Brisbane) — CORE 5, structural
    + any Phase 1 sessions during sleep hours
    NOTE: NYSE_CLOSE removed in V4 (not CORE 5, p=0.031, small N=595)

  TopStep (5 Express × MGC via ProjectX API):
    - TOKYO_OPEN (10:00 Brisbane) — 1 contract only, conditional edge
    - CME_REOPEN: DEAD (no validated survivors)

  Apex (1 account, manual only):
    - Phase 1 sessions during waking hours
    - No automation, no copy trading
```

### Per-Account Economics `LOCAL MODEL` (Monte Carlo sim)

**Tradeify MNQ lane** (all overnight sessions, 2 micros):

| Micros | Survival | Median $/mo | P5 $/mo |
|--------|----------|-------------|---------|
| 1 | ~100% | ~$280 | ~$85 |
| **2** | **~99%** | **~$560** | **~$170** |
| 3 | ~95% | ~$840 | — |

**TopStep MGC lane** (morning sessions, 2 micros):

| Micros | Risk/trade | $/trade | $/mo | DD risk |
|--------|-----------|---------|------|---------|
| 1 | $75 | ~$25 | ~$215 | 3.75% of DD |
| **2** | $150 | ~$50 | ~$430 | 7.5% of DD |

### Stack Math (grounded) `LOCAL MODEL`

| Lane | Accounts | Micros | Monthly | Annual |
|------|----------|--------|---------|--------|
| Tradeify MNQ × 5 | 5 | 2/acct | ~$2,800 | **~$34K** |
| TopStep MGC × 5 | 5 Express | 2/acct | ~$2,150 | **~$26K** |
| Apex manual × 1 | 1 | 1 | ~$66 | ~$800 |
| **Total prop income** | **11** | | **~$5,000/mo** | **~$60K/year** |

**$60K/year from props is the realistic ceiling.** Not $100K. The $100K target lives in Phase 3 (self-funded IBKR).

### Ramp Schedule

| Step | What | Gate Before Proceeding |
|------|------|----------------------|
| 1 | 1 Apex manual, 1 micro | Phase 1 complete (30+ trades, positive P&L) |
| 2 | 1 Tradeify automated, 1 micro | Automation passes paper-mode validation |
| 3 | 5 Tradeify + 1 TopStep, 1 micro | Zero API errors, zero missed sessions for 2 weeks |
| 4 | 5 Tradeify + 5 TopStep, 2 micros | First payout received on at least 2 accounts |
| 5 | All accounts at 2 micros | 2+ months clean, no rule breaches |

### Abort Scaling If

- Two API/execution errors in a month
- Live DD materially exceeds modeled P95 on any account
- Missed sessions become recurrent (automation reliability gap)
- Payout qualification stalls on 3+ accounts
- Any firm sends a compliance warning
- Tradeify flags your bot (they scan for similar orders across accounts)

### Apex 50K EOD — Drawdown Mechanics `OFFICIAL RULE`

Three distinct numbers — do not confuse them:

| Concept | Value | What It Means |
|---------|-------|--------------|
| **Liquidation threshold** | Starts at **$48,000** | Account closed if equity touches this intraday |
| **Safety net balance** | **$52,100** | When highest EOD close reaches this, the threshold freezes at $50,100 permanently |
| **Minimum payout balance** | **$52,600** | Safety net ($52,100) + minimum payout ($500) |

**How the EOD trailing DD works:**

1. Threshold is calculated from the **highest end-of-day closing balance**, not intraday peaks.
2. Recalculated once per day at **4:59:59 PM ET**. Enforced intraday during the next session.
3. If equity touches the threshold intraday → account liquidated/closed.
4. Threshold trails upward **only** when a new highest closing balance is set.
5. On a 50K EOD PA, the threshold **stops trailing** once it reaches **$50,100** (when highest close hits $52,100).
6. After freeze: $50,100 is a fixed floor forever. No more trailing.

**What this means operationally:**
- During the trading day, unrealized swings do NOT ratchet the threshold higher (unlike intraday trailing).
- You can be up $500 unrealized, give it back, and close flat — the threshold doesn't move because it's based on EOD close, not intraday peak.
- This gives materially more breathing room than intraday trailing accounts.

**Comparison to other Apex account types:**

| Type | DD Basis | When It Moves | Freeze Point |
|------|----------|--------------|--------------|
| **EOD PA** (our plan) | Highest EOD close | Once/day at 4:59:59 PM ET | $50,100 (when close hits $52,100) |
| Intraday Trailing PA | Highest intraday balance (incl. unrealized) | Real-time during open trades | $50,100 (same freeze) |
| Static PA | Fixed from start | Never | N/A — threshold is fixed |

Sources:
- EOD PA: https://support.apextraderfunding.com/hc/en-us/articles/47204516592795-EOD-Performance-Accounts-PA
- EOD Drawdown: https://support.apextraderfunding.com/hc/en-us/articles/45631563363483-EOD-Drawdown-Explained
- Intraday Trailing: https://support.apextraderfunding.com/hc/en-us/articles/45683513113115-Intraday-Trailing-Drawdown-Explained

### Apex EOD Payout Rules (Current March 2026) `OFFICIAL RULE`

- 5 qualifying days required (each day must be +$250 minimum)
- Minimum balance to request payout: $52,600 ($52,100 safety net + $500 minimum payout)
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

### Alternative Apex Manual Ladder `LOCAL MODEL ONLY`

This section exists to prevent ambiguity with older notes and memory files.

- The local sim work did identify a plausible strict-manual Apex pair:
  - `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5`
  - `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20`
- Modeled economics from that pass were approximately:
  - `1 micro`: `~100.0%` survival, `~$169/mo` median
  - `2 micros`: `~99.3%` survival, `~$336/mo` median
  - `3 micros`: `~96.0%` survival, `~$515/mo` median
- That implies `20 accounts x 2 micros` is roughly an `$80K/year` manual-only model, and `20 x 3 micros` is the first modeled manual path through `$100K/year`.

Decision rule:
- Keep this as an alternative `LOCAL MODEL` ladder only.
- Do **not** treat it as the canonical scaling architecture unless this file is explicitly rewritten to replace the current Tradeify + TopStep Phase 2 plan.
- Until then, the endorsed operating plan remains: Apex manual proof first, then automation-capable scaling on Tradeify / TopStep, then self-funded IBKR.

---

## Phase 2b: Account Grid (Month 6+)

Track every account, every assignment. One grid, one truth.

```
+------------------------------------------------------------------+
|  ACCOUNT GRID (edit as plans change)                               |
|                                                                   |
|  Account   Firm         Sessions               Micros  Method     |
|  -------   ----         --------               ------  ------     |
|  AX-01     Apex EOD     Phase 1 manual          1c     MANUAL     |
|                                                                   |
|  TF-01     Tradeify     PRECLOSE+COMEX+NYSE     2c     API-AUTO   |
|  TF-02     Tradeify     PRECLOSE+COMEX+NYSE     2c     API-AUTO   |
|  TF-03     Tradeify     PRECLOSE+COMEX+NYSE     2c     API-AUTO   |
|  TF-04     Tradeify     PRECLOSE+COMEX+NYSE     2c     API-AUTO   |
|  TF-05     Tradeify     PRECLOSE+COMEX+NYSE     2c     API-AUTO   |
|                                                                   |
|  TS-01     TopStep      TOKYO_OPEN MGC          1c     API-AUTO   |
|  TS-02     TopStep      TOKYO_OPEN MGC          1c     API-AUTO   |
|  TS-03     TopStep      TOKYO_OPEN MGC          1c     API-AUTO   |
|  TS-04     TopStep      TOKYO_OPEN MGC          1c     API-AUTO   |
|  TS-05     TopStep      TOKYO_OPEN MGC          1c     API-AUTO   |
|                                                                   |
|  IBKR      Self-funded  ALL 9 sessions          5-10c  API-AUTO   |
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

### Which Sessions Go on Which Firms `OFFICIAL RULE`

Driven by verified firm constraints:

| Constraint | Evidence | Effect |
|-----------|----------|--------|
| Apex: no automation, no copy trading | `OFFICIAL RULE` — PA Compliance page | Manual only, 1 account, Phase 1 proof |
| Tradeify: automation allowed, 5 accounts | `OFFICIAL RULE` — Guidelines for Traders | MNQ automation lane via Tradovate API |
| Tradeify: bot must be exclusive (no cross-firm) | `OFFICIAL RULE` — Guidelines for Traders | Separate bot instance per firm |
| TopStep: automation allowed, copier on Express only | `OFFICIAL RULE` — automation + copier pages | MGC automation lane via ProjectX API |
| TopStep: 5 Express + 1 Live, Live kills Express | `OFFICIAL RULE` — multiple accounts page | Stay on Express, don't accept Live promotion |

**Session → Firm assignments:**

| Session | Firm | Instrument | Why |
|---------|------|-----------|-----|
| CME_PRECLOSE | **Tradeify** | MNQ | Best overnight MNQ lane, automation allowed |
| COMEX_SETTLE | **Tradeify** | MNQ | Same — overnight, needs automation |
| NYSE_CLOSE | **Tradeify** | MNQ | Same Tradovate API, same accounts |
| NYSE_OPEN | **Tradeify** | MNQ | Late-night Brisbane, automation handles it |
| CME_REOPEN | ~~DEAD~~ | MGC | No validated survivors — do not trade |
| TOKYO_OPEN | **TopStep** | MGC | Conditional (P95 cleared, P99 not). 1 contract only until N=250. |
| EUROPE_FLOW | Phase 1 manual (Apex) or Tradeify | MNQ | Evening Brisbane — manual or automated |
| LONDON_METALS | Phase 1 manual (Apex) or Tradeify | MNQ | Evening Brisbane — same |
| SINGAPORE_OPEN | Phase 1 manual (Apex) or TopStep | MNQ/MGC | Morning Brisbane — manual or automated |

### Scaling Tiers

**Tier 1 (months 1-3): Manual proof.**
- 1 Apex account, manual Phase 1 sessions, 1 micro
- Purpose: prove you can execute the system

**Tier 2 (months 3-6): Add automation.**
- 1 Tradeify account, automated overnight MNQ sessions, 1 micro
- 1 TopStep Express, automated morning MGC sessions, 1 micro
- Purpose: prove the automation works reliably

**Tier 3 (months 6-9): Scale accounts.**
- 5 Tradeify accounts × 2 micros (MNQ via Tradovate API)
- 5 TopStep Express × 2 micros (MGC via ProjectX API)
- 1 Apex manual (unchanged)
- Purpose: ~$5,000/month = ~$60K/year from props

**Tier 4 (month 12+): Self-funded IBKR.**
- $50K own capital, all 9 sessions, 5-10 micros
- No prop constraints, no firm rules, DD is temporary
- Purpose: $56-112K/year — the real income engine

**Why this order matters:**
- Tier 1 proves the edge with someone else's money ($200 in eval fees, not $50K)
- Tier 2 proves the code works before you trust it with 10 accounts
- Tier 3 generates income while building the track record for Tier 4
- Tier 4 is where $100K/year actually lives — props were always the proof phase

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
6. **Re-label the edited claim mentally** as `OFFICIAL RULE`, `LOCAL MODEL`, or `UNRESOLVED` before you leave it in the file.

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

### Self-Funded IBKR (when prop edge is proven) `LOCAL MODEL` (sim projections)

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

At 40-55% win rate (our strategies), losing streaks are **mathematically guaranteed**: (geometric probability — not a model, just math)

| Win Rate | 5-Loss Streak | 8-Loss Streak | 10-Loss Streak |
|----------|--------------|---------------|----------------|
| 55% | Once per ~97 trades (~2 months) | Once per ~1,500 trades (~3 years) | Once per ~12,000 trades (rare) |
| 45% | Once per ~42 trades (~1/month) | Once per ~250 trades (~5 months) | Once per ~1,000 trades (~2 years) |
| 40% | Once per ~30 trades (~3 weeks) | Once per ~130 trades (~3 months) | Once per ~400 trades (~8 months) |

**A 5-loss streak at 45% win rate happens about once per month.** This is not a bug. This is math.

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

## Platform Setup Guide `UNRESOLVED` (platform UX can change — verify before following)

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

**Phase 2 note:** Apex is manual only — no copy trading, no automation. Scaling happens on Tradeify + TopStep via API automation (see Phase 2 section).

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
|    TOKYO_OPEN     10:00  MNQ  G5_CONT  5m ORB   RR 1.0        |
|    SINGAPORE_OPEN 11:00  MNQ  G8       15m ORB  RR 1.0        |
|                                                                |
|  EVENING (17:00-18:15 Brisbane — shifts with UK/summer DST)    |
|    First session   17:00  (see DST table for which is which)   |
|    Second session  18:00                                       |
|    EUROPE_FLOW     MNQ  G8       5m ORB   RR 1.0              |
|    LONDON_METALS   MNQ  G6_NOMON 15m ORB  RR 1.0  skip Mon   |
|                                                                |
|  NIGHT (23:30 summer / 00:30 winter Brisbane)                  |
|    NYSE_OPEN       MNQ  G4       5m ORB   RR 1.0              |
|                                                                |
|  EVERY SESSION:                                                |
|    1. Wait for ORB to form                                     |
|    2. Measure range in points                                  |
|    3. Range >= filter? Set bracket. Range < filter? Walk away. |
|    4. E2: stop-market both sides. First fill = trade.          |
|    5. Stop = entry - (ORB range x 0.75) for long (prop)        |
|    6. Target = entry + (ORB range x 0.75 x RR) for long       |
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

### Quarterly Deep Review

Every 3 months, do a full system review. This is the "step back and look at everything" check.

1. **Backtest refresh:** Re-run `python scripts/tools/pipeline_status.py --status` to check data freshness. If bars are stale, rebuild.
2. **Strategy re-validation:** Run `/validate-instrument` on all active instruments. Check for new validated strategies that should be in the grid.
3. **Firm rules audit:** Visit each firm's official support pages. Check for DD changes, fee changes, account limit changes, instrument eligibility changes. Update the Firm Rule Snapshot at the top of this file.
4. **Sim re-run:** Re-run the survival sim (`scripts/tmp_prop_firm_proper_pass.py`) with current data + current rules. If survival on any active pair drops below 95%, reduce micros or swap strategy.
5. **Concentration review:** What % of total income comes from the primary pair? The primary firm? If >80% either, plan Tier 2/3 diversification.
6. **Playbook version bump:** If anything material changed, update the version at the top and add to the change log.

---

## Change Log

| Date | Change | Evidence Class | Verified By |
|------|--------|---------------|-------------|
| 2026-03-16 | Initial V3 — consolidated manual + allocation + operating memo | LOCAL MODEL + OFFICIAL RULE | Claude + Codex |
| 2026-03-16 | Added evidence taxonomy (OFFICIAL RULE / LOCAL MODEL / UNRESOLVED) | PROCESS | Codex |
| 2026-03-16 | Added operational edge cases, journal, streak protocol, platform setup | LOCAL MODEL | Claude |
| 2026-03-16 | Added pre-trade checklist, capital limits, red lines, change control | PROCESS | Claude |
| 2026-03-16 | Code review: CME_REOPEN DST corrected, stats corrected, streak table recalculated, stop formula clarified, Phase 1→2 transition added. NOTE: SINGAPORE/LONDON aperture was incorrectly set to 5m in this pass (should be 15m per validated_setups). | LOCAL MODEL | Codex (reviewer) |
| 2026-03-16 | Evidence audit: restored SINGAPORE/LONDON aperture to 15m (validated ORB_G8=15m, ORB_G6_NOMON=15m per gold.db). Added inline LOCAL MODEL/OFFICIAL RULE/UNRESOLVED labels. Flagged 6 UNRESOLVED firm claims. Relabeled section header from "Official Rule Snapshot" to "Firm Rule Snapshot" to avoid over-trust of unverified items. Fixed pre-trade checklist news restriction claim to label TopStep/Tradeify as UNRESOLVED. | PROCESS | Claude (auditor) |
| 2026-03-24 | **V4 — Phase 1 trimmed from 5 sessions to 2** (EUROPE_FLOW + NYSE_OPEN) per CORE 5 unfiltered baseline from 10-year regime audit. TOKYO_OPEN moved to optional regime-only. SINGAPORE_OPEN, LONDON_METALS dropped (not CORE 5). NYSE_CLOSE dropped from Phase 2 Tradeify (not CORE 5). Paper book added as parallel track. Decision tree simplified. DST table simplified. Stats updated from 10-year backfill. | LOCAL MODEL (regime audit `research/output/2026-03-24-combined-gate-stress-test.md`) | Claude |
| 2026-03-16 | CRITICAL: All Phase 1 RR targets corrected from 1.5/2.0 → 1.0 (verified against family_rr_locks in gold.db). | LOCAL MODEL | Claude |
| 2026-03-16 | Added Apex EOD drawdown mechanics, post-safety-net sizing rules. | OFFICIAL RULE + LOCAL MODEL | Claude |
| 2026-03-16 | **PHASE 2 REWRITE**: Removed "20 Apex copy-traded accounts" plan (Apex prohibits automation + copy trading per official compliance page). Replaced with grounded 3-firm architecture: Tradeify 5 accounts MNQ automation via Tradovate API, TopStep 5 Express MGC automation via ProjectX API, Apex 1 account manual only. Prop ceiling ~$60K/year. $100K target moved to Phase 3 IBKR. Official firm rules populated in `resources/prop-firm-official-rules.md`. | OFFICIAL RULE + LOCAL MODEL | Claude + Codex |
| 2026-03-16 | Clarified that the `Apex manual ladder` finding is an alternative LOCAL MODEL only and does not replace the canonical `Apex manual proof -> Tradeify + TopStep scaling -> self-funded IBKR` plan. | PROCESS + LOCAL MODEL | Codex |
