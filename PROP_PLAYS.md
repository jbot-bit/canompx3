# PROP PLAYS
_Apex Trader Funding — 50K Trailing. Updated 2026-02-24._

## The Account
- **Apex 50K Trailing** — $197/month (grab 80-90% coupon → ~$20/mo)
- Trailing drawdown: **$2,500** (tracks peak unrealized balance)
- Profit target (eval): **$3,000**
- Max contracts: **10 (100 micros)** — half (50 micros) until Safety Net
- Safety Net: **$52,600** — once peak balance hits this, trail LOCKS at $50,100 forever
- Tradovate | No daily DD limit | No scaling restrictions beyond contract rule
- Consistency rule: no single day > 30% of total profit when withdrawing
- Lifetime PA activation: **$160** (don't do monthly)

---

## Why 50K Trailing
- **$2,500 DD budget = ~51R at MES.** System worst-case is 18.8R (~$920). 2.7x margin of safety.
- **Trail LOCKS at Safety Net.** Once peak hits $52,600, floor freezes at $50,100 forever. Becomes static with a bigger buffer ($2,500+) and more contracts (100 micro) than Static 100K ever had.
- **At 90% coupon: ~$20/mo eval.** Run 3-5 simultaneously for $60-100/mo total.
- **10 contracts** vs Static's 2. Room to scale to 2-3 micros per slot once trail locks.

---

## The Trailing DD Rule You Must Internalize

The trail tracks **peak unrealized balance**, not just closed P&L.

If a trade runs +$60 unrealized then reverses to your -$49 stop:
- Realized loss: $49
- Trail ratcheted up: $60
- **Total DD budget used: $109** (not $49)

Winners that almost-hit-target then reverse cost 2x on trailing. This changes which slots you prioritize.

**Mitigation:**
1. T80 time-stop on every trade (exit if target not hit by deadline)
2. Prefer RR 1.0-1.5 during eval and pre-Safety-Net phases
3. Never hold a stale trade "hoping" it comes back — the trail is eating budget every tick

---

## Phase 0 — Shotgun Eval (Week 1-2)

Wait for 80-90% coupon. Open **3x 50K Trailing** ($60/mo total).
Trade identical strategy on all three. You only need 1 to pass.

**Eval slots (prioritized for trailing safety):**

| Priority | Time (BRIS) | Slot | RR | $/R | T80 | Why |
|---|---|---|---|---|---|---|
| 1 | 3:00am | MES_CME_PRECLOSE | 1.0 | $49 | 16m | Fastest resolve, lowest phantom DD |
| 2 | 12:30am | MES_NYSE_OPEN (A0) | 1.5 | $49 | ~30m | Proven edge, moderate RR |
| 3 | 1:00am | MGC_TOKYO_OPEN | 2.0 | $72 | 32m | Highest Sharpe (4.52), bigger $/trade |
| 4 | 9:00am | MGC_LONDON_METALS | 2.0 | $72 | 36m | Best Sharpe/DD ratio in the system |

Trade all 4 from day 1 in eval. No phasing needed — it's not real money, just subscription fees.

**Eval math:**
- 4 slots, ~55% WR, average ~$40 net/trade, ~4 trades/day
- ~$160/day average → $3,000 target in **~3 weeks worst case**
- Your pace (1-2 weeks) = you're trading harder than this average

**Eval stop rules:**
- Down 3 losses in a session → stop for the day
- DD budget below $1,250 (50%) → drop to A0 + MES_CME_PRECLOSE only
- DD budget below $500 → A0 only, 1 micro, grind back or let the eval die

---

## Phase 1 — Lock the Trail (PA: $50,000 → $52,600)

**This is the danger zone. The trail is still chasing you.**

Goal: grow peak balance to $52,600. That's $2,600 of profit. Once hit, trail locks at $50,100 and never moves again.

**Phase 1 slots (conservative — trail-safe only):**

| Time (BRIS) | Slot | RR | $/R | T80 | Contracts |
|---|---|---|---|---|---|
| 3:00am | MES_CME_PRECLOSE | 1.0 | $49 | 16m | 1 micro |
| 12:30am | MES_NYSE_OPEN (A0) | 1.5 | $49 | ~30m | 1 micro |
| 9:00am | MGC_LONDON_METALS | 2.0 | $72 | 36m | 1 micro |

Max 3 concurrent. Under Safety Net you're capped at 5 contracts (50 micros) anyway.

**Phase 1 rules:**
- T80 time-stop on EVERY trade. No exceptions.
- Take profit at target immediately. Do not "let it run" — that ratchets the trail.
- Down 3 losses in a session → stop for the day
- DD budget below $1,250 → A0 + MES_CME_PRECLOSE only

**Phase 1 math:**
- 3 slots, ~$35 net/trade, ~3 trades/day = ~$105/day
- $2,600 target → ~25 trading days (~5 weeks)
- Aggressive: add MGC_TOKYO_OPEN as 4th slot → ~$140/day → ~19 days (~4 weeks)

---

## Phase 2 — Scale (Trail LOCKED, $52,600+)

**Trail is locked at $50,100. It will never move again.**
Full 10 contracts unlocked. Every dollar earned builds permanent buffer.

**Full schedule (1 micro each):**

| Time (BRIS) | Slot | RR | $/R |
|---|---|---|---|
| 11:30pm | MNQ_CME_REOPEN | 1.0 | ~$40 |
| 12:00am | MGC_CME_REOPEN | 2.5 | $74 |
| 12:30am | MES_NYSE_OPEN (A0) | 1.5 | $49 |
| 1:00am | MGC_TOKYO_OPEN | 2.0 | $72 |
| 2:00am | MNQ_SINGAPORE_OPEN | 2.0 | ~$40 |
| 3:00am | MES_CME_PRECLOSE | 1.0 | $49 |
| 9:00am | MGC_LONDON_METALS | 2.0 | $72 |
| 11:30am | MNQ_NYSE_OPEN | 2.0 | ~$40 |

8 slots. Max ~3-4 concurrent (sessions don't overlap much).

**Phase 2 income (1 micro per slot):**
- ~$460/month = ~$5,500/year per account
- At buffer $5,000+: scale to 2 micros on top slots → ~$920/month

**Phase 2 stop rules:**
- Buffer (balance - $50,100) below $1,000 → drop to Phase 1 slots
- Buffer below $500 → A0 + MES_CME_PRECLOSE only

---

## Phase 3 — Stack Accounts

Once Account #1 has 2+ payouts and is stable:
1. Open another 3x eval batch at next coupon sale
2. Pass another 50K
3. Activate (lifetime $160)
4. Run SAME strategy on both

Apex allows up to 20 funded accounts. Same strategy on all = compliance-safe.
**2 accounts at Phase 2 pace = ~$920/month. 3 accounts = ~$1,380/month.**

Scale contracts per account as buffer grows. 2 micros/slot on 3 accounts = ~$2,760/month.

---

## Payout Rules
- First payout: after 10 trading days in PA
- Frequency: up to 2/month
- Between payouts: 8 trading days, 5 showing $50+ profit
- 30% consistency: no single day > 30% of total profit at payout time
- Biggest single trade: MGC_CME_REOPEN at 2.5R = ~$185. Need $617+ total profit before first payout.
- First $25K profit: 100% yours. After: 90/10. After 5 payouts: 100%.

**Consistency tip:** bank many small green days. Don't request payout after a big day — wait until total profit dilutes it below 30%.

---

## Universal Rules (All Phases)
1. **T80 time-stop on every trade.** Non-negotiable on trailing accounts.
2. **Same instruments, same times, same sizes every day.** Apex compliance flags inconsistency.
3. **Never hold through session close.** ORB trades have defined session windows.
4. **ATR contraction + ORB compression = SKIP.** The AVOID signal applies regardless of account type.
5. **Friday high-vol at MGC TOKYO_OPEN = SKIP.** Confirmed AVOID signal.
