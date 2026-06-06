# Topstep Payout Economics — Fresh Sourcing 2026-06-06

**Purpose:** source the honesty-critical unknowns for the max-realistic-take-home
model (see `docs/plans/2026-06-06-max-realistic-takehome-model-design.md`). Every
figure annotated with article + fetch date. Firecrawl NOT installed → sourced via
WebFetch/WebSearch on official `help.topstep.com` articles + the on-disk canonical
scrape (`topstep_payout_policy.txt`, scraped 2026-04-08).

**THE BINDING ANSWER (operator question 2026-06-06 "per-withdrawal or total?"):**
The payout cap is **PER-REQUEST**, NOT lifetime/total. There is **NO maximum
number of payouts, no lifetime count limit, and no lifetime total-dollar cap** in
the policy. Each request is capped at **min(dollar cap, 50% of current balance)**.

---

## GAP 1 — Per-payout cap is TIER-SCALED (repo flat $5k/$6k was STALE)

Source: [Topstep Payout Policy, article 8284233](https://help.topstep.com/en/articles/8284233-topstep-payout-policy), fetched 2026-06-06.

> "Max payout per request: 50% of your account balance up to the cap below."

| Account size | XFA Standard cap | XFA Consistency cap |
|---|---|---|
| $50K  | **$2,000** | $3,000 |
| $100K | **$3,000** | $4,000 |
| $150K | **$5,000** | $6,000 |

**Live Funded Account payouts are NOT capped.**

**CORRECTION vs repo:** `trading_app/prop_firm_policies.py:59,93` hardcodes flat
`payout_cap_dollars=5_000` (Standard) / `6_000` (Consistency). That `$5,000` is
actually the **150K** Standard number applied to ALL tiers. For the live
`topstep_50k_mnq_auto` account the real Standard cap is **$2,000**, not $5,000 —
**less than half**. The withdrawal ceiling is far more binding than stale canon.

**FLAG (unresolved magnitude):** A "Limited Time Offering" (effective 3:30 PM CT,
Tue June 2 2026) lets you *increase* the payout cap by adding a Daily Loss Limit
at Combine checkout. Source: [DLL article 10490293](https://help.topstep.com/en/articles/10490293-daily-loss-limit-in-the-trading-combine-and-express-funded-account)
confirms the offering exists but the increased cap magnitude was not surfaced.
**Labelled assumption:** treat as optional upside, NOT baked into the base model.

## GAP 2 — Cost structure (NOT monthly subscription on funded; it's Combine-sub + one-time activation)

Sources: [Trading Combine Subscriptions 8284121](https://help.topstep.com/en/articles/8284121-trading-combine-subscriptions),
[XFA Activation 8284217](https://help.topstep.com/en/articles/8284217-express-funded-account-activation),
[Pricing 9208217](https://help.topstep.com/en/articles/9208217-topstep-pricing). Fetched 2026-06-06.

Two purchase **paths** (chosen at Combine checkout, cannot change after):

| Tier | Standard path: Combine $/mo | + XFA activation (one-time) | No-Activation-Fee path: Combine $/mo |
|---|---|---|---|
| 50K  | $49/mo  | $149 | $95/mo  |
| 100K | $99/mo  | $149 | $149/mo |
| 150K | $149/mo | $149 | $199/mo |

- **Combine subscription is monthly and STOPS once you pass** (paid only during eval).
- **XFA activation = $149 flat, one-time per account** (NOT tier-scaled).
- **Funded XFA has NO recurring monthly cost** beyond activation.
- **Reset fee = one monthly-subscription rebill** for your path/size.
- Note: the first WebSearch returned $599/$699/$829 activation — **NOT confirmed
  by any article fetch; discarded as stale/hallucinated.** Verified figures above.
- Payout processing fee: **$30** per ACH/Wire request (policy article, line 122).

## GAP 3 — Max concurrent accounts + copy-trade (bounds the multiply)

Sources: [XFA Parameters 8284215](https://help.topstep.com/en/articles/8284215-express-funded-account-parameters),
[XFA Activation 8284217](https://help.topstep.com/en/articles/8284217-express-funded-account-activation),
payout policy 8284233. Fetched 2026-06-06.

- **Max 5 active Express Funded Accounts** at once (mix of Standard/Consistency
  allowed; differs under Focused Trader Plan).
- **1 Live Funded Account** max; graduating to Live **closes all XFAs** (LFA
  starting balance = combined XFA balances up to the Live account size).
- **Copy-trading ("Trade Copier") is SANCTIONED** on Combine + XFA (NOT on LFA),
  Lead→Follower mirroring, up to **$750K combined buying power**. Explicitly
  marketed for "run the same strategy across multiple accounts."
- **Per-account independence (payout policy line 238-241):** "Each individual
  Funded Account follows the same Payout Policy. Payout requests from each account
  do not affect one another." → N accounts = N independent per-request ceilings.
- **PROHIBITED:** coordinated trading with OTHER PEOPLE, cross-account hedging,
  account farming, faked independence. (Running your OWN 5 accounts via the
  official Trade Copier is the sanctioned path, NOT farming.)
- **Copy-trade × payout interaction (policy line 220-221):** while a payout
  request is processing on an XFA in copy trading, the copy connection is
  AUTO-DISABLED and must be manually re-enabled after. → operational gap per
  payout event.

## GAP 4 — Payout frequency / cadence (per-request, gated by winning days)

Source: payout policy 8284233 (on-disk `topstep_payout_policy.txt` 2026-04-08 +
fetch 2026-06-06).

- Requests submittable during CME hours (Sun 5pm CT – Fri 5pm CT).
- **Standard:** 5 winning days of $150+ net, positive net profit since last
  payout (first payout exempt from the profit requirement). "Additional days
  after payout" before next eligibility.
- **Consistency:** 3 trading days, largest day ≤ 40% of total net profit.
- **NO calendar maximum frequency** — cadence is bounded only by how fast you
  accumulate the winning-days requirement. ⇒ annual ceiling ≈
  (payouts you can earn per year) × (per-request cap, balance-permitting) × 0.90.
- Minimum payout request: **$125**.

---

## ANTI-TUNNEL: mechanics the naive `cap × freq × 0.90` model MISSES

These bind HARDER than the headline cap and must be in the model:

1. **50%-of-balance rule is the EARLY ceiling, not the dollar cap.** To withdraw
   the full $2,000 (50K Std) you need balance ≥ $4,000 first. A bot at ~$1,005
   p50 can't reach the dollar cap early — `min($cap, 0.50×balance)` binds on the
   balance side until balance > 2× cap.

2. **MLL → $0 after first payout (policy lines 187, 214-216) = permanent survival
   tightening.** "Your Maximum Loss Limit will always be set to $0 after your
   first payout." The trailing-drawdown buffer is consumed; remaining capital
   above entry IS the entire loss budget thereafter. **Sizing must DECREASE after
   a payout, not increase** — this is the operator's withdrawal/buffer state
   machine. Withdrawing trades buffer for cash.

3. **Scaling-plan downgrade (policy lines 225-229):** a payout that drops you a
   level cuts buying power → fewer contracts → less edge. Withdraw-vs-compound
   tension is real and first-order.

4. **First-payout exemption:** first Standard payout has no positive-profit
   requirement → faster first bank, but triggers the MLL→$0 tightening.

5. **Live Funded graduation (~5 XFA payouts ≈ 25 winning days):** uncaps payouts
   (up to 100% after 30 LIVE winning days) BUT closes all XFAs and is single-
   account → trades the 5x multiplier for an uncapped single channel. A fork, not
   a free upgrade. (policy lines 17, 243, 267-280.)

---

## Reconciliation anchors (model MUST reproduce or refuse to report)

- MNQ-subset 90d DD = **$2,038.84** (canonical loader, parent session).
- Existing 3-lane `topstep_50k_mnq_auto` p50 ≈ **$1,005**.
- Tier MLLs (policy lines 195-205, matches `get_account_tier`): 50K=$2,000 /
  100K=$3,000 / 150K=$4,500. DD budget = 0.90 × MLL (express belt).
