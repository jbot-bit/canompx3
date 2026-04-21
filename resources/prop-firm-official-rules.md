# Prop Firm Official Rules — First-Party Fetches

**Refreshed:** 2026-04-21 (deploy-live v1 Stage 1, workstream E)
**Prior version:** 2026-03-16 (preserved in git history; see diff summary below)
**Method:** WebFetch (AI-assisted extraction) from first-party help-center URLs. Extractions are summarised — exact verbatim clauses are paraphrased by the extraction model and MAY lose nuance. Where exact wording matters (enforcement, penalties, scope carve-outs), verify against the source URL before acting.
**Hash caveat:** WebFetch returns extracted text, not raw HTML. Content-hashing is NOT possible with this tool. Re-verification requires re-fetching the URL. This is a known provenance gap. Future work: use Firecrawl for raw HTML + SHA-256.
**Scope of this refresh:** TopStep, Bulenox, MFFU (per deploy-live directive). Apex and Tradeify sections retained from 2026-03-16 fetch as DEPRECATED — both flagged dead for ORB in `memory/MEMORY.md`; no re-fetch performed today.

---

## Critical findings (must read)

### 1. TopStep copier policy REVERSED vs 2026-03-16 record

- **2026-03-16 record (now stale):** "Live Funded Accounts CANNOT use trade copiers" (article 8284140, now 404).
- **2026-04-21 canonical (Live Funded Account Parameters — updated this week, 2026-02-19 for balance calcs):** "Trade copiers are allowed: You can duplicate trades across multiple accounts using a trade copier" on Live Funded Accounts.
- **Implication:** The multi-firm / multi-account scaling plan (TopStep + Bulenox + MFFU) is NOT blocked by a TopStep copier ban on LFA, contrary to the prior record. This is a material policy change in TopStep's favour.
- **Action:** Re-verify with TopStep support before any capital commits — extraction is AI-summarised, and the reversal is large enough to warrant human confirmation.

### 2. TopStep: ProjectX API prohibited on Live Funded

- **Source:** `https://help.topstep.com/en/articles/10657969-live-funded-account-parameters`, extraction 2026-04-21.
- **Exact extracted phrase:** "Automated trading through the ProjectX API is prohibited with Live Funded Accounts."
- **Repo implication:** `trading_app/live/projectx/` exists. If TopStep LFA is wired through ProjectX → RULE VIOLATION. Route LFA through **Rithmic** or **Tradovate** (both supported per older TopStep copier article). Encoded as routing constraint in Stage 5.1b (Task #17).
- **Status:** CONFIRMED via live fetch. Gate B routing must reflect.

### 3. MFFU cross-account strategy coordination — AMBIGUOUS

- **Source:** `https://help.myfundedfutures.com/en/articles/8444599-fair-play-and-prohibited-trading-practices`, extraction 2026-04-21.
- **Extracted clause:** "Coordinating identical/opposite strategies across separate accounts."
- **Scope ambiguity:** Does "separate accounts" mean MFFU accounts only, or across firms? Article does not clarify. Running the same edge on MFFU AND another firm simultaneously *might* fall under this rule. Extraction model could not resolve.
- **MFFU copy trading article (10771500) separately states:** copying is "allowed across all account types" + Tradesyncer + Tradovate group copier + external copiers permitted.
- **Tension:** copy-trading policy permits copier use; fair-play policy prohibits coordinated strategies across accounts. Either the copier-is-OK applies within MFFU and not cross-firm, or the coordinated-strategies rule targets abuse (multi-account arbitrage) rather than same-edge replication. **Cannot resolve via extraction. Ask MFFU support for written clarification before putting the same automated lane on MFFU + any other firm.**
- **Status:** UNRESOLVED. Flagged as abort trigger candidate for Stage 5.1e (Task #20) — account binding halts if MFFU conflicts with same-lane-elsewhere.

### 4. Bulenox Funded Account gated behind 3 Master payouts

- **Source:** `https://bulenox.com/help/funded-account/`, extraction 2026-04-21.
- **Extracted clause:** "Upon the completion of three (3) successful payouts under a Master Account" traders transition to a Funded Account.
- **Implication for deployment plan:** Bulenox "Funded" (Pro) is NOT the starting account; Master is. Self-funded scaling plan's "Bulenox" line-item should start with Master.
- **Status:** CONFIRMED via live fetch.

### 5. Bulenox auto / copier / news rules — NOT FOUND in fetched articles

- Help-center pages fetched (qualification-account, master-account, funded-account, help-root FAQ) do not explicitly state rules for automated trading, copy trading, or news embargo.
- Terms of Use PDF (July 2021) is binary-encoded and text-extraction failed via WebFetch.
- **Status:** **UNVERIFIED**. Do not assume Bulenox allows automation or copiers. Must request written confirmation from Bulenox support before Stage 4 XFA wiring or Stage 5 broker routing commits a Bulenox account. Per directive: no training-memory fallback.

---

## TopStep

### Live Funded Account Parameters `FETCHED 2026-04-21`

- **Source:** `https://help.topstep.com/en/articles/10657969-live-funded-account-parameters`
- **Last updated on page:** "Updated this week" (balance calcs timestamped 2026-02-19)

**Automated trading:**
- Automated strategies ARE permitted on Live Funded Accounts.
- Trade copiers ARE allowed — "You can duplicate trades across multiple accounts using a trade copier."
- **EXCEPTION: ProjectX API automation is prohibited on LFAs.**
- TopStep disclaims support: "Topstep cannot help set up or troubleshoot automated strategies and will not make exceptions for errant trades or malfunctions."

**Risk / loss limits:**
- Daily Loss Limit: $2,000 (50K) / $3,000 (100K) / $4,500 (150K).
- Dynamic adjustment: if tradable balance ≤ $10,000, DLL = $2,000.
- Max Loss: balance drop below $1,000 → account liquidated immediately.

**Capital scaling:**
- Starting available: 20% or $10,000 minimum.
- 25% of Reserve Balance released per profit threshold (4 thresholds total).
- Capital review: Risk Team, each Monday.

**Missing from fetched page:** min trading days, news embargo, consistency specifics → labelled UNVERIFIED until linked articles fetched.

### Trading Combine Parameters `FETCHED 2026-04-21`

- **Source:** `https://help.topstep.com/en/articles/8284197-trading-combine-parameters`
- **Last updated on page:** "This week" (exact date not rendered).

**Position size (combined mini/micro at 10:1):**
- $50K → 5 minis / 50 micros
- $100K → 10 minis / 100 micros
- $150K → 15 minis / 150 micros

**Consistency:** "Best Day below 50% of total profits made."

**Automated trading:** "Yes, automated strategies are permitted, with a few things to note" — TopStep does not troubleshoot; review prohibited conduct first.

### Hedging `FETCHED 2026-04-21`

- **Source:** `https://help.topstep.com/en/articles/13747047-understanding-hedging`
- **Last updated on page:** "Over a week ago."

**Definition:** "Cross-account hedging occurs when you hold opposite positions across multiple accounts at the same time."

**Prohibited:**
- Simultaneous long+short same instrument across different accounts.
- Coordinating positions to offset risk across Combines, XFAs, LFAs.
- Activity that manipulates/abuses per CME Rule 534.

**Multi-account rules:**
- SAME markets across different accounts: allowed.
- OPPOSITE positions: prohibited.
- Violations tracked at trader level, apply across all accounts.
- Scope: Combine, XFA, LFA.

**Automated systems:** "You remain fully responsible for all activity across your accounts, including positions created through copy trading software, automated trading systems, or any third-party tools."

**Enforcement (staged):**
1. First detection → real-time warning + un-hedge window → auto-liquidation if not corrected.
2. Same-day repeat → temporary violation, trading prohibited rest of day.
3. Post-acknowledgment → permanent violation, accounts closed.

### Prohibited Trading Strategies `FETCHED 2026-04-21`

- **Source:** `https://help.topstep.com/en/articles/10305426-prohibited-trading-strategies-at-topstep`
- **Last updated on page:** "Over a year ago." → content may be stale relative to LFA article.

**Prohibited activities (extracted list):**
1. Account stacking (aggressive trade → hit Max Loss → switch accounts).
2. Intentional account depletion.
3. Terms-of-service violations.
4. Automated / AI / ultra-high-speed / mass-data-entry that manipulates or abuses.
5. Market-manipulative practices inconsistent with real futures markets.
6. Trading outside best bid/offer.
7. Purposefully trading Max Position Size into a major news event.

**Carve-outs / penalties:** not detailed in the extracted excerpt.

### Search / UNVERIFIED linked articles

- Daily Loss Limit, Max Loss Limit (separate articles) — URLs not captured in this run.
- Permitted Products and Trading Hours — not fetched.
- Consistency Target specifics — not fetched.

---

## Bulenox

### Qualification Account `FETCHED 2026-04-21`

- **Source:** `https://bulenox.com/help/qualification-account/`
- **Last updated on page:** 2023-08-14.

**Schedule:** 5 PM – 4 PM CT trading day; all positions closed by 15:59 CT; weekends/holidays excluded.

**Minimum trading days:** zero (for Master-account transition).

**Position / contracts:** multiple positions allowed; std + micro mixable (1 std = 10 micros); max position size varies by account type (not tabulated on this page).

**Drawdown — two options:**
- **Option 1 (No Scaling / Trailing):** drawdown always follows current balance; real-time with commissions; breach → account blocked.
- **Option 2 (EOD):** updated EOD; dynamic scaling on cash on hand; includes DLL ($400–$4,500 by size); DLL suspension is not a rule violation.

**Data fee:** Professional $116/month; non-professional included.

### Master Account `FETCHED 2026-04-21`

- **Source:** `https://bulenox.com/help/master-account/`
- **Last updated on page:** 2023-02-21. **STALE — nearly 3 years old.**

**Trailing / EOD Drawdown (TDA) by size:**
| Account | TDA |
|---|---|
| $25K | $1,500 |
| $50K | $2,500 |
| $100K | $3,000 |
| $150K | $4,500 |
| $250K | $5,500 |

**Drawdown freeze:** "Stops moving when [it] reaches the initial starting balance +$100."

**Minimum trading days for first payout:** 10 individual trading days.

**Payout schedule:** weekly, Wednesdays.

**Profit split:** first $10,000 to trader 100%; beyond that 90/10 in trader's favour.

**Withdrawal min:** $1,000. First three payouts capped ($1,000–$2,500 by account size).

**Consistency rule (40%):** at withdrawal request, no single trading day's profit may exceed 40% of total profit balance.

**NOT ADDRESSED on this page:** automated trading, copier, news trading, prohibited conduct.

### Funded Account (Pro) `FETCHED 2026-04-21`

- **Source:** `https://bulenox.com/help/funded-account/`
- **Effective date on page:** 2025-04-28 (balance caps).

**Qualification:** 3 successful payouts under a Master Account required to transition.

**Balance caps:**
| Account | Cap |
|---|---|
| $25K | $2,500 |
| $50K | $5,000 |
| $100K | $10,000 |
| $150K | $15,000 |
| $250K | $25,000 |

**Min trading days per reward:** 5 individual trading days.

**Consolidation:** all active Master Accounts consolidate into a single Funded Account.

**NOT ADDRESSED on this page:** drawdown specifics for Funded, daily loss, max loss, profit split, auto/copier/news/prohibited conduct.

### Terms of Use (July 2021) `FETCH_FAILED 2026-04-21`

- **Source:** `https://bulenox.com/wa-data/public/site/data/bulenox.com/Terms_of_Use.pdf`
- **Status:** UNVERIFIABLE via WebFetch (binary/compressed PDF). Per directive, no training-memory fallback. Future work: extract locally with PDF reader, re-ingest as text, re-hash.

### Bulenox — rules not verified

- Automated trading policy: **UNVERIFIED**.
- Copy-trading policy: **UNVERIFIED**.
- News trading policy: **UNVERIFIED**.
- Prohibited conduct enumeration: **UNVERIFIED**.
- Scaling plan (beyond 2023-02-21 page): **UNVERIFIED**.

---

## MFFU (MyFunded Futures)

### Evaluation Parameters `FETCHED 2026-04-21`

- **Source:** `https://help.myfundedfutures.com/en/articles/11802636-traders-evaluation-simplified`
- **Last updated on page:** "Over 3 weeks ago."

**Flex Plan (25K / 50K):**
- Profit Target: $1,500 / $3,000
- Max Loss (EOD): $1,000 / $2,000
- Daily Loss Limit: None
- Max Contracts: 3 mini / 30 micro (25K); 5 mini / 50 micro (50K)

**Pro & Rapid Plans (25K / 50K / 100K / 150K):**
| Size | Profit Tgt | Max Loss (EOD) | Max Contracts (Rapid) | Max Contracts (Pro) |
|---|---|---|---|---|
| $25K | $1,500 | $1,000 | 3 mini / 30 micro | 3 mini / 30 micro |
| $50K | $3,000 | $2,000 | 5 mini / 50 micro | 6 mini / 60 micro |
| $100K | $6,000 | $3,000 | 10 mini / 100 micro | 9 mini / 90 micro |
| $150K | $9,000 | $4,500 | 15 mini / 150 micro | varies |

**Universal:**
- Consistency rule: 50% (Evaluation only).
- Min trading days: 2.
- T1 News Trading: yes (prohibited — see News Policy).
- No DLL across any plan.

### Payout Policy `FETCHED 2026-04-21`

- **Source:** `https://help.myfundedfutures.com/en/articles/13745661-payout-policy-overview-best-and-fastest-prop-firm-payouts`
- **Last updated on page:** "Over 2 months ago."

**Rapid:**
- Daily payout requests; first eligible 24 hours after first trade.
- Profit split: 90% (sim-funded plans, effective 2026-01-12).
- Min withdrawal: $500.
- No consistency rules.
- Buffers: 50K $2,100 / 100K $3,100 / 150K $4,600.

**Flex:**
- 5 winning days minimum before payout.
- 80% split.
- Min withdrawal: $250.
- Earn caps per plan: 25K $3,000 / 50K $5,000.
- After first payout: Max Loss Limit resets to $100.

**Pro:**
- Every 14 calendar days from first trade.
- 80% split.
- Min withdrawal: $1,000.
- Max cap: $100,000 per user in sim-funded stage.
- Buffers: 50K $2,100 / 100K $3,100 / 150K $4,600.
- Withdrawal while in buffer: up to 60% of profits.

**Processing:** most instant; manual review 6–12 business hours.

### News Trading Policy `FETCHED 2026-04-21`

- **Source:** `https://help.myfundedfutures.com/en/articles/8230009-news-trading-policy`
- **Last updated on page:** "Over 2 months ago."

**Embargo window:** no open positions OR active orders 2 minutes before AND after any data release.

**Tier 1 (prohibited for Rapid Sim Funded + Pro Sim Funded; permitted for evaluations + 25K/50K Flex):**
- FOMC Meetings, FOMC Minutes, Employment Report, CPI (all traders).
- EIA (energy traders).
- Agricultural Reports (agricultural traders).

**Rationale stated:** "News Trading is not simulated 1:1 with live markets, it is not allowed."

### Fair Play and Prohibited Trading Practices `FETCHED 2026-04-21`

- **Source:** `https://help.myfundedfutures.com/en/articles/8444599-fair-play-and-prohibited-trading-practices`
- **Last updated on page:** "Over 5 months ago."

**Automated (conditional stack — ALL conditions must hold simultaneously):**
- **HFT prohibited** (unconditional).
- **Automation allowed only if NOT exploiting favourable simulated fills.** The "permitted only if" phrasing is a conditional permission, not a blanket allowance — exploitation of sim-fill quirks (tight brackets beating real-spread slippage, fills across gapped/illiquid bars, etc.) voids the permission and becomes a violation.
- **Live-account automation must comply with CME guidelines** (e.g., CME Rule 534).
- Prior-draft summary of this block read too permissive — the clause is a stacked conditional, NOT three independent permissions.

**Order management / market conduct prohibited:**
- Simultaneous multiple limit orders at identical prices to manipulate fills.
- Trading gapped / illiquid markets to profit from isolated fills.
- Exploiting absence of slippage with tight brackets.
- Coordinating identical/opposite strategies across separate accounts (see Critical Finding #3 — scope ambiguous).

**Copy / device:**
- Copy-trading BETWEEN TRADERS prohibited (enter/exit/cancel each other's positions).
- Device sharing prohibited → permanent service restriction.

**Hedging:**
- Of ANY kind prohibited (same-asset buy + sell simultaneous). E-mini NQ + micro NQ counted as same (NQ index). CME Rule 534 compliance.

**Enforcement:** account termination, profit confiscation, evaluation review denial, ineligibility for funding/refunds.

### Copy Trading Policy `FETCHED 2026-04-21`

- **Source:** `https://help.myfundedfutures.com/en/articles/10771500-copy-trading-at-myfundedfutures`
- **Last updated on page:** "Over a month ago."

**Permitted:**
- Copy trading ALLOWED across all account types.
- Tradesyncer integration supported.
- Tradovate built-in group copier supported.
- External copier solutions can be used.

**Liability:** user assumes full responsibility; MFFU disclaims liability; no technical support, troubleshooting, or account resets for copier issues.

**NOT ADDRESSED:**
- Whether SAME strategy across multiple personal MFFU accounts is permitted (ambiguous vs Fair Play clause — see Critical Finding #3).
- Cross-firm copying.
- Rithmic R|API compatibility specifically.
- Bracket order behaviour under copier.
- Penalties for violations.

### Hedging `FETCHED 2026-04-21`

- **Source:** `https://help.myfundedfutures.com/en/articles/12011241-hedging-what-you-should-know`
- **Last updated on page:** "Over 5 months ago."

- Hedging of any kind prohibited.
- Definition: simultaneous buy + sell same underlying at same time.
- E-mini + micro NQ = hedging (same NQ index).
- CME Rule 534 compliance required.
- Hedging through unrelated assets permissible; heavy reliance may hinder trader evaluation.
- **Not addressed:** multi-account hedging; whether copier usage counts as hedging; penalties.

### Funded-account articles — NOT fully fetched

- `Keys to Managing Your Funded Account` — generic guidance, no specific drawdown numbers.
- `2% Price Limit Rule`, `Cross Instrument Policy`, `Inactivity Rule`, `Activation Fee` — URLs captured, content not extracted this run.

---

## Apex — DEPRECATED 2026-03-16 fetch retained for audit only

**Status:** NOT re-fetched 2026-04-21. Memory flags Apex dead for ORB. Retained below purely as historical record. Any production use requires re-fetch + re-verification.

### EOD Performance Account (PA) `OFFICIAL RULE — 2026-03-16`
Source: https://support.apextraderfunding.com/hc/en-us/articles/47204516592795-EOD-Performance-Accounts-PA

- No intraday trailing drawdown. EOD drawdown calc once/day at market close, enforced intraday.
- DLL enforced intraday. Tier-based scaling. 100% payout split at eligibility.

**50K EOD PA (target):** Max DD $2,000 EOD; 4 mini / 40 micro; tier DLL; inactivity rule yes.
**Other sizes:** 25K $1,000 / 2 contracts; 100K $3,000 / 6 contracts; 150K $4,000 / 10 contracts.

### Legacy PA Compliance (may differ from EOD PA) `OFFICIAL RULE — 2026-03-16`
Source: https://support.apextraderfunding.com/hc/en-us/articles/31519788944411-Performance-Account-PA-and-Compliance

- **Automation PROHIBITED.** AI / autobots / algos / HFT / hands-off systems all prohibited → immediate closure + forfeiture.
- **Copy trading PROHIBITED.** "PA and Live Prop Accounts must be traded by the actual individual listed on the account" — no bots / mirrors / services.
- Contract scaling: half max until trailing threshold reached.
- 30% per-trade negative P&L rule.
- 30% consistency rule (windfall) at payout until 6th payout or Live transfer.
- 5:1 max RR. Stops required.
- Directional only; no hedging across accounts/instruments.
- 8-trading-day payout evaluation, $50 min profit on 5 different days.

### Tradeify — DEPRECATED 2026-03-16 fetch retained for audit only

**Status:** NOT re-fetched 2026-04-21. Memory flags dead for ORB.

- Bots allowed with conditions: sole ownership, no sharing, HFT prohibited, exclusive per-firm use (cross-firm prohibited).
- Microscalping rule: 50%+ of trades > 10s.
- Copy trading: only between accounts you own/manage.
- **Critical platform limitation:** Tradovate Group Trading does NOT support bracket orders → blocks E2 stop-market brackets in copy mode.
- Account limits: Eval unlimited; Sim Funded 5 per user, 5 per household.

---

## Diff summary vs 2026-03-16 record

| Firm / Rule | 2026-03-16 | 2026-04-21 | Effect on plan |
|---|---|---|---|
| TopStep LFA copier | Prohibited | **Allowed** | UNBLOCKS multi-account scaling on TopStep side |
| TopStep auto on LFA via ProjectX | Not stated | **Prohibited** | ROUTING CONSTRAINT — use Rithmic/Tradovate for LFA, not ProjectX |
| TopStep hedging enforcement | Not stated | Staged 3-step (warning → block → closure) | Elevates cross-account risk — stricter same-lane discipline |
| Bulenox Funded qualification | Unclear | 3 Master payouts required | Funded is not a starting account — must do Master first |
| Bulenox auto / copier / news | Absent | **UNVERIFIED** (first-party sources silent) | BLOCKS Bulenox deployment until written confirmation obtained |
| MFFU copy trading | Absent | Allowed (Tradesyncer / Tradovate / external) | UNBLOCKS MFFU copier path |
| MFFU cross-account coordination | Absent | **AMBIGUOUS** (same-edge-across-firms could be "coordinated") | Pending Stage 5.1e resolution — may require written support clarification |
| MFFU news (Rapid + Pro Sim Funded) | Not captured | T1 events PROHIBITED | News gate required at order layer |
| MFFU hedging (intra-instrument) | Not captured | E-mini + micro NQ counted as same | Same-underlying hedging constraint on strategies using both |

---

## Open items (for ops follow-up, NOT this run)

1. Human confirmation of TopStep LFA copier reversal before any LFA capital commit.
2. Bulenox support written confirmation: auto / copier / news / prohibited conduct.
3. MFFU support written confirmation: "coordinating identical/opposite strategies across separate accounts" vs copy-trading allowance — does same-edge across firms qualify?
4. Bulenox Terms-of-Use PDF: local text extraction, re-ingest, SHA-256 hash.
5. TopStep: fetch DLL, Max Loss Limit, Permitted Products, Trading Hours articles (linked but not captured this run).
6. MFFU: fetch Keys to Managing Funded, 2% Price Limit, Cross Instrument, Inactivity — numeric rules not captured.

---

## Provenance index (2026-04-21 fetches)

| URL | Fetch timestamp (Brisbane) | Extraction method | Status |
|---|---|---|---|
| `https://help.topstep.com/en/articles/8284097-...` | 2026-04-21 | WebFetch | **404 — URL retired** |
| `https://help.topstep.com/en/articles/8284140-...` | 2026-04-21 | WebFetch | **404 — URL retired** |
| `https://help.topstep.com/en/articles/10657969-live-funded-account-parameters` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.topstep.com/en/articles/8284197-trading-combine-parameters` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.topstep.com/en/articles/13747047-understanding-hedging` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.topstep.com/en/articles/10305426-prohibited-trading-strategies-at-topstep` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://www.topstep.com/rules/` | 2026-04-21 | WebFetch | 404 |
| `https://help.topstep.com/en/collections/4402747-funded-accounts` | 2026-04-21 | WebFetch | 404 |
| `https://help.topstep.com/en/collections/3498930-funded-accounts` | 2026-04-21 | WebFetch | 404 |
| `https://bulenox.com/` | 2026-04-21 | WebFetch | OK (navigation-only) |
| `https://www.bulenox.com/rules` | 2026-04-21 | WebFetch | 404 |
| `https://bulenox.com/help/` | 2026-04-21 | WebFetch | OK (navigation-only) |
| `https://bulenox.com/help/qualification-account/` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://bulenox.com/help/master-account/` | 2026-04-21 | WebFetch | OK (summary-extracted; page last-updated 2023-02-21) |
| `https://bulenox.com/help/funded-account/` | 2026-04-21 | WebFetch | OK (summary-extracted; balance caps effective 2025-04-28) |
| `https://bulenox.com/help/frequently-asked-questions/` | 2026-04-21 | WebFetch | OK (4 FAQs only; not comprehensive) |
| `https://bulenox.com/wa-data/public/site/data/bulenox.com/Terms_of_Use.pdf` | 2026-04-21 | WebFetch | **FETCH_FAILED** (binary PDF — UNVERIFIABLE via this tool) |
| `https://myfundedfutures.com/` | 2026-04-21 | WebFetch | OK (navigation) |
| `https://myfundedfutures.com/rules` | 2026-04-21 | WebFetch | 404 |
| `https://myfundedfutures.com/terms` | 2026-04-21 | WebFetch | OK (T&C general; no numeric rules) |
| `https://help.myfundedfutures.com/en/` | 2026-04-21 | WebFetch | OK (index) |
| `https://help.myfundedfutures.com/en/articles/8230009-news-trading-policy` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.myfundedfutures.com/en/articles/8444599-fair-play-and-prohibited-trading-practices` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.myfundedfutures.com/en/articles/12011241-hedging-what-you-should-know` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.myfundedfutures.com/en/articles/8528346-keys-to-managing-your-funded-account` | 2026-04-21 | WebFetch | OK (no specific numerics) |
| `https://help.myfundedfutures.com/en/articles/10771500-copy-trading-at-myfundedfutures` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.myfundedfutures.com/en/articles/11802636-traders-evaluation-simplified` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.myfundedfutures.com/en/articles/13745661-payout-policy-overview-...` | 2026-04-21 | WebFetch | OK (summary-extracted) |
| `https://help.myfundedfutures.com/en/collections/5808821-traders-evaluation` | 2026-04-21 | WebFetch | OK (index) |
| `https://help.myfundedfutures.com/en/collections/5808827-funded-accounts` | 2026-04-21 | WebFetch | OK (index) |
| `https://help.myfundedfutures.com/en/collections/5811873-payout-information` | 2026-04-21 | WebFetch | OK (index) |
| `https://help.myfundedfutures.com/en/collections/14650040-trading-practices-and-risk-management` | 2026-04-21 | WebFetch | OK (index) |

**Hash note:** Per above, WebFetch returns extracted text, not raw HTML, so SHA-256 hashing is not possible with this tool. Flagged as future work.
