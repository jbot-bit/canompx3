# Prop Firm Official Rules — Fetched from Support Pages

**Date fetched:** 2026-03-16
**Method:** WebFetch from official support URLs
**Status:** Partial — Apex and Tradeify block automated fetching (403). TopStep succeeded.

## IMPORTANT: Verification Required

- Apex and Tradeify sections are EMPTY — their sites return 403 to automated tools.
- These sections MUST be populated manually by visiting the URLs and copy-pasting.
- TopStep data was fetched successfully and is included below.
- ALL data here should be re-verified before making financial decisions.

---

## TopStep

### Automated Trading `OFFICIAL RULE`
Source: https://help.topstep.com/en/articles/8284097-can-automated-strategies-be-used-in-the-trading-combine-and-funded-account

- Automated strategies CAN be used in Trading Combine and Funded Accounts
- TopStep cannot assist in setup or troubleshoot automated systems
- TopStep will not assume responsibility for errant trades or malfunctions
- Must test on Practice Account first
- Must review Prohibited Conduct guidelines
- Applies to: Trading Combine, Funded Accounts, Practice Accounts

### Trade Copier `OFFICIAL RULE`
Source: https://help.topstep.com/en/articles/8284140-what-is-a-trade-copier

- Trade copiers work with: Trading Combines and Express Funded Accounts
- **Live Funded Accounts CANNOT use trade copiers** (explicit restriction)
- Cross-account copying only within same trading environment
- Bidirectional: Combine to Express or vice versa
- No limit on how many accounts receive copied trades
- Trade copier may introduce latency
- Small balance discrepancies develop over time (sequential execution)
- Hedging across accounts: PROHIBITED
- Trading in concert with others: PROHIBITED
- Supported copiers: TopstepX, Tradovate, Rithmic, Quantower (all free)

### Multiple Express Funded Accounts `OFFICIAL RULE`
Source: https://help.topstep.com/en/articles/8284218-multiple-express-funded-accounts

- Maximum 5 active Express Funded Accounts simultaneously
- Can hold different account sizes together (e.g. two 50K + three 150K)
- Hedging across accounts: PROHIBITED
- Coordinating trades with others: PROHIBITED
- TopStep has right to deny multiple accounts for any reason
- When called to Live Funded, all Express accounts close
- Individual accounts follow separate payout policies

### Multiple Funded Accounts `OFFICIAL RULE`
Source: https://help.topstep.com/en/articles/8284219-can-i-have-more-than-one-funded-level-account

- Maximum 5 Express Funded Accounts active simultaneously
- Maximum 1 Live Funded Account per trader
- When you receive Live Funded, ALL Express accounts are closed
- Cannot hold Express + Live simultaneously
- If at capacity and pass another Combine, new accounts enter pending/hold
- Shoulder Tap scenario: restricted to 1 account until it closes

---

## Apex `POPULATED 2026-03-16`

### EOD Performance Account (PA) `OFFICIAL RULE`
Source: https://support.apextraderfunding.com/hc/en-us/articles/47204516592795-EOD-Performance-Accounts-PA

**EOD PA at a Glance:**
- No intraday trailing drawdown
- EOD Drawdown calculated once per day at market close and enforced intraday
- Daily Loss Limit (DLL) enforced intraday
- Tier Based Scaling
- 100% Payout Split (upon meeting payout eligibility requirements)

**Account Parameters (50K EOD PA — our target):**
- Max Drawdown (EOD): $2,000
- Scaling: Tier Based
- Max Contracts: 4 (mini) — equivalent to 40 micro
- DLL: Tier Based
- Inactivity Rule: Yes

**Other sizes:**
- 25K: DD $1,000, Max 2 contracts
- 100K: DD $3,000, Max 6 contracts
- 150K: DD $4,000, Max 10 contracts

**Rules:**
1. Account balance (incl. unrealized PnL) may NEVER touch or fall below EOD Drawdown threshold
2. If it does → all positions auto-liquidated, PA permanently closed
3. No prohibited trading activity

### Legacy PA Compliance Rules `OFFICIAL RULE`
Source: https://support.apextraderfunding.com/hc/en-us/articles/31519788944411-Performance-Account-PA-and-Compliance

**NOTE:** This page is labeled "Legacy Performance Accounts" — may apply to older account types. EOD PA rules may differ on some points. Underwrite to the stricter interpretation.

**Automation: PROHIBITED**
- AI, Autobots, Algorithms, Fully Automated Trading Systems, HFTs: ALL PROHIBITED on PA or Live accounts
- "Hands-off, set-and-forget, set-and-walk-away, trading continuously 24 hours a day" — ALL PROHIBITED
- Using automation → immediate closure of PA or Live account, forfeiture of all funds and balances
- All software must follow User Agreement guidelines. Submit software for approval before use.
- Results from trade copiers or external programs are trader's sole responsibility

**Copy Trading: PROHIBITED**
- "PA and Live Prop Accounts must be traded by the actual individual listed on the account"
- "Not by any other party, person, system, automated trading bot, copy, or trade mirror service"
- Violation → immediate breach of contract, closure of all accounts

**Contract Scaling:**
- Trade HALF max contracts until trailing threshold reached
- 50K PA: trade up to 5 contracts initially (half of 10 max on legacy)
- Full contracts available after EOD balance exceeds trailing threshold ($52,600 on legacy 50K)
- Single violation: close excess immediately, 8 compliant days before next payout
- Repeated violations: account closure + forfeiture

**30% Negative P&L Rule (per-trade):**
- Open unrealized loss cannot exceed 30% of start-of-day profit balance (per trade)
- New accounts: 30% of trailing threshold (e.g., 30% of $2,500 = $750 on 50K)
- Once safety net exited: 30% of start-of-day profit
- If profit doubles safety net: drawdown limit increases to 50%

**30% Consistency Rule (windfall):**
- No single trading day > 30% of total profit at payout request time
- Resets after each approved payout
- Applies until 6th payout or transfer to Live Prop Account
- Formula: Highest Profit Day / 0.3 = Minimum Total Profit Required

**5:1 Risk-Reward Ratio:**
- Max risk-reward ratio of 5:1 (stop ≤ 5× profit target)
- Stop losses REQUIRED on every trade (mental stops permitted unless on Probation)

**Directional Only:**
- No hedging — cannot hold long AND short on same or correlated instruments
- Applies across ALL accounts, ALL instruments (indices, metals, grains, currencies)
- No bracket orders in both directions without directional bias

**Required: Defined System with Set Rules**
- Must have trackable, explainable strategy with set entries/stops/targets
- Apex can request: marked-up charts, live Zoom session, recording of trading sessions
- Only directional strategies with clear bias approved

**Payout Evaluation:**
- 8 trading days evaluation period
- Minimum $50 profit on 5 different trading days
- Flipping trades allowed if $50 minimum met

**Safety Net (Legacy — first 3 payouts):**
- Safety net = drawdown + $100 (e.g., 50K = $2,500 + $100 = $2,600 on legacy)
- Minimum payout $500
- For higher payouts: balance must exceed safety net by extra amount above $500

**Still need manual population:**
- [ ] EOD Drawdown Explained: https://support.apextraderfunding.com/hc/en-us/articles/45631563363483-EOD-Drawdown-Explained
- [ ] EOD Payouts: https://support.apextraderfunding.com/hc/en-us/articles/47205823183003-EOD-Payouts
- [ ] Account Limits: https://support.apextraderfunding.com/hc/en-us/articles/4406804554779-How-Many-Paid-Funded-Accounts-Am-I-Allowed-to-Have
- [ ] Consistency Rules: https://support.apextraderfunding.com/hc/en-us/articles/4404875002139-What-are-the-Consistency-Rules-For-PA-and-Funded-Accounts

---

## Tradeify `POPULATED 2026-03-16`

### Guidelines for Traders `OFFICIAL RULE`
Source: https://help.tradeify.co/en/articles/10468318-guidelines-for-traders

**Bots/Algorithmic Trading: ALLOWED (with conditions)**
- Must prove sole ownership of bot/strategy
- No one else can have access to or use it
- Tradeify scans for similar orders on other accounts
- May require live video of enabling code on your own PC
- **Exclusive use: using bot across multiple firms is AGAINST policy**
- No High-Frequency Trading (HFT) bots
- Tradeify reserves right to request documentation if flagged

**Microscalping Rule:**
- Over 50% of trades must be longer than 10 seconds
- Over 50% of profit must come from trades held longer than 10 seconds
- Fail → cannot activate evaluation or request payout

**Account Idle Time (Funded):**
- Must place at least one trade per week (Mon-Fri), per account
- Tradeify will message before taking action on idle accounts

**News/DCA/Flipping/Scaling:**
- No rules against any of these
- DCA allowed but "averaging into oblivion" (adding to losers without strategy) discouraged

**Good Faith Policy:**
- No exploiting platform errors, price discrepancies, or technical delays
- Violation → account termination

### Copy/Group Trading `OFFICIAL RULE`
Source: https://help.tradeify.co/en/articles/10468299-group-trading-copy-trading (Tradovate Order Entry guide)

**Copy Trading Rules:**
- **"You may only group trade between accounts you own and manage"**
- Copying other strategies or copy trading with others: NOT permitted
- Third-party copiers: at your own risk, no Tradeify support for errors

**Group Trading Setup (Tradovate):**
- Enable "Group Trade Add-on" in Tradovate Application Settings
- Create group, drag accounts in, set Qty per account
- Recommendation: Qty = 1 per account for max flexibility
- Order qty MUST be multiple of total group qty (accounts × per-account qty)

**CRITICAL LIMITATION:**
- **Bracket orders (ATM Strategies) are NOT supported while Group Trading**
- "Exit All Positions Cancel All" only affects selected account, NOT entire group
- Must close each account's positions separately

**Implication for our system:** E2 stop-market brackets cannot be placed via Tradovate Group Trading. Would need to place entry orders per-account via API, or use single-account manual execution and let API copy to followers.

### Account Limits `OFFICIAL RULE`
Source: https://help.tradeify.co/en/articles/10468251-how-many-simulated-funded-accounts-can-i-have-at-once

- Evaluation accounts: UNLIMITED
- **Simulated Funded Accounts: max 5 per individual user**
- **Household limit: 5 funded accounts total across all users in household**
- Mix and match account types (Growth, Select Flex/Daily, Lightning)
- Failed/expired accounts do NOT count toward limit
- Two people in same household each with 3 funded = NOT allowed (exceeds 5)

**Still need manual population:**
- [ ] Trailing Drawdowns: https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns
- [ ] Trading Times: https://help.tradeify.co/en/articles/10495876-rules-permitted-times-to-trade
- [ ] Essential Rules: https://help.tradeify.co/en/articles/12268167-essential-trading-rules-overview

---

## MFFU `NEEDS MANUAL POPULATION`

- [ ] Fair Play: https://help.myfundedfutures.com/en/articles/8444599-fair-play-and-prohibited-trading-practices
- [ ] Copy Trading: https://help.myfundedfutures.com/en/articles/10771500-copy-trading-at-myfundedfutures
- [ ] Account Limits: https://help.myfundedfutures.com/en/articles/11802636-traders-evaluation-simplified
