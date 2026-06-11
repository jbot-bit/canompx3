# Prop Firm Ranking for the canompx3 Bot - 2026

**Prepared:** 2026-06-11  
**Scope:** Fully automated, copy-routed micro-futures ORB bot (MNQ / MGC / MES)  
**Decision:** Where to buy/route funded accounts if Topstep is capping growth.

This is not a generic prop-firm review. It is a deployment audit for this bot.

## 0. Evidence Labels

- **MEASURED:** confirmed from current repo files or an official firm page scraped on 2026-06-11.
- **INFERRED:** logical consequence of measured rules, but not directly stated by the firm.
- **UNSUPPORTED:** review-site, affiliate, forum, or vendor claim not corroborated by an official source.

Primary sources are separated from unofficial signals in the source list. Firm rules change quickly; re-check official pages immediately before buying accounts or routing live orders.

## 1. Bot Hard Filters

The bot needs all of these:

1. Full automation, not just semi-automated click-assist.
2. Copy routing across accounts owned by the same trader.
3. EOD/static drawdown preference. Intraday trailing drawdown is hostile to ORB giveback.
4. Tradovate, Rithmic, ProjectX/TopstepX, or another platform we can actually drive.
5. No hidden "same strategy across firms" clause that invalidates a portfolio-level copy router.

Repo authority for current implementation:

- `trading_app/prop_profiles.py` currently models Topstep, Tradeify, Bulenox, MyFundedFutures variants, and self-funded.
- It does not model TakeProfitTrader, Apex, Earn2Trade, Phidias, or Alpha Futures as runnable profiles.
- `topstep_50k_mnq_auto` is the active profile; Tradeify and Bulenox profiles exist but are inactive / integration-gated.

Research/risk grounding:

- Criterion 11 requires account-death Monte Carlo with firm daily loss, trailing DD, and consistency rules; survival must be at least 70% over 90 days before deployment.
- Carver risk sizing says prop-style small drawdown accounts force low effective volatility targets; a larger notional account only matters if it increases usable drawdown buffer and payout capacity, not because the headline notional is bigger.
- Carver portfolio rules warn that diversification scaling must be correlation-aware and capped; stacking identical copies is not true diversification.

## 2. Current Verdict

**MEASURED:** Topstep is capping growth. It remains useful, but it is not the growth engine.

**Best next firm sleeve:** Tradeify Select Flex, preferably 150K if C11 survives the larger drawdown/profit-target profile and the exclusive-use bot clause is handled.  
**Second sleeve:** MyFundedFutures Flex/Pro/Builder, chosen plan-by-plan to avoid Rapid's intraday trailing drawdown.  
**Third sleeve:** Bulenox EOD Master, but downgrade the old "11-account long-term scaler" claim because official Funded Account rules now say active Master accounts can be consolidated into one Funded Account after three payouts at risk-management discretion.  
**Keep:** Topstep XFA as the current ProjectX-native sleeve.  
**Do not use for this bot:** TakeProfitTrader, Apex, Earn2Trade, Phidias, unless the execution mode changes to genuine human-in-the-loop/semi-automated.  
**Watchlist only:** Alpha Futures. Official language restricts HFT/100+ trade-day automation, but does not clearly approve fully hands-off bots; needs written compliance approval before build work.

## 3. Ranking

| Rank | Firm / plan | Bot fit | Why | Main blocker |
|---|---:|---|---|---|
| 1 | Tradeify Select Flex | **MEASURED: best current target** | Officially allows owned bots under conditions; own-account copy is permitted; EOD trailing; 5 funded accounts; 150K has $4.5K EOD DD and 120 micro cap. | Bot must be exclusive to Tradeify; cross-firm identical bot use is against policy unless approved. |
| 2 | MyFundedFutures Flex/Pro/Builder | **MEASURED: strong second sleeve** | Official automation allowance; copy trading across MFFU accounts; fast payouts; 25K/50K can support up to 5 sim-funded accounts, 100K/150K cap at 3. | Plan-specific caps; Flex 50K currently capped at 3 for new users; avoid Rapid intraday trailing; T1 news language conflicts across official pages. |
| 3 | Topstep XFA | **MEASURED: keep, capped** | Current integration works through ProjectX/TopstepX; 5 active XFAs; EOD MLL; optional DLL; no universal XFA consistency if Standard path. | LFA path prohibits ProjectX API automation; XFA closes on LFA call-up; payout resets MLL to $0. |
| 4 | Bulenox EOD Master | **MEASURED: candidate, downgraded** | EOD option, $2.5K DD on 50K, 11 active Master ceiling, 100% first $10K then 90/10. | 40% payout consistency; official automation wording is abuse-focused not a clean full-bot approval; Funded Account consolidation can collapse Masters into one account. |
| 5 | Alpha Futures | **INFERRED / PENDING** | Official page suggests HFT/100+ trades/day is the automation concern; 90% split, up to 750K simulated funding, no news restriction on most account types. | No repo integration/profile; fully hands-off bot not explicitly approved in official text. |
| DQ | TakeProfitTrader | **MEASURED: disqualified** | None for this bot. | Official PRO rules ban bots/algos and require manual execution. |
| DQ | Apex Trader Funding | **MEASURED: disqualified** | Scaling looks huge on paper. | Official PA compliance bans automation on all account types. |
| DQ | Earn2Trade | **MEASURED: disqualified for copy-router** | Possible manual single-account path only. | Official help says trade copiers are not allowed on any program. |
| DQ | Phidias | **MEASURED: disqualified for full automation** | Copy trading and static/EOD drawdown look attractive. | Official rules ban robots/fully automated algorithms; only semi-automated monitored trading is permitted. |

## 4. Disconfirming Findings

### TPT is not a bot firm

**MEASURED:** TakeProfitTrader's official PRO rules say automated or bot trading is not allowed and trades must be manually executed.  
**Conclusion:** Remove TPT from the automation ranking. It may be fine for a discretionary/manual trader, but it is not a canompx3 deployment target.

### Apex is not an automation workaround

**MEASURED:** Apex's PA compliance page states automation is strictly prohibited on all account types, including AI, algorithms, fully automated systems, and HFT.  
**Conclusion:** Apex remains a no-go. The 20-account scale is irrelevant if the bot itself is prohibited.

### Earn2Trade is incompatible with the copy-router

**MEASURED:** Earn2Trade says trade copiers are not allowed on any of its programs.  
**Conclusion:** Even if automation could be made to work, the current `CopyOrderRouter` premise fails.

### Phidias is a tempting false positive

**MEASURED:** Phidias official rules allow copy trading within account limits, with up to 5 accounts per product family, static drawdown for Express-to-Live, and EOD drawdown for Fundamental/Premium.  
**MEASURED:** The same rules ban robots, fully automated trading algorithms, and any other form of automated trading; only monitored semi-automation is permitted.  
**Conclusion:** Do not put canompx3 on Phidias unless we build a real human-in-loop execution mode.

### Bulenox old scaling claim is overstated

**MEASURED:** Bulenox Master Account rules allow up to 11 active Master Accounts, but only 3 can be activated simultaneously until milestones unlock more.  
**MEASURED:** Bulenox Funded Account rules say after three successful Master payouts, qualified traders may transition to a Funded Account and all active Masters are consolidated into one Funded Account.  
**Conclusion:** Bulenox is still a candidate, but it is not a clean "11 accounts forever" scaling engine.

### MFFU needs a news-rule preflight

**MEASURED:** MFFU's evaluation overview says T1 news trading is allowed.  
**MEASURED:** MFFU Rapid 50K says no T1 news trading in sim-funded.  
**MEASURED:** MFFU Fair Play says Tier 1 economic-data trading is restricted.  
**Conclusion:** MFFU remains attractive for automation/copy/payout trust, but do not route US_DATA_830 / US_DATA_1000 lanes until the exact plan's funded-stage news rule is confirmed in writing.

### Tradeify is strong but not cross-firm neutral

**MEASURED:** Tradeify allows owned bots/algorithms if the trader can prove sole ownership, no one else uses them, they are not HFT, and the bot is used solely within Tradeify. Tradeify also says it scans for similar orders on other accounts and may require a live video of enabling the code.  
**MEASURED:** Tradeify permits group trading only between accounts the trader owns and manages.  
**Conclusion:** Tradeify is the best firm-specific sleeve. It is not safe to assume the exact same bot can be run simultaneously at other firms without policy risk.

## 4A. Second-Pass Trap Audit - 2026-06-11

### Tradeify "bot solely within Tradeify" scope

**MEASURED:** Tradeify's official wording is broader than account-copying. It says the bot/algorithm/strategy must be solely owned by the trader and used solely within Tradeify; using it across multiple firms is against policy. It also says Tradeify scans for similar orders on other accounts and may request live video proof of the code being enabled.

**INFERRED:** Treat the rule as strategy/order-decision exclusivity, not just account-ID exclusivity. A separate canompx3 process does not solve this if it runs the same lane logic, same instruments, same timestamps, and same order fingerprints at Topstep/MFFU/Bulenox. A separate repo copy also does not solve it if it is substantively the same bot.

**INFERRED:** It is likely acceptable to keep canompx3 as the parent platform if the Tradeify sleeve is genuinely firm-specific: separate runtime config, separate credentials, separate strategy IDs, separate lane universe, no cross-firm copying, and no simultaneous identical order stream at another firm. Written Tradeify compliance confirmation is still required before paying for a 150K account.

**Practical rule:** Tradeify can be the top target only if we create a Tradeify-only strategy sleeve. Do not run the current `topstep_50k_mnq_auto` lane book unchanged on Tradeify while it is also trading Topstep.

### Topstep LFA ProjectX API gap

**MEASURED:** The repo does understand part of the Topstep trap. `trading_app/prop_profiles.py` states XFA and LFA are mutually exclusive, that LFA promotion destroys XFAs, and the active `topstep_50k_mnq_auto` profile is `is_express_funded=True`.

**MEASURED:** The current Topstep firm spec records `platform="topstepx"`, `auto_trading="full"`, and `firm_specific_rules={"max_live_accounts": 1}`. It does not encode the official LFA rule as a typed gate: "ProjectX API automation prohibited on Live Funded Accounts." The current guardrails also do not appear to encode the official TopstepX API personal-device / no-VPS rule as a machine-enforced deployment check.

**MEASURED:** This does not invalidate the current XFA profile by itself. The active profile is XFA, not LFA.

**INFERRED:** The dangerous failure mode is promotion/state drift: if a Topstep profile becomes LFA or a new LFA profile is added while keeping `BROKER=projectx` / TopstepX API automation, the repo should fail closed before any order is routed. Today this is known in docs/comments, but not strongly enough as a live preflight invariant.

**Required engineering follow-up:** add a fail-closed rule: `firm=topstep && is_express_funded=false && platform in {topstepx, projectx} && automation_enabled => BLOCK`. Add a separate deployment-host assertion for TopstepX API: no VPS, no VPN, no remote server, personal device only.

### Bulenox consolidation damage

**MEASURED:** Bulenox Master rules support up to 11 active Master Accounts, staged after initial milestones. Official Funded Account rules say that after three successful Master payouts, Bulenox may transition the trader to a Funded Account, and all active Master Accounts are consolidated into one Funded Account at Risk Management discretion.

**MEASURED:** Bulenox listed first-three Master payout caps are size-dependent; the 50K Master cap is $1,500 per payout and the 250K cap is $2,500 per payout. The Funded Account balance cap is also size-dependent: 50K -> $5,000, 100K -> $10,000, 150K -> $15,000, 250K -> $25,000.

**INFERRED:** This breaks the old "11 accounts forever" thesis by about 10-to-1 on durable account breadth. Eleven active Masters can become one Funded Account, so the durable drawdown-bucket count compresses from 11 to 1: a 90.9% reduction in account-count breadth.

**INFERRED:** It does not fully kill Bulenox. It changes the model from durable account fan-out to a Master-stage payout extraction sleeve with a consolidation cliff. For example, if 11 x 50K Masters all reached three capped payouts before consolidation, gross capped first-three-payout capacity is `11 * 3 * $1,500 = $49,500` before split, fees, activation, breach risk, and 40% consistency. After consolidation, the durable state is one 50K Funded Account capped at $5,000 balance. For 11 x 250K Masters, the equivalent capped first-three-payout gross is `11 * 3 * $2,500 = $82,500`, then one 250K Funded Account capped at $25,000 balance.

**Practical rule:** Bulenox remains useful only if the EV is modeled as `Master-stage payouts - fees - breach risk - consistency drag - consolidation cliff`. It is not a primary long-term copy-router scaling solution.

### Multi-firm one-live-account thesis

**MEASURED:** Consolidation is common but not universal. Bulenox can consolidate active Masters into one Funded Account. MFFU says multiple Sim Funded accounts are merged into one Live Funded Account, with multiple live accounts reserved for exceptional risk-manager discretion. Tradeify's current Elite Live material is different: each funded account that has received at least one payout can become its own Elite Live account, subject to Tradeify's account/program restrictions.

**INFERRED:** The viable version of the thesis is not "max all accounts everywhere." It is "earn one durable account per compliant firm sleeve, then diversify across firms." That can create real payout optionality because each firm is a separate counterparty, rulebook, payout desk, and failure domain.

**INFERRED:** Expected return can be positive only if each sleeve passes its own rule-aware C11 and has positive post-fee expectancy:

`EV_firm = P(pass_eval) * P(reach_payout_or_live) * expected_net_payouts - evaluation_fees - activation_fees - reset_fees - breach_cost - compliance_shutdown_risk`

Portfolio EV is the sum of firm EVs, but portfolio risk is not diversified as much as the account count implies if the same ORB book trades correlated MNQ/MES/MGC lanes at the same times.

**Practical rule:** Use the multi-firm ladder as a validation and payout-extraction bridge before own-capital scaling. Do not treat it as proof the strategy is ready for self-funded size until execution quality, slippage, live telemetry, and forward-OOS behavior are measured outside prop-firm constraints.

## 5. Account Tier Logic

The user's instinct is right: sometimes a more expensive account is better if the drawdown buffer and payout capacity scale faster than the fee.

### Tradeify Select

**MEASURED official Select table:**

| Size | Price seen | Max DD | Contracts | Eval consistency | Funded consistency |
|---|---:|---:|---:|---:|---:|
| 50K | $111 | $2,000 EOD | 4 mini / 40 micro | 40% | none |
| 100K | $181 | $3,000 EOD | 8 mini / 80 micro | 40% | none |
| 150K | $251 | $4,500 EOD | 12 mini / 120 micro | 40% | none |

**INFERRED:** 150K Select is likely the best Tradeify account for this bot if the pass cost is acceptable, because it gives 2.25x the 50K drawdown for 2.26x the listed price and materially higher payout caps. It does not add much if we stay permanently at one micro per lane, but it gives the C11 simulator more room for ORB tail days.

### Topstep XFA

**MEASURED official XFA rules:**

| Size | Max Loss Limit | Max accounts | Automation |
|---|---:|---:|---|
| 50K | $2,000 | Up to 5 XFAs | ProjectX/TopstepX works for XFA |
| 100K | $3,000 | Up to 5 XFAs | Same |
| 150K | $4,500 | Up to 5 XFAs | Same |

**MEASURED:** LFA permits automated strategies generally, but ProjectX API automation is prohibited on LFA.  
**INFERRED:** Topstep 150K XFA can be worth testing, but only as an XFA sleeve. It does not solve the LFA automation ceiling.

### MyFundedFutures

**MEASURED official cap table:**

- Up to 10 active evaluations.
- Up to 5 active sim-funded accounts if holding only 25K/50K account sizes.
- New 50K Flex accounts are capped at 3; pre-March 24 grandfathered accounts may hold 5.
- If any 100K or 150K sim-funded account is held, total sim-funded accounts cap at 3.

**MEASURED official payout/account plan snippets:**

- Flex: 80/20, 5 winning days, 25K/50K only, MLL resets to $100 after first payout.
- Pro: 80/20, up to $100K sim-stage payout cap per user, reviewed for live after 3 consecutive payouts or $20K profit milestone.
- Rapid 50K sim-funded: $2K intraday trailing high-water-mark drawdown, no funded consistency, 90/10.

**INFERRED:** For canompx3, avoid Rapid despite fast payout optics. Prefer EOD/static-style plans already represented in repo (`mffu_builder`, `mffu_flex`) and run plan-specific C11 before buying. Because official MFFU pages conflict on Tier 1 news treatment, any MFFU sleeve must either exclude US_DATA lanes or obtain written support approval first.

### Bulenox

**MEASURED official Master Account values:**

| Size | EOD/trailing DD | Activation fee | Active account ceiling |
|---|---:|---:|---:|
| 50K | $2,500 | $148 | Up to 11 Masters, staged activation |
| 100K | $3,000 | $248 | Same |
| 150K | $4,500 | $498 | Same |
| 250K | $5,500 | $898 | Same |

**INFERRED:** 50K EOD Master is the best first Bulenox test because it gives a better DD/fee ratio than 100K and avoids overpaying for notional we may not use. 150K/250K only make sense after Rithmic conformance and C11 prove the bot can use the extra buffer without tripping the 40% consistency rule.

## 6. Practical Deployment Sequence

1. **Do not abandon Topstep.** Keep the existing `topstep_50k_mnq_auto` XFA sleeve because it is the current proven integration.
2. **Tradeify first.** Build/refresh a Tradeify-only bot variant or get written compliance approval for cross-firm use. Then C11-test 100K and 150K Select Flex. If both pass, buy 150K first.
3. **MFFU second.** Select EOD/static-style plans only. Do not route Rapid until the strategy is modified to flatten/bank open profit aggressively enough for intraday trailing. Confirm the funded-stage T1 news rule before routing US_DATA lanes.
4. **Bulenox third.** Finish Rithmic conformance, then test 50K EOD Master. Treat 11-account scale as pre-funded/Master-stage opportunity, not guaranteed durable post-funded scale.
5. **Alpha watchlist.** Ask compliance one narrow question before any build: "Can I run my own fully automated ORB futures bot with bracket orders, under 10 trades/day, unattended after session start, across accounts I own?"
6. **Exclude TPT/Apex/Earn2Trade/Phidias for the current bot.**

## 7. Router Changes Before Scaling

Before increasing account fan-out:

1. Add per-firm strategy IDs / variants so Tradeify's exclusive-use clause is not violated by identical cross-firm order fingerprints.
2. Add per-account entry jitter and quantity variance only if it is explicitly allowed by each firm and does not create C11 drift.
3. Add firm-rule profiles for `tradeify_select_150k`, `mffu_*` chosen plan, and `bulenox_50k_eod`.
4. Re-run C11 with the exact firm rule set: DD type, DLL, consistency, payout reset/floor, max contracts, copy count, and funded transition.
5. Treat all copies as correlated exposure, not diversified bets.

## 7A. Firm-Specific Quant Decision Model

Every prop firm must be modeled as its own state machine. Do not rank firms by headline notional, advertised account count, or payout split alone.

### Required state machine per firm

1. **Evaluation state:** fee, reset cost, target, drawdown type, max contracts, minimum days, consistency, news restrictions, automation restrictions.
2. **Sim/funded state:** activation fee, payout cadence, payout caps, payout split, consistency, account-count cap, trade copier rules, inactivity rules.
3. **Live-transition state:** forced or optional, can it be refused, are sim accounts dormant/closed/merged, is live account single or multiple, does automation still work.
4. **Failure/recovery state:** reset available or not, account closed or back-to-funded, whether failing one account contaminates other accounts or firm access.
5. **Compliance state:** bot ownership, API/VPS restrictions, same-strategy-across-firms restrictions, HFT/frequency definitions, manual-monitoring requirements.

### EV equation

For each firm sleeve:

`EV_firm = P_eval_pass * P_funded_survive * P_payout_eligible * expected_net_payouts - fees - resets - activation_costs - platform_costs - compliance_shutdown_risk - opportunity_cost`

Where:

- `P_eval_pass` must be estimated from the exact lane book under the firm's evaluation target and rule set.
- `P_funded_survive` is Criterion 11: 90-day survival >= 70% using that firm's daily loss, trailing DD, consistency, max contracts, copy count, and transition mechanics.
- `expected_net_payouts` must include payout caps, splits, withdrawal rules, and whether payouts reset the max loss limit.
- `compliance_shutdown_risk` is not optional. If a firm bans same-strategy cross-firm bots, fully automated bots, VPS/API usage, or copy routing, the sleeve is zero unless the deployment avoids that rule.

### Portfolio equation

`EV_portfolio = sum(EV_firm_i)` only after each sleeve is individually legal and C11-safe.

Risk aggregation is not the simple sum of account drawdowns. Treat correlated ORB losses as shared shock risk:

`portfolio_tail_loss ~= sum(firm_exposure_i * shared_ORB_shock_i) + idiosyncratic_firm_rule_losses`

Carver portfolio logic applies: identical or near-identical order streams are not diversification. Different firms diversify counterparty/rule/payout risk, not strategy return risk.

### Ranking axes

| Axis | Why it matters | Bad proxy |
|---|---|---|
| Legal automation fit | Determines whether the bot can run at all | "Other traders use bots there" |
| Drawdown geometry | Determines survival under ORB tail days | Headline account size |
| Payout friction | Determines cash extraction before consolidation/live transition | Advertised split |
| Transition mechanics | Determines whether scaling collapses after success | Sim-funded account count |
| Copy/bot exclusivity | Determines cross-firm viability | Separate process name |
| Platform integration | Determines execution reliability and fill evidence | Website says Tradovate/Rithmic |
| Correlation load | Determines portfolio tail risk | Number of accounts |
| Operator burden | Determines mistake/compliance risk | "Set and forget" marketing |

### Practical interpretation

**MEASURED:** Prop firms are not interchangeable broker accounts. They are rule-constrained payout games with different failure modes.

**INFERRED:** The best path is a small number of high-quality, rule-clean sleeves across firms, not maximum account count at one firm. One durable account at each of Topstep XFA, Tradeify, MFFU, and maybe Bulenox can be useful if each one is legal, C11-safe, and fee-positive.

**UNSUPPORTED:** A multi-firm ladder does not prove the bot is ready for own-capital sizing by itself. It validates operational discipline, compliance survival, fill behavior, and payout extraction under prop constraints. Own-capital deployment still needs separate sizing, slippage, drawdown, and broker-risk modeling.

## 8. Bottom Line

**MEASURED:** Topstep is capped by 5 XFAs and the LFA ProjectX API prohibition.  
**MEASURED:** Tradeify is the strongest immediate non-Topstep target, but only if we respect its exclusive-use bot policy.  
**MEASURED:** MFFU is the strongest trust/payout diversification sleeve, but plan selection matters, Rapid is a bad ORB fit, and US_DATA lanes need written news-rule confirmation.  
**MEASURED:** Bulenox is useful but no longer deserves the old unqualified #1 scaling claim because of Funded Account consolidation and 40% payout consistency.  
**UNSUPPORTED:** Any claim that Apex/TPT/Phidias are usable for this fully automated bot is contradicted by their own current rules.

**Recommendation:** Keep Topstep XFA live, then prioritize Tradeify Select 150K as the first "pay more for a better account" experiment, subject to C11 survival and written/compliance-safe bot exclusivity. Then add MFFU EOD/static-style accounts. Treat Bulenox as Rithmic-stage optional growth, not the primary escape hatch.

## Sources

### Official / Primary

- [Topstep Express Funded Account Parameters](https://help.topstep.com/en/articles/8284215-express-funded-account-parameters) - XFA cap, payout paths, DLL optional.
- [Topstep Live Funded Account Parameters](https://help.topstep.com/en/articles/10657969-live-funded-account-parameters) - ProjectX API automation prohibited on LFA.
- [TopstepX API Access](https://help.topstep.com/en/articles/11187768-topstepx-api-access) - API access constraints, including personal-device / no-VPS operation.
- [Topstep Express Funded Account Rules](https://www.topstep.com/express-funded-account-rules) - MLL by account size and up to 5 XFAs.
- [Topstep Payout Policy](https://help.topstep.com/en/articles/8284233-topstep-payout-policy) - XFA payout resets MLL to zero.
- [Tradeify Guidelines for Traders](https://help.tradeify.co/en/articles/10468318-guidelines-for-traders) - bot ownership, exclusive use, HFT restriction.
- [Tradeify Tradovate Order Entry and Copy Trading](https://help.tradeify.co/en/articles/13196343-tradovate-order-entry-copy-trading) - own-account group trading.
- [Tradeify Select Evaluation Accounts](https://help.tradeify.co/en/articles/12853921-select-evaluation-accounts) - Select account sizes, DD, contracts, funded cap.
- [Tradeify Trailing Max Drawdowns](https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns) - all accounts use EOD trailing drawdown; hard breach semantics.
- [MyFundedFutures Fair Play and Prohibited Trading Practices](https://help.myfundedfutures.com/en/articles/8444599-fair-play-and-prohibited-trading-practices) - automation permitted if not exploitative/HFT.
- [MyFundedFutures Copy Trading](https://help.myfundedfutures.com/en/articles/10771500-copy-trading-at-myfundedfutures) - copy trading across MFFU accounts.
- [MyFundedFutures Traders Evaluation Simplified](https://help.myfundedfutures.com/en/articles/11802636-traders-evaluation-simplified) - account costs and account caps.
- [MyFundedFutures Payout Policy Overview](https://help.myfundedfutures.com/en/articles/13745661-payout-policy-overview-best-and-fastest-prop-firm-payouts) - plan payout mechanics.
- [MyFundedFutures Rapid 50K](https://help.myfundedfutures.com/en/articles/13134709-rapid-plan-50k-a-comprehensive-look) - Rapid intraday trailing drawdown.
- [Bulenox Master Account](https://bulenox.com/help/master-account/) - 11 Master cap, fees, DD, consistency, payouts.
- [Bulenox Qualification Account](https://bulenox.com/help/qualification-account/) - EOD scaling and DLL.
- [Bulenox Funded Account](https://bulenox.com/help/funded-account/) - post-Master funded transition and consolidation.
- [Bulenox Signup Terms](https://bulenox.com/member/signup/nH8dxpCHF) - algorithm/auto-trading abuse restrictions.
- [TakeProfitTrader PRO Account Rules](https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15171769361053-PRO-Account-Rules) - bot/algo ban.
- [Apex Prohibited Activities](https://apextraderfunding.com/help-center/getting-started/prohibited-activities/) and [Apex PA Compliance](https://apextraderfunding.com/help-center/performance-accounts-pa/legacy-performance-account-pa-compliance/) - automation prohibition.
- [Earn2Trade Copy Trading Policy](https://help.earn2trade.com/en/articles/12034590-am-i-allowed-to-copy-trades-across-multiple-accounts) - trade copier ban.
- [Phidias Rules](https://phidiaspropfirm.com/rules) - copy trading, EOD/static drawdown, automation ban, consistency.
- [Alpha Futures Prohibited Trading Practices](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices) - HFT/100+ trade-day automation restriction.
- [Alpha Futures Program Page](https://alpha-futures.com/) - funding, payout and rule overview.

### Local Methodology / Repo Authority

- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
- `docs/institutional/literature/carver_2015_ch11_portfolios.md`
- `docs/institutional/literature/lopez_de_prado_2018_afml_ch_3_7_8.md`
- `docs/research-input/topstep/topstepx_api_access_2026-05-24.md`
- `docs/audit/2026-04-08-topstep-canonical-audit.md`
- `docs/audit/2026-04-15-topstep-scaling-reality-audit.md`
- `trading_app/prop_profiles.py`

### Unofficial / Cautionary Only

- CrossTrade, PickMyTrade, TradersPost, PropFirmMatch, DamnPropFirms, QuantVPS, Reddit, YouTube and affiliate reviews were used only as discovery leads or cautionary signals. They do not override official rules.
