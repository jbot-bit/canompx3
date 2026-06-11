# Prop-Firm EV Scorecard V1

Prepared: 2026-06-11

Scope: research-only scorecard for Topstep XFA, Tradeify Select 150K, MFFU Flex 50K, and Bulenox 50K EOD. This document does not authorize live deployment, broker routing, account purchase, allocation changes, profile edits, or database writes.

Companion source manifest: [prop_firm_ev_scorecard_2026.sources.yaml](./prop_firm_ev_scorecard_2026.sources.yaml)

## Evidence Labels

| Label | Meaning |
| --- | --- |
| `MEASURED` | Directly supported by official firm documentation or current local repo source. |
| `MEASURED-CONFLICT` | Two or more official firm surfaces conflict; do not average or choose without purchase-dashboard or written support confirmation. |
| `INFERRED` | Logical consequence of official rules, local state-machine modelling, or canompx3 risk policy. |
| `UNSUPPORTED` | Not present in official sources reviewed for this v1. |

## Disconfirming Findings First

1. `MEASURED-CONFLICT`: Tradeify Select 150K price is not clean. The official help pricing reference says Select 150K is `$369` with a `$239` reset, while the public Select plan page currently displays `$251` and a `$239` reset. This blocks numeric EV until checkout-dashboard truth or written support confirmation.
2. `MEASURED`: Topstep XFA is useful, but the repo's flat `$5,000` Standard XFA payout-cap assumption is stale for non-150K accounts. Official payout caps are tier-specific: 50K Standard `$2,000`, 100K Standard `$3,000`, 150K Standard `$5,000`.
3. `MEASURED`: Topstep LFA is not a straight bot-growth path for this ProjectX/API setup. Official LFA docs say all XFAs close on call-up and ProjectX API automation is prohibited in the LFA.
4. `MEASURED`: MFFU Flex 50K allows automation and copy trading across MFFU accounts, but it is not automatically best. The official current Flex 50K terms are 80/20, `$2,000` max sim payout request, five max sim payouts, live transition/merge risk, live/sim mutual exclusion, and Tier 1 news restrictions under Fair Play.
5. `MEASURED`: Bulenox supports up to 11 Master accounts, but the official Funded Account transition can consolidate all active Masters into one Funded account after three successful Master payouts. That materially breaks the old "11 accounts forever" scaling thesis.
6. `INFERRED`: Running one account per firm can validate operational plumbing and diversify firm-specific rule risk, but it does not make ORB outcomes additive. Same-session ORB losses remain correlated across firms.
7. `INFERRED`: No row below means "deploy live bot." A buy verdict means "candidate for C11, broker routing, live preflight, and compliance verification."

## Method

Directional EV only:

```text
Directional EV = P(survive eval) * P(reach payout state) * capped payout capacity * allowed copy/account count
               - fees
               - transition haircut
               - compliance-failure haircut
               - correlated-tail haircut
```

No numeric EV is assigned unless official fee, payout, transition, and compliance fields are all present and non-conflicting. C11 survival inputs are listed, but no C11 pass/fail claim is made until the simulator runs.

Local research framing:

- `docs/institutional/pre_registered_criteria.md`: C11 requires survival and prop-firm guardrails before promotion claims.
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`: position sizing and risk allocation matter more than nominal account size.
- `docs/institutional/literature/carver_2015_ch11_portfolios.md`: cross-market or cross-account diversification must be treated as correlation-aware, not simple addition.
- `RESEARCH_RULES.md`: no "return", "edge", "validated", or "profitable" claim without local evidence and preregistered verification.

## Ranked Buy/No-Buy Table

| Rank | Firm / Plan | Verdict | EV Band | State Machine | Fees | Payout / Caps | Automation / Copy | Transition Risk | C11 / Tail Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Tradeify Select 150K | `BUY-LATER` | `positive candidate with compliance gate` (`INFERRED`) | Select eval -> Select Flex funded -> payouts -> Elite/live path (`MEASURED`) | Price conflict: `$369` help page vs `$251` public plan page; reset `$239` (`MEASURED-CONFLICT`) | 90/10; 150K Select Flex payout cap `$5,000`; funded consistency removed (`MEASURED`) | Bots allowed, but automated systems must be solely owned and used exclusively within Tradeify; copy only between owned/managed accounts (`MEASURED`) | Exclusivity may block same bot or same strategy running at other firms (`INFERRED`) | Best candidate only after written exclusivity clarification and C11 simulation (`INFERRED`) |
| 2 | Topstep XFA, canompx3 active 50K sleeve | `WATCH` | `fee/cap constrained` (`INFERRED`) | Trading Combine -> XFA -> payouts -> LFA call-up risk (`MEASURED`) | Activation fee applies; no monthly sub after pass; API fee `$29/mo` or `$14.50/mo` with Topstep discount (`MEASURED`) | 90/10; 50K Standard cap `$2,000`; 100K `$3,000`; 150K `$5,000`; min `$125` (`MEASURED`) | XFA automation allowed; copier allowed up to `$750K`; LFA ProjectX API prohibited (`MEASURED`) | LFA call-up closes all XFAs and cannot be declined while staying XFA (`MEASURED`) | Keep as current known integration, but do not model LFA as bot scale-up (`INFERRED`) |
| 3 | MFFU Flex 50K | `WATCH` | `fee/cap constrained` (`INFERRED`) | Eval -> sim-funded -> payout cycles -> live transition/merge -> cooldown (`MEASURED`) | Flex 50K eval `$153`; activation `$0`; live costs may include CME professional data, commissions, platform fees (`MEASURED`) | 80/20; max sim request 50% profits capped `$2,000`; five max sim payouts; live min request `$250` (`MEASURED`) | Automation allowed if not exploiting sim fills; copy trading across MFFU accounts allowed; copying other traders/collaboration banned (`MEASURED`) | Live transition after five consecutive approved payouts, `$100K` sim cap, or risk review; live and sim cannot trade simultaneously (`MEASURED`) | Viable later if Tier 1 news blocks and live-merge haircut pass C11 (`INFERRED`) |
| 4 | Bulenox 50K EOD | `NO-BUY` | `fee/cap constrained` (`INFERRED`) | Qualification -> Master -> three-payout window -> funded consolidation/refusal closeout (`MEASURED`) | Master activation `$148`; reset `$78`; professional data `$116/mo`; non-pro data included (`MEASURED`) | First `$10K` no commission then 90/10; first three 50K payouts capped `$1,500`; min `$1,000`; 40% consistency every payout (`MEASURED`) | Bots/copying not forbidden; trader carries responsibility; one Rithmic user ID (`MEASURED`) | After three successful payouts, all active Masters can consolidate into one Funded account; refusal closes Master and no reward payout (`MEASURED`) | Do not buy before Rithmic conformance, C11, and consolidation haircut are explicitly modelled (`INFERRED`) |

No `BUY-FIRST` row is assigned in v1. The best row is Tradeify `BUY-LATER`, because the purchase price and bot exclusivity gate are unresolved.

## Firm State Machines

### Topstep XFA

State machine (`MEASURED`):

```text
Trading Combine -> XFA -> payout cycles -> forced/optional LFA call-up risk
```

Official state constraints:

- `MEASURED`: Up to five active XFAs are allowed.
- `MEASURED`: All XFAs close when the trader is called up to LFA.
- `MEASURED`: LFA docs say a trader cannot decline call-up and keep trading XFA.
- `MEASURED`: XFA automated strategies are allowed subject to Topstep rules.
- `MEASURED`: LFA automated strategies are allowed, but ProjectX API is prohibited.
- `MEASURED`: TopstepX API access is personal-device only; VPS, VPN, and remote servers are prohibited.

EV interpretation:

- `INFERRED`: Treat Topstep as a known integration and validation sleeve, not an uncapped growth sleeve.
- `INFERRED`: For canompx3, XFA state and LFA state must be separate account states because the automation surface changes.
- `INFERRED`: The 50K sleeve cannot borrow the 150K `$5,000` Standard payout cap.

C11 inputs:

- Account size: 50K active sleeve.
- Payout cap: `$2,000` Standard XFA for 50K.
- Drawdown: repo marks Topstep as EOD trailing; official LFA daily loss limits differ by size.
- Automation gate: XFA ProjectX API path cannot be assumed legal in LFA.
- Hosting gate: personal-device-only API restriction.
- Simulator status: `UNSUPPORTED` for C11 pass/fail until simulator output exists.

### Tradeify Select 150K

State machine (`MEASURED`):

```text
Select evaluation -> Select Flex funded -> payout cycles -> Elite/live path
```

Official state constraints:

- `MEASURED`: 150K Select evaluation has `$9,000` target, `$4,500` EOD trailing drawdown, no daily loss limit, 40% evaluation consistency, and max 12 mini or 120 micro contracts.
- `MEASURED`: Select Flex 150K funded payout cap is `$5,000`.
- `MEASURED`: Funded consistency is removed on the reviewed Select page.
- `MEASURED`: Bots are allowed if solely owned and designed by the trader.
- `MEASURED`: Automated trading systems must be used exclusively within Tradeify.
- `MEASURED`: Tradovate group trading is allowed only between accounts owned or managed by the trader.
- `MEASURED-CONFLICT`: 150K Select price is `$369` on the help pricing reference but `$251` on the public Select plan page.

EV interpretation:

- `INFERRED`: Tradeify is the top candidate only if "bot solely within Tradeify" is solved in writing.
- `INFERRED`: The conservative compliance interpretation is whole-system exclusivity: the same bot or same automated system should not run on other firms while used at Tradeify.
- `UNSUPPORTED`: Official docs reviewed do not answer whether separate codebases with separate lanes are allowed if the strategy family is similar.

C11 inputs:

- Account size: `$150,000`.
- Profit target: `$9,000`.
- Max loss limit: `$4,500` EOD trailing.
- Max contracts: 12 mini or 120 micro.
- Payout cap: `$5,000`.
- Compliance gate: exclusivity clarification required.
- Simulator status: `UNSUPPORTED` for C11 pass/fail until simulator output exists.

### MFFU Flex 50K

State machine (`MEASURED`):

```text
evaluation -> sim-funded -> payout cycles -> live transition/merge -> cooldown
```

Official state constraints:

- `MEASURED`: Flex 50K evaluation has `$3,000` target, `$2,000` EOD max loss limit, no mandatory daily loss limit, max 3 mini or 30 micro contracts, 50% eval-only consistency, and two minimum trading days.
- `MEASURED`: Flex 50K sim-funded has 80/20 split, five winning days of at least `$150`, max sim request 50% of profits capped at `$2,000`, and five max sim payouts.
- `MEASURED`: Activation fee is `$0`.
- `MEASURED`: Automation is allowed if it does not exploit favorable sim fills or violate live exchange behavior.
- `MEASURED`: HFT, sim-fill exploitation, Tier 1 restricted-news trading, collaborative identical/opposite strategies, copying other traders, and hedging are prohibited.
- `MEASURED`: Copy trading across MFFU accounts is allowed.
- `MEASURED`: Live transition can occur after five consecutive approved payouts, the `$100K` sim cap, or risk review.
- `MEASURED`: Live and sim-funded accounts cannot be traded at the same time.

EV interpretation:

- `INFERRED`: MFFU is compliance-viable for automation, but caps and live transition make it less attractive than a clean Tradeify 150K path.
- `INFERRED`: Copying across MFFU accounts does not create independent payout capacity if all accounts run the same ORB lane.
- `INFERRED`: Builder is not the comparable non-Rapid target here; Flex 50K is the relevant official non-Rapid sleeve for this v1.

C11 inputs:

- Account size: `$50,000`.
- Profit target: `$3,000`.
- Max loss limit: `$2,000` EOD.
- Max contracts: 3 mini or 30 micro, with funded scaling.
- Payout cap: `$2,000` max sim request and five max sim payouts.
- News gate: exclude or block Tier 1 restricted windows.
- Simulator status: `UNSUPPORTED` for C11 pass/fail until simulator output exists.

### Bulenox 50K EOD

State machine (`MEASURED`):

```text
qualification -> Master -> three-payout window -> funded consolidation/refusal closeout
```

Official state constraints:

- `MEASURED`: 50K EOD Master activation is `$148`.
- `MEASURED`: Qualification reset is `$78` outside the billing-date reset.
- `MEASURED`: Professional data is `$116/mo`; non-professional data is included.
- `MEASURED`: Up to 11 active Masters are allowed, with initial staging of 3.
- `MEASURED`: First `$10,000` has no commission, then 90/10.
- `MEASURED`: Weekly Wednesday payouts, 10 trading days, minimum `$1,000`, 50K first-three-payout cap `$1,500`, 40% consistency for every payout.
- `MEASURED`: Option 2 EOD drawdown stops moving at initial balance plus `$100` after Qualification and Master.
- `MEASURED`: After three successful Master payouts, Risk Management may transition to Funded, consolidate all active Masters into one Funded account, and close a Master with no reward payout if the trader refuses.
- `MEASURED`: Trade copiers, algorithms, strategies, and bots are not forbidden on the reviewed Qualification page.

EV interpretation:

- `INFERRED`: Bulenox has clean Master numbers, but the consolidation/refusal state is too punitive to model as 11 durable payout lanes.
- `INFERRED`: A staged 3-Master setup is the maximum shape that should even be considered before modelling Funded consolidation.
- `INFERRED`: Current repo marks Bulenox inactive; keep it no-buy until Rithmic conformance and C11 survival are proven.

C11 inputs:

- Account size: `$50,000`.
- TDA: `$2,500` EOD trailing.
- Daily loss: `$1,100` until removal condition.
- Safety reserve: `$2,600`.
- First three payout caps: `$1,500` each.
- Consistency: 40% every payout.
- Simulator status: `UNSUPPORTED` for C11 pass/fail until simulator output exists.

## Repo Consistency Notes

Local source comparison against [trading_app/prop_profiles.py](../../trading_app/prop_profiles.py) and [trading_app/prop_firm_policies.py](../../trading_app/prop_firm_policies.py):

| Surface | Status | Finding |
| --- | --- | --- |
| `trading_app/prop_firm_policies.py` Topstep Express Standard cap | `STALE` | Local policy uses `$5,000` as a flat Standard XFA cap. Official Topstep payout policy is tiered: 50K `$2,000`, 100K `$3,000`, 150K `$5,000`. |
| `trading_app/prop_profiles.py` Topstep profile notes | `PARTIAL` | Notes correctly separate Express and Live risk, but any model that treats LFA as ProjectX/API growth is stale against official LFA API prohibition. |
| `trading_app/prop_profiles.py` Tradeify notes | `PARTIAL` | Notes capture bot exclusivity and cross-firm concern, but current price must be `MEASURED-CONFLICT`. |
| `trading_app/prop_profiles.py` MFFU Flex | `MOSTLY CURRENT` | Local fields align with official Flex 50K economics, including 80/20, `$2,000` sim cap, five max sim payouts, and live transition. Add explicit Fair Play news/collaboration gates before code-backed EV. |
| `trading_app/prop_profiles.py` Bulenox | `PARTIAL` | Local notes capture first-three payout caps and up to 11 Masters, but the scorecard must include the official funded consolidation/refusal closeout risk. |

No profile edits were made in this v1.

## Coverage And Future Tests

This v1 is documentation plus source manifest only. Future code-backed scorecard work should add parser/tests rather than hand-maintaining these fields.

Exact current/future test targets:

```powershell
python -m pytest tests/test_prop_profiles_mffu.py -q
python -m pytest tests/test_research/test_prop_firm_ev_scorecard.py -q
```

Required future assertions:

- Topstep 50K Standard XFA cap must not inherit 150K `$5,000` cap.
- Tradeify Select 150K price remains `MEASURED-CONFLICT` until dashboard checkout or written support resolves it.
- MFFU automation status is allowed with Fair Play restrictions, not banned.
- Bulenox 11-Master count is not treated as durable after the three-payout consolidation transition.
- No verdict string may imply deployability without C11, broker routing, live preflight, and compliance gates.

## Official Sources

Topstep:

- [Express Funded Account Parameters](https://help.topstep.com/en/articles/8284215-express-funded-account-parameters)
- [Live Funded Account Parameters](https://help.topstep.com/en/articles/10657969-live-funded-account-parameters)
- [TopstepX API Access](https://help.topstep.com/en/articles/11187768-topstepx-api-access)
- [Payout Policy](https://help.topstep.com/en/articles/8284233-topstep-payout-policy)

Tradeify:

- [Select Evaluation Accounts](https://help.tradeify.co/en/articles/12853921-select-evaluation-accounts)
- [Guidelines for Traders](https://help.tradeify.co/en/articles/10468318-guidelines-for-traders)
- [Tradovate Order Entry and Copy Trading](https://help.tradeify.co/en/articles/13196343-tradovate-order-entry-copy-trading)
- [Rules: Trailing Max Drawdowns](https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns)
- [Tradeify Pricing Reference](https://help.tradeify.co/en/articles/14369021-tradeify-pricing-reference)
- [Select Flex and Select Daily Payout Policies](https://help.tradeify.co/en/articles/12853966-select-flex-and-select-daily-payout-policies)
- [Select Plan](https://tradeify.co/select-plan)

MFFU:

- [Flex Plan 50,000 Comprehensive Guide](https://help.myfundedfutures.com/en/articles/15072271-flex-plan-50-000-a-comprehensive-guide)
- [Traders Evaluation Simplified](https://help.myfundedfutures.com/en/articles/11802636-traders-evaluation-simplified)
- [Copy Trading at MyFundedFutures](https://help.myfundedfutures.com/en/articles/10771500-copy-trading-at-myfundedfutures)
- [Fair Play and Prohibited Trading Practices](https://help.myfundedfutures.com/en/articles/8444599-fair-play-and-prohibited-trading-practices)
- [Understanding Live Funded Account](https://help.myfundedfutures.com/en/articles/10101257-understanding-live-funded-account-at-myfunded-futures)

Bulenox:

- [Qualification Account](https://bulenox.com/help/qualification-account/)
- [Master Account](https://bulenox.com/help/master-account/)
- [Funded Account](https://bulenox.com/help/funded-account/)

## Next Research Move

Run a code-backed C11/account-state simulator that consumes `prop_firm_ev_scorecard_2026.sources.yaml`, current canompx3 ORB trade distributions, and per-firm payout/drawdown rules. The simulator should output survival probability, payout-state reach probability, rule-block reasons, and correlated ORB tail exposure per firm without changing live config.

Suggested prompt:

```text
Build a code-backed simulator for docs/research/prop_firm_ev_scorecard_2026.sources.yaml. Use current canompx3 ORB trade distributions, model each prop firm as a separate state machine, and output C11 survival probability, payout-state reach probability, compliance blockers, transition haircuts, and correlated ORB tail risk. Do not mutate live config, broker settings, allocation files, prop profiles, or DBs. Add tests under tests/test_research/test_prop_firm_ev_scorecard.py and verify with python -m pytest tests/test_prop_profiles_mffu.py -q plus the new test target.
```
