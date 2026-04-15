# TopStep Scaling Reality Audit — 2026-04-15

**Date:** 2026-04-15
**Author:** Claude (Sonnet 4.6 session ad996f49)
**Trigger:** User question — "can I actually run 5 XFA and 1 LFA, or does LFA remove all XFA accounts?"
**Status:** Canonical — supersedes prior informal assumptions. Affects deployment plan, prop_profiles scaling ceiling, and all revenue projections.

---

## 1. The Correction

**PRIOR ASSUMPTION (WRONG):** 5 XFAs + 1 LFA run concurrently. Aggregate annual ceiling ≈ $300K XFA payouts + LFA upside.

**CANONICAL RULE (CORRECT):** XFA **or** LFA — never both. LFA promotion is mandatory and destroys all XFAs.

**Sources (all local canonical scrape, `docs/research-input/topstep/`):**

- `topstep_live_funded_parameters.md:280` — *"You can only have one (1) Live Funded Account active. When you receive a Live Funded Account, all Express Funded Accounts are closed."*
- `topstep_live_funded_parameters.md:264` — *"it is not possible to stay in an Express Funded Account and decline to move to a Live Funded Account. Once our Risk Managers determine you're ready to move to a Live Funded Account, your options are to make the switch from Express to Live or close your Express Funded Account."*
- `topstep_live_funded_parameters.md:276` — *"if you lose your Live Funded Account, you would need to pass a Trading Combine and show consistency in an Express Funded Account before being called up to Live again."*
- `topstep_xfa_parameters.txt:228,360` — same consolidation rule restated.
- `topstep_payout_policy.txt:243` — *"When a trader is called up to the Live Funded Account, all open Express Funded Accounts will be closed. The starting balance of the Live Funded Account will be based on the combined total balance of your eligible Express Funded Accounts, up to your Live Account Size."*

**Implication:** The path is sequential, not additive:
1. Pre-LFA: up to 5 XFA concurrent
2. Forced promotion between 3rd and 5th payout (Topstep discretion, cannot decline)
3. Post-LFA: 1 account only, starting capital = 20% of combined XFA balance up to tier cap, min $10K
4. If LFA fails → "Shoulder Tap" back to **1** XFA (not 5), rebuild from there

---

## 2. LFA Survival — external data

**<1% of traders called up from XFA to LFA survive long-term.**
Source: [propfirmapp Topstep Review](https://propfirmapp.com/prop-firms/topstep) — "less than 1% of traders called up from Express Funded to Live Funded actually make it, partially because they're removing anyone they suspect of cheating."

This reframes LFA from "the real upside" to "a high-failure stage most traders do not clear." Tier size (50K vs 100K vs 150K) matters less than previously assumed because the survival probability dominates the expected value calculation.

---

## 2b. Revised multi-firm revenue math (post canonical scrape)

**Prior estimate:** ~$180K/yr gross across 36 accounts at $5K/account average.
**Corrected after Bulenox + Apex canonical research:**

| Firm | Concurrent | Net/account/yr (est.) | Annual net | Risk/caveat |
|---|---|---|---|---|
| TopStep | 5 XFA | $5,612 × 0.88 survive (B2F) = ~$4.9K | **~$25K** | Caps at LFA promotion (forced, destroys XFAs) |
| Bulenox | 3 Master | ~$7–8K (better split: 100% first $10K) | **~$21–24K** | 3-concurrent cap, not 11; 40% consistency on Master |
| Apex | 20 PA | IF bot allowed unsupervised: ~$6K; if monitored-only: ~$3K | **$60K–$120K** | **Bot policy ambiguous on funded PA** — material risk |

**Honest revised ceiling range: $106K–$169K/yr gross**, depending on Apex automation resolution.

**Apex bot-policy uncertainty is the single biggest variable in this model.** If fully autonomous bot deploy is actually prohibited on Apex PA (not just "must monitor"), the 20-account ceiling collapses to evaluation-only income (non-recurring). If monitored-bot deploy is acceptable (human logs in during session), the full 20 is realistic.

**Resolution requirement: direct verification with Apex support or login to the ZenDesk-blocked canonical docs before committing to a 20-PA Apex deployment plan.**

## 3. Multi-firm — the real scaling answer

**TopStep alone is capped.** Beyond 5 XFA pre-LFA or 1 LFA post-LFA, there is no further Topstep-only scale.

**Multi-firm accounts ARE legal and universally permitted** — no firm tracks what you do elsewhere, provided you own all accounts, no cross-firm hedging, and not HFT/signal-copying-third-parties.

Sources:
- [propfirmapp Multiple Prop Firm Accounts Policies](https://propfirmapp.com/learn/multiple-prop-firm-accounts)
- [Cointracts Trade Multiple Prop Firms](https://www.cointracts.com/learning-hub/can-you-trade-multiple-prop-firms-at-once-master-the-copy-trading-rules)
- [OnlyPropFirms Automated Futures Trading](https://onlypropfirms.com/articles/automated-futures-trading)

### Per-firm account ceilings (updated 2026-04-15 after canonical scrape)

| Firm | Concurrent funded cap | Bot policy | First-$ split tier | Profit split post-tier |
|---|---|---|---|---|
| **TopStep** | 5 XFA pre-LFA → 1 LFA (XFAs destroyed at promotion) | Allowed per canonical; TopstepX API explicit support; native platform may block EAs | First $10K: 90% (no separate tier) | 90/10 (post-2026-01-12 traders) |
| **Bulenox** | **3 Master concurrent** (up to 11 lifetime/eventual via unlock path — external sources vary) | **FULLY PERMITTED — most bot-friendly.** No HFT. | **First $10K: 100%** | 90/10 |
| **Apex** | **20 Performance Accounts simultaneously** | Partial: bots OK in evaluation, "fully automated" prohibited on funded PA (must monitor); DCA bots + copy-own-accounts explicitly permitted | **First $25K: 100%** | 90/10 |

**Practical aggregate ceiling across 3 firms (post canonical scrape):**
- **Automated-friendly concurrent: 5 TS + 3 Bulenox + 20 Apex(eval) = 28 accounts during eval phase**
- **Post-funded concurrent (realistic): 5 TS + 3 Bulenox + 20 Apex(with monitoring) = 28 accounts requiring some human presence**
- **Eventual ceiling after Bulenox unlocks to full cap: 5 TS + 11 Bulenox + 20 Apex = 36 accounts** (matches original external-source estimate, but gated by time and performance)

**Full canonical detail:**
- `docs/research-input/bulenox/README.md` — Bulenox rules, pricing, automation policy
- `docs/research-input/apex/README.md` — Apex rules, payout ladder, automation nuance

---

## 4. Simulation results — data-grounded sizing decision

All simulations run on 6 live-lane bot (`lane_allocation.json` 2026-04-13) over 1,792 historical trading days (2019-05-06 to 2026-04-10). Script inline in session; numbers reproducible from `orb_outcomes` + canonical `_load_strategy_outcomes` loader.

### Stage 1: combine pass rates (rolling 90-day windows, 353 starts each)

| Size | Contracts | Pass% | MLL-blow% | Median days to pass |
|---|---|---|---|---|
| 50K | 1 | 45.0% | **2.5%** | 47 |
| 50K | 2 | 60.9% | 16.4% | 34 |
| 50K | 3 | 51.8% | 28.3% | 25 |
| 100K | 1 | 22.9% | **0.0%** | 70 |
| 100K | 3 | 53.5% | 16.4% | 39 |
| 150K | 1 | 2.0% | **0.0%** | 85 |
| 150K | 5 | 55.5% | 16.7% | 39 |

**Key finding: 50K @ 1 contract has the best safety-at-speed profile** (2.5% blow rate, 47-day median pass).

### Stage 2: tail-day risk (consistency target exposure)

At 1 MNQ/lane over 7 years (1,792 days):
- Days ≥ $1,500 (50K cap): **1/1792 = 0.06%** — single outlier (Apr 9 2025 Liberation Day rebound, $1,949)
- Days ≥ $3,000 (100K cap): 0
- Days ≥ $4,500 (150K cap): 0

**Consistency target is a non-issue at 1ct regardless of size.** At 2ct: 0.78% at 50K. At 5ct: 8.76% (don't go there).

**Worst single loss day in 7yr: -$2,320 (Apr 7 2025, tariff crash).** Second-worst: -$761. ONE outlier day dominates entire MLL-blow risk at every size.

### Stage 3: XFA lifecycle (100 random 252-day starts, post-combine-pass)

| Size | Contracts | Survive% | Payouts/yr | Annual gross | Net (90/10) | Days → 1st payout |
|---|---|---|---|---|---|---|
| 50K / 100K / 150K | **1** | **51%** | **5.3** | $6,235 | **$5,612** | **88d** |
| 50K | 2 | 9% | 15.1 | $22,699 | $20,429 (91% blow) | 41d |
| 100K | 2 | 12% | 13.5 | $20,126 | $18,114 (88% blow) | 60d |
| 150K | 3 | 1% | 19.0 | $40,516 | $36,464 (99% blow) | 68d |

**At 1ct, XFA economics are identical across sizes.** Bot's tail risk (single -$2,320 day) caps safe contract count at 1. Higher contract counts yield higher gross but blow up almost always.

### With Back2Funded (2 reactivations per XFA)

Effective annual survive: **88%** → ~$22K realized net per XFA.

### 5-XFA parallel scenario (Topstep only)

| Size | Annual net (B2F) | Year-1 after combine cost |
|---|---|---|
| **50K** | **~$124K** | **$122K** |
| 100K | $124K | $122K |
| 150K | $124K | $122K |

Economics identical. Size doesn't matter for Topstep-only cash flow at 1ct.

---

## 5. Sizing decision — 50K confirmed

**Stick with 50K for TopStep** across the portfolio. Three independent reasons:

1. **Economics identical at 1ct across all sizes** (same bot, same daily PnL, same payout cap).
2. **You cannot safely run 2-3 contracts** at any size — 88–99% blow rates from the Apr 7 2025 outlier. Multi-contract use of 100K/150K buying power is structurally unsafe with current bot edge.
3. **Cheapest failure cost:** 50K combine ≈ $299 amortized vs ~$369 (100K) or ~$352 (150K). Matters during bot-proving phase.

**LFA tier size (would-be upside of 150K XFAs)** is swamped by the <1% LFA survival rate. The 3x tier size is worth ~$1K upfront optionality, not the thousands of extra combine cost for 5×150K vs 5×50K.

---

## 6. Prohibited-conduct landmines (do not trigger)

From `docs/research-input/topstep/topstep_prohibited_trading_strategies.md`:

- **Account stacking.** "Repeatedly trades aggressively, hits MLL in one account, switches to another to repeat, until a successful outcome yields a significant profit." Our bot pattern (same signals, same risk per account, no MLL-chasing) is NOT this. Do NOT adopt "extract and recycle" phrasing in deployment — that is exactly what the rule bans.
- **Cross-account hedging.** Opposite positions in same/correlated instruments across XFAs, LFAs, Combines. Same-direction copies are fine. Keep bot direction-consistent across all accounts.
- **High-frequency trading** (excessive orders/cancellations). Our 30-trades/day ORB bot is orders of magnitude below HFT. No risk at current edge.
- **Trading similarly in partnership with other individuals** (same trades in same time increments). Rule targets coordination between **people**, not one user's own-account copies. Self-owned multi-account copies are explicitly permitted via built-in copier.
- **Using AI/ultra-high-speed/mass data entry to manipulate or get unfair advantage.** Algorithmic bot = fine. HFT arbitrage = not fine. Our bot is algorithmic, not manipulative.

---

## 7. TopstepX bot nuance

**Canonical docs (Topstep help center):** "Yes, automated strategies are permitted" in both Combine and XFA. With disclaimers: no exception handling for bot malfunctions, trader fully responsible.

**External sources (QuantVPS, OnlyPropFirms):** "TopstepX native platform does not support EAs / bots — all trades must be manual." BUT "TopstepX API allows developers and serious traders to build, test, and run their custom strategies."

**Interpretation:** Our bot must connect via TopstepX API, not the native GUI's built-in execution. Current bot deployment (`topstep_50k_mnq_auto` on practice account 20092334) is likely already API-based. **Verify before live.**

---

## 8. Correlation risk at scale

**The catastrophic scenario:** 36 accounts running identical signals, all experience -$2,320 on an Apr 7 2025-style outlier day = **-$83,520 in one day**. Any account whose MLL is breached is permanently closed (Back2Funded covers 2 reactivations on Topstep; unclear at other firms).

**Mitigation patterns from research:**
- **Rotation trading** — designate which accounts trade which sessions/days, staggered timing, reduce simultaneous exposure
- **Per-firm account lane allocation** — don't copy all signals to all accounts; split so no firm has ALL the lanes
- **Size tapering** — run 1ct on newest accounts, build buffer before adding more lanes

**This is a real risk and must be designed into multi-firm deployment.** Not modelled in Stage 1–3 sims (which assumed independent accounts).

---

## 9. Open research gaps

1. ✅ **Bulenox canonical rule scrape — COMPLETED 2026-04-15.** See `docs/research-input/bulenox/README.md`. Key discoveries: **3 Master Accounts concurrent (not 11)**, 100% profit on first $10K, fully bot-permissive, $195/mo subscription at 50K, weekly Wednesday payouts, 40% consistency rule on Master phase, 10 trading days min before first payout.
2. ✅ **Apex canonical rule scrape — PARTIAL 2026-04-15.** See `docs/research-input/apex/README.md`. Canonical pages blocked by HTTP 403 (ZenDesk); relied on cross-referenced third-party sources. Key: **20 PA concurrent**, 6% profit target flat, 6-step payout ladder with cap removed after step 6, 100% first $25K, static DD post-funded (starting+$100). **BOT POLICY AMBIGUOUS:** "fully automated prohibited on PA" vs "copy-own-accounts permitted" — requires direct verification before funded deploy.
3. **Apex prohibited-activities canonical verification** — 403-blocked from our fetcher. Must verify via login or direct support contact before live PA bot deployment. Highest-risk gap.
4. **Topstep LFA typical lifespan** before Shoulder Tap — external "<1% survive long-term" is coarse. Forum-scraping for specific trader trajectories could sharpen.
5. **TopstepX API bot deployment status verification** — confirm current practice-account bot is API-based, not native execution.
6. **Exact per-size payout caps** at Bulenox (first-3-payout range values) and Apex (steps 2–5 intermediate values). Third-party sources give ranges, not exact numbers per step.
7. **Bulenox PDF Terms of Use** — not yet fetched (`https://bulenox.com/wa-data/public/site/data/bulenox.com/Terms_of_Use.pdf`). Likely contains cross-account hedging, account-stacking, and other prohibited-conduct detail.

---

## 10. Action items feeding into deployment plan

1. **Keep `topstep_50k_mnq_auto` profile at 50K.** 100K/150K offer no cash-flow advantage at 1ct. Consider 100K/150K only for LFA tier optionality later (low-priority).
2. **Do NOT over-promise TopStep income.** Ceiling is ~$25K/yr across 5 XFAs with B2F (before LFA consolidation wipes them). Multi-firm is where real scale lives.
3. **Scrape Bulenox + Apex canonical rules** before designing multi-firm deployment. Critical path to the $180K/yr ceiling.
4. **Before live TopStep deployment, verify bot connects via TopstepX API**, not native platform. Otherwise may be flagged for EA-on-native violation.
5. **Design correlation-risk mitigation** (rotation, lane split across firms, or size tapering) as part of multi-firm plan. Single outlier day must not wipe the whole book.
6. **Reference this audit in `trading_app/prop_profiles.py` module docstring** so future scaling decisions cite the canonical constraint, not recalculate from scratch.

---

## 11. Reproducibility

Simulations built inline during session (not committed as reusable scripts — yet). Reproducible from:
- `orb_outcomes` + `validated_setups` tables
- `lane_allocation.json` 2026-04-13 snapshot (6 live lanes)
- `trading_app.strategy_fitness._load_strategy_outcomes` as canonical eligible-day loader
- Raw bot daily PnL series saved to `/tmp/bot_daily_pnl.json` (ephemeral; re-run script to regenerate)

If reproducibility is needed long-term, promote the simulator scripts to `scripts/research/topstep_scaling_sim.py`. Deferred until there's a second consumer of the output.

---

## 12. Source manifest

**Canonical local docs (scraped Apr 8 2026, `docs/research-input/topstep/`, 20 files + 1 image, re-scrape quarterly):**
- `topstep_live_funded_parameters.md`
- `topstep_xfa_parameters.txt`
- `topstep_trading_combine_parameters.md`
- `topstep_payout_policy.txt`
- `topstep_mll_article.md`
- `topstep_dll_article.md`
- `topstep_consistency_target.md`
- `topstep_scaling_plan_article.md`
- `topstep_prohibited_trading_strategies.md`
- `topstep_cross_account_hedging.md`
- `topstep_dynamic_live_risk_expansion.md`

**External sources (multi-firm + LFA survival + bot policy):**
- [propfirmapp — Topstep Review](https://propfirmapp.com/prop-firms/topstep) — LFA <1% survival stat
- [propfirmapp — Multiple Prop Firm Accounts](https://propfirmapp.com/learn/multiple-prop-firm-accounts) — legal status, rotation strategy
- [QuantVPS — Topstep vs Apex vs Bulenox](https://www.quantvps.com/blog/topstep-vs-apex-vs-bulenox)
- [QuantVPS — Topstep Trade Copier](https://www.quantvps.com/blog/topstep-trade-copier)
- [TradersX — Bulenox Overview](https://tradersx.io/prop-firms/bulenox) — 11-account ceiling, 100%/90% split
- [OnlyPropFirms — Automated Futures Trading](https://onlypropfirms.com/articles/automated-futures-trading) — Apex 20 accounts, bot support
- [Cointracts — Trade Multiple Prop Firms](https://www.cointracts.com/learning-hub/can-you-trade-multiple-prop-firms-at-once-master-the-copy-trading-rules)
- [h2tfunding — Does Topstep Allow Automated Trading](https://h2tfunding.com/does-topstep-allow-automated-trading/)

---

## Revision history

- 2026-04-15: Initial creation. Audit triggered by user challenge on LFA consolidation rule. Corrected prior Claude-session assumption; rebuilt scaling math around XFA↔LFA exclusivity and multi-firm reality.
