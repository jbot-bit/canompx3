---
name: Tradeify Canonical Source Corpus
purpose: Frozen-in-time verbatim text from help.tradeify.co, used as the authoritative ground truth for all Tradeify compliance/sizing code in this repo
type: research-input-corpus
firm: tradeify
established: 2026-05-31
last_scraped: 2026-05-31
authority: PRIMARY-SOURCE — overrides training memory, MEMORY.md, code comments, and CLAUDE.md where they conflict
---

# Tradeify Canonical Source Corpus

**Created:** 2026-05-31
**Trigger:** Re-establish prop-firm rule foundation before any drawdown sim / sizing / deploy decision. Tradeify is the repo's PRIMARY MNQ scaling lane (`prop_profiles.py`), previously modeled with NO snapshot source.
**Next re-fetch:** 2026-08-31 (quarterly), or immediately after any Tradeify policy announcement.
**Source mix:** `help.tradeify.co` (Intercom help center) — **canonical, verbatim only**. No third-party reviewers. No paraphrase.

## Why this exists

Same authority basis as the TopStep/MFFU corpora: every Tradeify sizing/compliance fact in production code must trace to a primary source captured at a known point in time. Each `tradeify_*.md` file is the **full verbatim curl→text dump** of one help-center article (structural HTML tags → newlines, no word changed) with a provenance header.

**Fetch method:** `curl -sL -A "<browser UA>" <url> | html2text`. Server-rendered → curl returns literal rule text. WebFetch not used (paraphrase forbidden).

## Plan taxonomy (live, verified 2026-05-31)

Tradeify's plan set is **richer than the repo models** (repo had a single "Select Flex" row). Three evaluation families + funded/live programs:

| Plan family | Drawdown | Eval consistency | Eval daily loss | Funded consistency | Sizes |
|---|---|---|---|---|---|
| **Growth** | EOD trailing | None | Yes (soft breach) | 35% | 25K/50K/100K/150K |
| **Select** (Flex & Daily) | EOD trailing | 40% (eval only) | None | None (funded) | 25K/50K/100K/150K |
| **Lightning** (funded) | EOD trailing | progressive 20→25→30% | Yes (soft breach) | progressive (see note) | 25K/50K/100K/150K |

All Tradeify accounts use **End-of-Day (EOD) trailing drawdown**, enforced real-time, hard-breach (no recovery). "Tradeify 3.0" program update (article 14135902) captured.

> **The repo (`trading_app/prop_profiles.py:307`) models Tradeify as a single "Select Flex" lane** (`min_hold_seconds=10`, no consistency). See reconciliation matrix `docs/audit/results/2026-05-31-prop-rule-reconciliation.md` for the gaps (Growth, Lightning, Select Daily unmodeled; min-hold mis-encoded — see § 5).

---

## 1. Select Evaluation — per-size (verbatim, `tradeify_select_evaluation.md` 12853921)

| Size | Profit Target | Max Drawdown (EOD) | Daily Loss | Consistency | Max Contracts (eval) |
|---|---|---|---|---|---|
| 25K | $1,500 | $1,000 | None | 40% | 1 mini / 10 micro |
| 50K | $3,000 | $2,000 | None | 40% | 4 mini / 40 micro |
| 100K | $6,000 | $3,000 | None | 40% | 8 mini / 80 micro |
| 150K | $9,000 | $4,500 | None | 40% | 12 mini / 120 micro |

Min 3 trading days (due to consistency rule). Note: accounts purchased before new-dashboard launch retain original $2,500 (50K) profit target including on reset.

---

## 2. Growth Evaluation — per-size (verbatim, `tradeify_growth_evaluation.md` 10495915)

EOD trailing DD, **no consistency** (can pass in 1 day), **daily loss limit (soft breach)** present:

| Size | Eval Max Contracts | Funded Max Contracts | Daily Loss Limit (soft breach) |
|---|---|---|---|
| 25K | 1 mini (10 micro) | — | $600 |
| 50K | 4 mini (40 micro) | 5 mini (50 micro) | $1,250 |
| 100K | 8 mini (80 micro) | 10 mini (100 micro) | $2,500 |
| 150K | 12 mini (120 micro) | 15 mini (150 micro) | $3,750 |

(Eval contracts are the lower set; funded contracts step up after passing.) Profit targets per size mirror Select ($1,500/$3,000/$6,000/$9,000 by size). Growth Sim Funded consistency = **35%** (per consistency article).

---

## 3. Lightning Funded — per-size (verbatim, `tradeify_lightning_funded.md` 10495938)

Same contract ladder as Growth (eval 1/4/8/12 mini; funded 5/10/15 mini for 50/100/150K). **Progressive consistency**: first payout 20%, second 25%, third+ 30% (accounts purchased after Sep 12 2025 8:00 AM EST); accounts before that date keep flat 20%. **Instant payouts** from dashboard once consistency + profit objective met. 150K Lightning purchased before Mar 31 daily-loss = $3,750. Legacy 150K pre-dashboard params retained on reset (see article).

---

## 4. Trailing Max Drawdown — lock table (verbatim, `tradeify_rules_trailing_max_drawdown.md` 10495897)

EOD balance at which the trailing drawdown LOCKS (then floor fixed at $100 above start, never moves up):

| Account Size | Growth | Lightning | Select Flex | Select Daily |
|---|---|---|---|---|
| 25K | $26,100 | $26,100 | $26,100 | $26,100 |
| 50K | $52,100 | $52,100 | $52,100 | $52,100 |
| 100K | $103,600 | $104,100 | $103,100 | $102,600 |
| 150K | $155,100 | $155,350 | $154,600 | $153,600 |

Verbatim: "Once locked, the drawdown floor becomes fixed at $100 above starting balance (e.g., $50,100 for 50K …) and never moves up again." Drawdown only locks on **Sim Funded** accounts, not Evaluations. Hard breach = permanent fail.

---

## 5. Consistency rule (verbatim, `tradeify_rules_consistency.md` 10468320)

- **Growth Sim Funded: 35%**
- **Select Evaluation: 40%** (evaluation phase only; **no consistency rule for Select accounts in funded mode**)
- **Lightning Funded** (purchased after Sep 12 2025 8:00 AM EST): gradual — first payout 20%, second 25%, third+ 30%. Before that date: flat 20% all payouts.
- Formula: `Biggest End of Day PnL / Consistency % = Total Balance Needed`.

### Microscalping / min-hold rule (verbatim — REPO MIS-ENCODES THIS)

From `tradeify_rules_news_trading.md` + `tradeify_rules_hedging_micros_minis.md`:

> "Over 50% of your trades are longer than 10 seconds AND over 50% of your profit must come from trades held longer than 10 seconds."

This is a **distributional rule** (>50% of trades AND >50% of profit must clear 10s), **NOT a per-trade 10-second minimum hold**. The repo's `min_hold_seconds=10` is a simplification. Hedging breach requires ALL THREE: opposing micro+mini positions, hedge duration >10 seconds, AND >$250 profit from the hedge — meeting only one or two does not breach.

---

## 6. Fees / payouts / firm-wide rules (captured verbatim)

- **Activation fees** — `tradeify_activation_fees.md` (10468246).
- **Commission fees** — `tradeify_commission_fees.md` (10468315).
- **Payout policies** — `tradeify_select_payout.md` (12853966, Select Flex & Daily), `tradeify_growth_payout.md` (11083796), `tradeify_lightning_payout.md` (10495932), `tradeify_rise_payouts.md` (12844518, main method).
- **News trading** — `tradeify_rules_news_trading.md` (10495874).
- **Permitted times** — `tradeify_rules_permitted_times.md` (10495876).
- **Supported products** — `tradeify_rules_supported_products.md` (10468222).
- **Daily loss limit (mechanics)** — `tradeify_rules_daily_loss_limit.md` (10468321). Soft breach (pauses, doesn't fail).
- **Pricing reference** — `tradeify_pricing_reference.md` (14369021).
- **Select vs Growth** — `tradeify_select_vs_growth.md` (13252431). **3.0 updates** — `tradeify_3_0_updates.md` (14135902). **New Select plan** — `tradeify_select_new_plan.md` (12987441).
- **Get funded / reset** — `tradeify_get_funded.md` (10495917), `tradeify_reset_evaluation.md` (10468256).

---

## 7. Image guard

Content-image manifest in each file header. Flagged + visually inspected this scrape:

- `tradeify_rules_consistency.md` — 1 image = a worked-example "Consistency Rule Table" (Trading Day / PnL / % with green/red pass-fail). The actual threshold %s are in the article text. Illustrative. ✅
- `tradeify_reset_evaluation.md` — 1 image (UI). No rule lost.

No Tradeify per-plan sizing rule (target / DD / contracts / daily-loss) was image-only — all render as HTML text (DD lock table, per-size grids all captured as text).

---

## 8. File index

| File | Article ID | Updated | Coverage |
|---|---|---|---|
| `tradeify_which_plan.md` | 10468265 | Apr 2, 2026 | Plan chooser overview |
| `tradeify_select_evaluation.md` | 12853921 | Apr 2, 2026 | Select eval per-size grid |
| `tradeify_growth_evaluation.md` | 10495915 | Apr 2, 2026 | Growth eval per-size + daily loss |
| `tradeify_lightning_funded.md` | 10495938 | Apr 7, 2026 | Lightning funded per-size + consistency |
| `tradeify_rules_trailing_max_drawdown.md` | 10495897 | Apr 10, 2026 | EOD DD lock table (4 plans × 4 sizes) |
| `tradeify_rules_consistency.md` | 10468320 | Apr 2, 2026 | Consistency %s (Growth/Select/Lightning) +img |
| `tradeify_rules_daily_loss_limit.md` | 10468321 | Apr 10, 2026 | Daily loss soft-breach mechanics |
| `tradeify_rules_hedging_micros_minis.md` | 10495868 | Apr 2, 2026 | Hedging + micro/mini + 10s breach conditions |
| `tradeify_rules_news_trading.md` | 10495874 | Apr 2, 2026 | News + microscalping (>50%/10s) rule |
| `tradeify_rules_permitted_times.md` | 10495876 | Apr 2, 2026 | Permitted trading times |
| `tradeify_rules_supported_products.md` | 10468222 | recent | Supported products/assets |
| `tradeify_select_new_plan.md` | 12987441 | Mar 31, 2026 | New Select plan + live-program changes |
| `tradeify_select_vs_growth.md` | 13252431 | recent | Select vs Growth comparison |
| `tradeify_3_0_updates.md` | 14135902 | Apr 7, 2026 | Tradeify 3.0 program updates |
| `tradeify_pricing_reference.md` | 14369021 | recent | Pricing reference |
| `tradeify_activation_fees.md` | 10468246 | Apr 2, 2026 | Activation fees |
| `tradeify_commission_fees.md` | 10468315 | Apr 28, 2026 | Commission fees |
| `tradeify_lightning_payout.md` | 10495932 | Apr 23, 2026 | Lightning funded payout policy |
| `tradeify_growth_payout.md` | 11083796 | Apr 23, 2026 | Growth funded payout policy |
| `tradeify_select_payout.md` | 12853966 | Mar 31, 2026 | Select Flex & Daily payout policies |
| `tradeify_rise_payouts.md` | 12844518 | recent | Rise payouts (main method) |
| `tradeify_get_funded.md` | 10495917 | Apr 2, 2026 | Getting funded after passing |
| `tradeify_reset_evaluation.md` | 10468256 | Apr 2, 2026 | Resetting a failed evaluation +img |

**Total: 23 verbatim article snapshots.**
