---
name: MyFundedFutures (MFFU) Canonical Source Corpus
purpose: Frozen-in-time verbatim text from help.myfundedfutures.com, used as the authoritative ground truth for all MFFU compliance/sizing code in this repo
type: research-input-corpus
firm: mffu
established: 2026-05-31
last_scraped: 2026-05-31
authority: PRIMARY-SOURCE — overrides training memory, MEMORY.md, code comments, and CLAUDE.md where they conflict
---

# MyFundedFutures (MFFU) Canonical Source Corpus

**Created:** 2026-05-31
**Trigger:** Re-establish prop-firm rule foundation before any drawdown sim / sizing / deploy decision (operator-emphatic: stale firm numbers ⇒ the whole downstream chain is garbage). MFFU is the (eventual) deploy target for the MNQ book.
**Next re-fetch:** 2026-08-31 (quarterly), or immediately after any MFFU policy announcement.
**Source mix:** `help.myfundedfutures.com` (Intercom help center) — **canonical, verbatim only**. No third-party reviewers. No paraphrase.

## Why this exists

Per CLAUDE.md "Volatile Data Rule" and `.claude/rules/institutional-rigor.md` rule 7, every MFFU sizing/compliance fact in production code must trace to a primary source captured at a known point in time. Each `mffu_*.md` file is the **full verbatim curl→text dump** of one help-center article (structural HTML tags → newlines, no word changed), with a provenance header (source URL, article ID, scrape date, article "Updated" date, content-image manifest).

**Fetch method (reproducible):** `curl -sL -A "<browser UA>" <url> | html2text`. The MFFU/Intercom help center is server-rendered, so curl returns the literal rule text. WebFetch was **not** used (its summarizer paraphrases — forbidden). Image-borne rules are flagged in each file's header (see § Image guard).

## Plan taxonomy (live, verified 2026-05-31)

MFFU runs **four** plans (corrects stale memory which guessed "Builder/Rapid/Scale/Expert"):

| Plan | Collection ID | Sizes published | Drawdown model | Sim split | Sim consistency |
|---|---|---|---|---|---|
| **Rapid** | 17350372 | 25K, 50K, 100K, 150K | **Intra-day trailing** (locks at $100) | 90/10 | None in sim |
| **Builder** | 19480282 | 50K only (2 MLL options) | **EOD trailing** | **80/20** | 50% (resets per payout) |
| **Flex** | 18158247 | 25K, 50K | **EOD trailing** + $1,000 intraday soft-pause | **80/20** | 50% (eval only) |
| **Pro** | 18640090 | 3 scale plans (per-size grid not in Pro collection — see gap note) | (highlights: no daily loss limit) | — | see consistency article |

> **The repo (`trading_app/prop_profiles.py`) models MFFU as Rapid ONLY.** Builder, Flex, and Pro are entirely unmodeled. See the reconciliation matrix `docs/audit/results/2026-05-31-prop-rule-reconciliation.md`.

---

## 1. Rapid Plan — per-size evaluation parameters (verbatim, EOD MLL column)

Source: `mffu_rapid_25k.md` (14116402), `mffu_rapid_50k.md` (13134709), `mffu_rapid_100k.md` (13286542), `mffu_rapid_150k.md` (13286582).

| Size | Profit Target | Max Loss Limit (EOD) | Daily Loss | Max Contract | Consistency | Min Days | T1 News |
|---|---|---|---|---|---|---|---|
| 25K | $1,500 | $1,000 | None | 3 mini / 30 micro | 50% (Eval Only) | 2 Days | Yes |
| 50K | $3,000 | $2,000 | None | 5 mini / 50 micro | 50% (Eval Only) | 2 Days | Yes |
| 100K | $6,000 | $3,000 | None | 10 mini / 100 micro | 50% (Eval Only) | 2 Days | Yes |
| 150K | $9,000 | $4,500 | None | 15 mini / 150 micro | 50% (Eval Only) | 2 Days | Yes |

**Rapid Sim Funded (verbatim, `mffu_rapid_50k.md`):** Initial balance $0; Max Loss Distance $2,000 from equity HWM; **Drawdown Type: Intra-day trailing**; "Once your trailing Max Loss reaches $100, it locks there" (25K locks at $25,100 — i.e. start+$100); Consistency: None in Sim Funded; News: no T1 in sim funded; **Required buffer $2,100** before payout; Payout frequency **Daily — every 24h**; min payout $500; **Profit split 90% trader / 10% MFFU**.

---

## 2. Builder Plan — 50K, two MLL options (verbatim, `mffu_builder_50k.md` 14290805)

Builder is **EOD trailing**, NOT intraday like Rapid. Two MLL options at checkout (the ONLY eval difference):

| Parameter | $2,000 MLL option | $1,500 MLL option |
|---|---|---|
| Profit Target | $3,000 | $3,000 |
| Max EOD Drawdown (MLL) | $2,000 | $1,500 |
| Starting Minimum Balance | $48,000 | $48,500 |
| Daily Loss Limit | $1,000 — soft pause | $1,000 — soft pause |
| Max Contracts | 4 Minis / 40 Micros | 4 Minis / 40 Micros |
| Consistency Rule (eval) | None | None |
| Minimum Trading Days | 1 Day | 1 Day |
| News Trading | Allowed | Allowed |
| Drawdown Model | EOD Trailing | EOD Trailing |

**Builder Sim Funded payouts (verbatim):** "The Builder Plan runs on a flat **$2,000 payout cap, up to 5 sim payouts, each paid at 80/20** in your favor." 50% consistency rule applies in sim ("single largest profit day cannot represent more than 50% of total profits in the cycle; resets after each approved payout"). MLL "locks permanently once it reaches $100 above the starting balance." **Only one Sim Funded account active per user; after breach, earliest new account = following trading day.**

**Builder Transition to Live:** after 5th approved sim payout → live funded. Live MLL matches option ($2,000 or $1,500 EOD trailing). Live payouts daily. No consistency rule on live. Post-breach cooldown protocol applies (see article).

---

## 3. Flex Plan — 25K, 50K (verbatim, `mffu_flex_25k.md` 15070844, `mffu_flex_50k.md` 15072271)

Flex 50K (verbatim): Profit target **$3,000**; Max loss limit (EOD) **$2,000**; Daily loss limit **None**; Consistency **50% (evaluation stage only)**; **Drawdown model: EOD trailing**; **Activation fee $0**; adds a **$1,000 intraday soft pause**. Sim funded: starts at $0. Payout policy: net profit between payouts $500, min payout $500, **max sim payouts 5**, consistency in payout stage **None**, withdraw up to **50% of total profits per payout**, **max payout per request $2,000**, **profit split 80/20** (keep 80%). (Legacy 25K Flex article 13536146 + sim-reset 13536701 + path-forward 13521620 also captured.)

---

## 4. Pro Plan (verbatim, `mffu_pro_highlights.md` 11802674)

Pro Plan (Nov 2025 highlights): "Three Scale Plans to Choose From", main objective transition to live funding, **No Daily Loss Limit**, larger payouts. Consistency article `mffu_pro_consistency.md` (8694840) + 1-day add-on `mffu_pro_1day_addon.md` (12879226) captured.

> **GAP (honest):** the Pro collection has only 3 articles (highlights, consistency, 1-day add-on). The **per-size Pro evaluation grid (profit target / MLL / contracts by size) is NOT published as text in the Pro collection** — the highlights article links out to "evaluation rules". Not fabricated here. Re-fetch needs the Pro evaluation-rules article (linked from highlights) if Pro is ever modeled.

---

## 5. Firm-wide rules (NOT a fixed schema — captured as-is)

Each captured verbatim in its own file:

- **Drawdown mechanics** — `mffu_intraday_drawdown_explained.md` (12802721). Rapid = intraday trailing $2,000 from HWM, locks at $100. (6 content images = platform UI screenshots of the DD panel, visually verified illustrative — no extra numeric rule.)
- **News trading** — `mffu_news_trading_policy.md` (8230009). T1 (high-impact) news restrictions; differs eval vs sim funded.
- **Fair play / automation** — `mffu_fair_play_prohibited.md` (8444599). Automated strategies permitted; HFT/latency-arb/copy-abuse prohibited.
- **Activation fee** — `mffu_activation_fee.md` (12398151). "$0 activation fees for any of our plans."
- **Inactivity rule** — `mffu_inactivity_rule.md` (11972075).
- **Hedging** — `mffu_hedging.md` (12011241).
- **Cross-instrument policy** — `mffu_cross_instrument_policy.md` (10244682).
- **2% price-limit rule** — `mffu_2pct_price_limit_rule.md` (9698984). **Contains an image-borne CME price-limit table** transcribed into the file (illustrative example, not a per-plan rule).
- **Payout policy overview** — `mffu_payout_policy_overview.md` (13745661).
- **Live funded account** — `mffu_live_funded_account.md` (10101257) + FAQ `mffu_live_accounts_faq.md` (12109396). Live = EOD DD; forced-Live trigger; reserve/bonus.
- **Rapid reserve / performance bonus** — `mffu_rapid_reserve_bonus.md` (13286746).
- **Account sharing** — `mffu_account_sharing.md` (8230018).

---

## 6. Image guard (per operator caution: "some of it might be images, not text")

Every snapshot file's header lists a **content-image manifest**. Boilerplate (logos, social icons, emoji, avatars) is filtered. Articles flagged with content images were **visually inspected** this scrape:

- `mffu_intraday_drawdown_explained.md` — 6 images = trading-platform UI screenshots (DD panel showing the $2,000/$48,007.49 example). Rule already in text. ✅ no rule lost.
- `mffu_news_trading_policy.md` — 2 images (illustrative). Rule in text.
- `mffu_2pct_price_limit_rule.md` — 1 image = **a real rule-table (CME price limits 7%/13%/20% for NQU4)** NOT in body text → **transcribed verbatim into the file** so it is not silently dropped. Illustrative example, not an MFFU per-plan sizing rule.

No MFFU per-plan sizing rule (profit target / MLL / contracts / payout) was found to be image-only — all such tables render as HTML text.

---

## 7. File index

| File | Article ID | Updated (per article) | Coverage |
|---|---|---|---|
| `mffu_rapid_25k.md` | 14116402 | Mar 25, 2026 | Rapid 25K eval+sim params |
| `mffu_rapid_50k.md` | 13134709 | Mar 20, 2026 | Rapid 50K eval+sim params + payouts |
| `mffu_rapid_100k.md` | 13286542 | Mar 20, 2026 | Rapid 100K |
| `mffu_rapid_150k.md` | 13286582 | Apr 30, 2026 | Rapid 150K |
| `mffu_rapid_live.md` | 13134718 | recent | Rapid → Live transition params |
| `mffu_rapid_reserve_bonus.md` | 13286746 | Jan 16, 2026 | Reserve program / performance bonus |
| `mffu_intraday_drawdown_explained.md` | 12802721 | Jan 8, 2026 | Intraday DD mechanics (+6 UI imgs) |
| `mffu_builder_50k.md` | 14290805 | Apr 15, 2026 | Builder 50K full guide (2 MLL options) |
| `mffu_flex_25k.md` | 15070844 | recent | Flex 25K guide |
| `mffu_flex_50k.md` | 15072271 | recent | Flex 50K guide + payout policy |
| `mffu_flex_25k_legacy.md` | 13536146 | Mar 9, 2026 | Legacy 25K Flex |
| `mffu_flex_sim_reset.md` | 13536701 | Jan 27, 2026 | Flex sim-funded reset |
| `mffu_flex_path_forward.md` | 13521620 | Mar 25, 2026 | Flex path forward |
| `mffu_pro_highlights.md` | 11802674 | Nov 11, 2025 | Pro sim+live highlights |
| `mffu_pro_1day_addon.md` | 12879226 | recent | Pro 1-day add-on |
| `mffu_pro_consistency.md` | 8694840 | Feb 23, 2026 | Consistency for sim/expert/pro |
| `mffu_news_trading_policy.md` | 8230009 | Feb 23, 2026 | News policy (+2 imgs) |
| `mffu_fair_play_prohibited.md` | 8444599 | Nov 24, 2025 | Automation / prohibited practices |
| `mffu_activation_fee.md` | 12398151 | Feb 4, 2026 | $0 activation fee |
| `mffu_inactivity_rule.md` | 11972075 | Nov 10, 2025 | Inactivity rule |
| `mffu_hedging.md` | 12011241 | Nov 10, 2025 | Hedging policy |
| `mffu_cross_instrument_policy.md` | 10244682 | Mar 3, 2026 | Cross-instrument policy |
| `mffu_2pct_price_limit_rule.md` | 9698984 | Mar 2, 2026 | 2% price-limit rule (+img table transcribed) |
| `mffu_payout_policy_overview.md` | 13745661 | Feb 22, 2026 | Payout policy overview |
| `mffu_live_funded_account.md` | 10101257 | Feb 25, 2026 | Live funded account |
| `mffu_live_accounts_faq.md` | 12109396 | Nov 10, 2025 | Live accounts FAQ |
| `mffu_account_sharing.md` | 8230018 | Nov 10, 2025 | Can someone else trade my account |

**Total: 27 verbatim article snapshots.**
