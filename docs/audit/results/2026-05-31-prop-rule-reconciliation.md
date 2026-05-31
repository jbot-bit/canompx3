# Prop-Firm Rule Reconciliation — repo model vs verbatim live source (2026-05-31)

**Scope:** docs-only review artifact. Compares what `trading_app/prop_profiles.py` models
against the **verbatim help-center snapshots** captured this session under
`docs/research-input/{mffu,tradeify,topstep,bulenox}/`. **No code/schema edits, no sim run.**

**Legend:** `MATCH` repo == live · `Δ` value drifted · `NEW` published by firm, repo doesn't model ·
`MISSING` repo models nothing for this cell · `NO-SCHEMA-HOME` firm rule has no field in current
`PropFirmSpec`/`PropFirmAccount` schema (input to the later encoding decision).

**Source authority:** the `docs/research-input/<firm>/` verbatim snapshots win over this table and
over `prop_profiles.py` comments. Re-verify against the snapshot file cited per row.

---

## A. Headline finding (the foundation problem the operator flagged)

> Every drawdown sim / breach-probability / sizing decision is only as correct as these
> inputs. Three material problems found:

1. **MFFU is modeled as Rapid only.** Builder (the deploy target per HANDOFF/memory), Flex,
   and Pro are **entirely unmodeled** — yet Builder has a *different drawdown model*
   (EOD trailing vs Rapid's intraday), *different split* (80/20 vs 90/10), and a *consistency
   rule* (50%, resets per payout) Rapid's sim doesn't have. Sizing/survival math built on the
   Rapid spec does NOT transfer to a Builder account.
2. **MFFU contract caps in `ACCOUNT_TIERS` are LIVE-reduced values, not sim-funded values.**
   Repo: 100K=`6/60`, 150K=`8/80`. Verbatim sim-funded: 100K=`10/100`, 150K=`15/150`. The
   survival sim trades the *sim* account, so caps are understated (known bug, re-confirmed
   verbatim this session). 50K=`5/50` matches.
3. **Tradeify min-hold is mis-encoded.** Repo `min_hold_seconds=10` reads as a per-trade 10s
   minimum. Verbatim microscalping rule is *distributional*: ">50% of trades AND >50% of
   profit from trades held >10s." Different semantics. (Low impact for ORB — holds are
   27–100+ min — but the field lies about the rule.)

---

## B. MFFU — repo vs verbatim

Repo spec: `dd_type="intraday_trailing"`, split 90/10, consistency None, news_restriction True,
min_hold None. `ACCOUNT_TIERS`: (50K, 2000, 5/50), (100K, 3000, 6/60), (150K, 4500, 8/80).
Live source: `docs/research-input/mffu/` (scraped 2026-05-31).

### B1. Rapid plan (the only modeled plan)

| Size | Field | Repo | Live verbatim | Status |
|---|---|---|---|---|
| 25K | (all) | — | target $1,500 / MLL $1,000 / 3 mini 30 micro / 50% eval | **MISSING** (repo has no 25K) |
| 50K | profit target | (n/a in tier) | $3,000 | ref |
| 50K | MLL (sim) | $2,000 | $2,000 (intraday, locks at $100) | MATCH |
| 50K | contracts (sim) | 5 mini / 50 micro | 5 mini / 50 micro | MATCH |
| 100K | MLL | $3,000 | $3,000 | MATCH |
| 100K | contracts (sim) | **6 / 60** | **10 / 100** | **Δ** (repo = LIVE-reduced, not sim) |
| 150K | MLL | $4,500 | $4,500 | MATCH |
| 150K | contracts (sim) | **8 / 80** | **15 / 150** | **Δ** (repo = LIVE-reduced, not sim) |
| all | sim split | 90/10 | 90/10 | MATCH |
| all | sim consistency | None | None in sim funded | MATCH |
| all | payout buffer | (none) | $2,100 required buffer | **NO-SCHEMA-HOME** |
| all | payout freq | (none) | Daily / 24h | **NO-SCHEMA-HOME** |

### B2. Builder plan — DEPLOY TARGET, entirely unmodeled

| Field | Repo | Live verbatim (`mffu_builder_50k.md`) | Status |
|---|---|---|---|
| plan exists | no | 50K, two MLL options ($2,000 / $1,500) | **NEW** |
| drawdown model | (Rapid intraday) | **EOD trailing** | **NEW** (differs from Rapid) |
| profit target | — | $3,000 | NEW |
| starting min balance | — | $48,000 ($2K opt) / $48,500 ($1.5K opt) | NEW |
| daily loss | — | $1,000 soft pause | **NEW / NO-SCHEMA-HOME** (soft-pause kind) |
| contracts | — | 4 mini / 40 micro | NEW |
| sim consistency | — | 50% (resets per payout) | **NEW / NO-SCHEMA-HOME** |
| payout structure | — | flat $2,000 cap, up to 5 sim payouts | **NEW / NO-SCHEMA-HOME** |
| split | — | **80/20** (not 90/10) | NEW |
| MLL lock | — | locks at $100 above start | NEW |
| 1-account rule | — | one sim funded active per user | **NO-SCHEMA-HOME** |
| live transition | — | after 5th approved sim payout | **NO-SCHEMA-HOME** |

### B3. Flex plan — unmodeled

| Field | Live verbatim (`mffu_flex_50k.md`) | Status |
|---|---|---|
| sizes | 25K, 50K | **NEW** |
| drawdown | EOD trailing + $1,000 intraday soft-pause | **NEW** |
| 50K target / MLL | $3,000 / $2,000 | NEW |
| consistency | 50% (eval only) | NEW |
| split | 80/20 | NEW |
| max payout/request | $2,000; up to 50% of profits; 5 sim payouts | **NO-SCHEMA-HOME** |
| activation fee | $0 | NO-SCHEMA-HOME |

### B4. Pro plan — unmodeled, partial source

| Field | Live verbatim | Status |
|---|---|---|
| plan exists | "Three Scale Plans", no daily loss limit, larger payouts | **NEW** |
| per-size grid | **not published in Pro collection** (links out) | **GAP — re-fetch needed** |

---

## C. Tradeify — repo vs verbatim

Repo spec: `dd_type="eod_trailing"`, split 90/10, consistency None, news False,
`min_hold_seconds=10`. `ACCOUNT_TIERS`: (50K, 2000, 4/40), (100K, 3000, 8/80), (150K, 4500, 12/120).
Live source: `docs/research-input/tradeify/` (scraped 2026-05-31).

| Plan | Size | Field | Repo | Live verbatim | Status |
|---|---|---|---|---|---|
| Select Eval | 50K | MLL / contracts | 2000 / 4 mini 40 micro | $2,000 / 4 mini 40 micro | MATCH |
| Select Eval | 50K | consistency | None | **40% (eval only)** | **Δ** (repo None; live 40% in eval) |
| Select Eval | 100K | MLL / contracts | 3000 / 8/80 | $3,000 / 8 mini 80 micro | MATCH |
| Select Eval | 150K | MLL / contracts | 4500 / 12/120 | $4,500 / 12 mini 120 micro | MATCH |
| Select Eval | 25K | (all) | — | $1,500 / $1,000 / 1 mini 10 micro / 40% | **MISSING** |
| Select funded | all | consistency | None | None (funded) | MATCH |
| Growth | all | plan exists | no | EOD trailing, no eval consistency, DLL soft breach (25K $600/50K $1,250/100K $2,500/150K $3,750), funded consistency 35% | **NEW** |
| Lightning | all | plan exists | no | progressive consistency 20→25→30%, instant payouts | **NEW** |
| Select Daily | all | plan exists | no (only "Select Flex" implied) | distinct DD-lock column + payout policy | **NEW** |
| all | DD lock floor | (static lock @ start) | lock at $100 above start; per-plan EOD lock table | **Δ / NO-SCHEMA-HOME** (per-plan lock points differ: see `tradeify_rules_trailing_max_drawdown.md`) |
| all | min hold | `min_hold_seconds=10` | microscalping: >50% trades AND >50% profit held >10s | **Δ** (semantics differ) |
| all | hedging breach | (none) | ALL 3: opposing micro+mini + >10s + >$250 profit | **NO-SCHEMA-HOME** |

---

## D. TopStep — repo vs verbatim (re-scrape diff)

Re-scraped 2026-05-31 (articles "Updated yesterday/this week"). Core values stable; MLL article
was **rewritten** (clearer prose, added explicit "MLL starts $2,000 below" line; 50K TC MLL floor
= $48,000). Scaling ladder image re-downloaded and **visually re-verified — unchanged** from the
2026-04-08 `images/xfa_scaling_chart.png` parse (50K 2/3/5 · 100K 3/4/5/10 · 150K 3/4/5/10/15 lots).

| Size | Field | Repo | Live verbatim | Status |
|---|---|---|---|---|
| 50K | MLL | $2,000 | $2,000 (50K XFA) | MATCH |
| 100K | MLL | $3,000 | $3,000 | MATCH |
| 150K | MLL | $4,500 | $4,500 | MATCH |
| 50/100/150K | scaling ladder | top-of-ladder 5/10/15 mini | image ladder (day-1 lower) | **Δ** (known F-1: repo = top-of-ladder, day-1 enforcement pending) |
| all | MLL article prose | (cite 2026-04-08) | rewritten 2026-05-31 | **Δ (doc refreshed, values same)** |
| LFA | DLL | deferred (F-3) | $2K/$3K/$4.5K, $10K low-bal → $2K | NO-SCHEMA-HOME (deferred, no LFA today) |

---

## E. Bulenox — repo vs verbatim

Repo spec: `dd_type="eod_trailing"`, split tiers, consistency None, news False.
`ACCOUNT_TIERS`: (25K, 1500, 2/20), (50K, 2500, 5/50), (100K, 3000, 10/100), (150K, 4500, 15/150),
(250K, 5500, 25/250). Live source: `docs/research-input/bulenox/` (verbatim added 2026-05-31).

| Size | Field | Repo | Live verbatim | Status |
|---|---|---|---|---|
| 50K | max DD | $2,500 | $2,500 (qualification trailing) | MATCH |
| 100K | max DD | $3,000 | $3,000 | MATCH |
| 150K | max DD | $4,500 | $4,500 | MATCH |
| all | Option 1 vs Option 2 | (single eod_trailing) | **Option 1 trailing no-DLL** vs **Option 2 EOD+scaling+DLL** | **Δ / NO-SCHEMA-HOME** (repo collapses the two options) |
| 50K | daily loss (Opt 2) | (none) | $1,100 | **NEW / NO-SCHEMA-HOME** |
| 100K/150K/250K | daily loss (Opt 2) | (none) | $2,200 / $3,300 / $4,500 | NEW / NO-SCHEMA-HOME |
| all | per-tier scaling ladder | top-of-ladder contracts only | balance-tier ladders (50K 2/4/7 …) | **Δ / NO-SCHEMA-HOME** |
| Master | consistency | None | **40% single-day rule** | **Δ** (repo None; live 40% on Master) |
| Master | payout caps | (none) | first 3 payouts capped, then uncapped | **NO-SCHEMA-HOME** |
| Master | profit split | (split tiers) | 100% first $10K, then 90/10 | verify vs repo tiers |
| Master | activation fee / reserve | (notes only) | one-time fee + locked reserve | NO-SCHEMA-HOME |

---

## F. NO-SCHEMA-HOME rule inventory (input to the later encoding decision)

Firm rule-kinds with no field in current `PropFirmSpec` / `PropFirmAccount`:

- **Payout cap + count + frequency per cycle** (MFFU Builder flat $2,000×5; Flex $2,000/req×5;
  Bulenox first-3-capped-then-uncapped; Rapid daily/24h + $2,100 buffer).
- **Soft-pause daily loss** (distinct from hard daily-loss breach) — MFFU Builder/Flex $1,000;
  Tradeify Growth/Lightning soft-breach DLL.
- **Per-balance-tier scaling ladders** (Bulenox; TopStep image ladder) — repo stores only
  top-of-ladder caps.
- **Per-plan EOD drawdown lock points** (Tradeify 4-plan lock table).
- **Progressive / resetting consistency** (MFFU Builder 50% resets per payout; Tradeify
  Lightning 20→25→30%).
- **MLL-option fork at checkout** (MFFU Builder $2,000 vs $1,500).
- **Forced-Live / transition triggers** (MFFU after 5th sim payout; one-sim-account-per-user).
- **Distributional min-hold / microscalping** (Tradeify >50% trades + >50% profit >10s).
- **Multi-condition hedging breach** (Tradeify ALL-3 conditions).
- **Reserve / performance-bonus structures** (MFFU Rapid reserve; Bulenox locked reserve).

---

## G. Verification performed this session

1. **Verbatim spot-check (byte-for-byte, 3 cells)** — see § H below: re-curled MFFU Rapid 50K,
   Tradeify Select 50K, Bulenox 50K daily-loss and matched the on-disk quoted text.
2. **Enumeration completeness** — MFFU README lists all 4 plans + every size found in the live
   collection listings (Rapid 25/50/100/150K, Builder 50K, Flex 25/50K, Pro 3-scale). Tradeify
   lists Growth/Select/Lightning × 25/50/100/150K.
3. **Image guard** — every flagged content image was visually inspected; the only image-borne
   *rule table* (MFFU 2% price-limit CME table) was transcribed verbatim into its snapshot. All
   per-plan sizing tables render as HTML text (no image-only sizing rule found).

## H. Spot-check log (byte-for-byte)

| Cell | Snapshot file quote | Fresh re-curl matched? |
|---|---|---|
| MFFU Rapid 50K | "Profit Target $3,000 … Maximum Loss Limit (EOD) $2,000 … Max Contract 5 mini / 50 micro" | ✅ |
| MFFU Rapid 50K lock | "Once your trailing Max Loss reaches $100, it locks there." | ✅ |
| Tradeify Select 50K | "Profit Target: $3,000 … Maximum Drawdown: $2,000 (End of Day) … Maximum Contracts: 4 mini / 40 micro" | ✅ |
| Bulenox 50K DLL | "50K Account - $1100 Daily loss limit" | ✅ |

---

## Out of scope (explicit follow-ups, gated on review of this corpus)

- Encoding any plan/size into `prop_profiles.py` (capital-path schema decision + adversarial gate).
- The `firm_specific_rules` schema fork (the NO-SCHEMA-HOME rules above).
- Freshness/snapshot/completeness drift checks in `pipeline/check_drift.py`.
- MFFU Builder backtest of the deployed MNQ book (`account_survival.py`).

**STOP HERE.** This table is the deliverable for review.

---

## Verdict / Decision

**VERDICT: foundation re-established; corpus ready for review. NO encoding yet.** The verbatim
snapshots are now the canonical prop-firm rule source on disk. Three repo-vs-live discrepancies
are confirmed material and MUST be resolved before any sizing/sim trusts `prop_profiles.py`:
(1) MFFU Builder (deploy target) is entirely unmodeled with a different DD model/split/consistency;
(2) MFFU contract caps are LIVE-reduced, not sim-funded (survival sim understates caps);
(3) Tradeify `min_hold_seconds=10` mis-encodes the distributional microscalping rule.
**Decision:** approve the corpus, then schedule the gated encoding follow-ups. Do not deploy or
re-size on the current repo spec for MFFU Builder/Flex/Pro until encoded.

## Reproduction / Outputs

- Fetch (reproducible): `curl -sL -A "<browser UA>" <url> | html2text` per article; scraper
  `.claude/scratch/_t/scrape_article.py` (UTF-8 decode, image-manifest guard).
- Enumeration: `curl <help-home> | grep collections/articles` → per-collection article lists
  (MFFU 4 plan collections + cross-cutting; Tradeify Accounts&Rules/Payouts/Live; Bulenox help
  index; TopStep articles).
- Outputs on disk: `docs/research-input/{mffu (28 files),tradeify (24),bulenox (7 new),topstep
  (5 refreshed)}/` + this matrix. 66 files, zero production-code diff (`git status` confined to
  `docs/`).
- Byte-for-byte spot-check (§ H): 3 cells re-curled and matched on-disk quoted text (MFFU Rapid
  50K MLL-lock; Tradeify Select 50K drawdown; Bulenox 50K daily-loss). `check_drift.py` = 172
  passed / 0 skipped / 22 advisory.

## Caveats / Disconfirming / Limitations

- **Pro per-size grid not captured** — the MFFU Pro collection (3 articles) does NOT publish the
  per-size eval grid as text; it links out. NOT fabricated. Re-fetch the linked Pro
  evaluation-rules article before modeling Pro.
- **Help-center text is volatile** — captures are point-in-time (2026-05-31). TopStep MLL article
  was rewritten upstream between 2026-04-08 and now (values stable, prose changed). Re-fetch
  quarterly (next 2026-08-31) or after any firm announcement.
- **Bulenox §2/§4 legacy prose** in its README predates the verbatim snapshots and may still carry
  paraphrase/third-party ranges; on any conflict the `bulenox_*.md` verbatim files win.
- **`html2text` is lossy on layout** — tables are linearized to text; the image guard flags
  image-borne content but only the MFFU CME price-limit table was manually transcribed. Other
  flagged images were visually verified as UI screenshots / worked examples (no unique rule).
- **Reconciliation reads repo values from `prop_profiles.py` comments + `ACCOUNT_TIERS`** as of
  this session's HEAD; a concurrent peer session was committing to `main` — re-verify repo side if
  `prop_profiles.py` changed after this commit.
- This is a **source-capture + diff artifact only**. It does not validate that any rule is correct
  for trading, nor run a survival/payout sim. Those are gated follow-ups.
