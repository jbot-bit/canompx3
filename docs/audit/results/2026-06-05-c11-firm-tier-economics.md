# C11 Firm × Tier Economics — Does a bigger account or a different firm earn more?

**Date:** 2026-06-05 · **Tool:** Claude Code · **Mode:** READ-ONLY research (Tier A)
**Profile under study:** `topstep_50k_mnq_auto` (3-lane MNQ, 1 micro/lane, dynamic allocator)

## Scope / Question

The operator asked, while away: (3) does the **$150k** path earn more overall
than $100k? (4) check **payout policies**; (5) wasn't **MyFundedFutures** the
better firm? (6) do they allow **bots**? (7) **web-verify** firm policies (8) and
two follow-ups raised mid-pass: (9) what about the **Builder** account, and (10)
the **account-cycling / "hold funds, extract on a cadence, spin up another when
one is force-transitioned and useless"** logic.

This is a **net-extractable-return** question across {firm} × {tier 50k/100k/150k},
not a gate-clearance one. It is read-only: no profile was created, no code or DB
written. Profile creation (if warranted) is a separate Tier-B stage.

## Decision / Verdict

**Bigger account size does NOT increase return on this book.** At the book's
hardcoded **1 micro/lane** sizing, gross edge is **identical** across $50k / $100k
/ $150k — a bigger MLL buys only drawdown headroom, not earnings. **Stay on
`topstep_50k_mnq_auto`; solve C11 via the ORB cap fix, not a bigger account.**
This is a *null result on the size axis* and it is the correct, intended answer —
not a failure to find a better option.

A bigger tier's *one* real advantage is that its express DD belt (0.90 × MLL)
clears the C11 drawdown gate without needing the cap fix. But that is a
gate-clearance lever bought at higher account cost, identical in effect to the
cheaper cap fix already proven on the 50k profile (peer terminal, 2026-06-05:
`cap_x0.75` → 90d DD $1,535 ≤ $1,800 belt; `cap_x0.80` → $1,594, 97.5% edge).

**Firm:** no firm beats Topstep for *this automated book.* MFFU/Tradeify/Bulenox
match the 90/10 split and all allow bots, but MFFU's Rapid and Builder plans carry
a **forced-Live transition trap** that is hostile to an unattended bot. Builder is
**80/20** (worse split) and a **single-account funnel**. The account-cycling
("make another") strategy is **structurally throttled** by 1-account-per-user +
21-day cooldown rules — the firms engineer against it.

## Findings (source-traced — values PRINTED from canonical loaders)

Trace script: `C:/Users/joshd/c11_matrix/firm_tier_economics_trace.py` (read-only,
outside repo). Imports `trading_app.prop_profiles`, `prop_firm_policies`,
`account_survival`.

### 1. The load-bearing fact: contracts do NOT scale with account size

`trading_app/account_survival.py:615-616` hardcodes
`contracts_per_trade_micro=1` ("Current project daily lanes are 1-contract micro
lanes per account") — **independent of `account_size`**. The tier's
`max_contracts_micro` ceiling (50/100/150) is never consulted for sizing. So the
book runs the **same 3 micros on every tier** → identical gross edge. This is the
executable proof of the `prop_profiles.py:16` comment (whose ~6-week-stale vintage
the plan rightly flagged): re-derived here by execution, it holds.

### 2. Topstep tier table (resolved, not inferred)

| tier | MLL | express belt (0.90×MLL) | tier micro cap | **book micros run** |
|---|---|---|---|---|
| $50k | $2,000 | **$1,800** | 50 | **3** |
| $100k | $3,000 | **$2,700** | 100 | **3** |
| $150k | $4,500 | **$4,050** | 150 | **3** |

Verifies the memory claim "$2,700 for 100k": the budget resolver multiplies the
**tier's own MLL** by `STRICT_DD_BUDGET_FRACTION_EXPRESS = 0.90`, not a hardcoded
$2,000. (Code comment at `account_survival.py:65` says "$1,800" — that is just the
$50k instance, not a hardcode.)

The full uncapped book's worst 90d DD is **$2,039** (peer-confirmed baseline). So:
- $50k belt $1,800 → **FAILS** by $239 (needs the cap fix, or a bigger tier).
- $100k belt $2,700 → **CLEARS** uncapped ($661 room), zero edge loss.
- $150k belt $4,050 → clears with even more room, but **no extra earnings** vs $100k.

⇒ Between $100k and $150k: **$150k earns the same and clears the same gate with
more slack you don't need.** $150k is strictly dominated on return-per-account-cost
for this book. The operator's "$150k more overall return?" → **No.**

### 3. Payout policies (thread 4) — and a CANON-vs-LIVE DRIFT

Repo canon (`PAYOUT_POLICIES`, scraped 2026-04-08) models a **flat** per-payout
cap: Standard $5,000 / Consistency $6,000, 90/10 split, 50%-of-balance cap, 5
winning days of $150+.

**⚠ Web verification found the live policy has changed — flag as
`UNSUPPORTED until reconciled`:** the official Topstep help center
([article 8284233](https://help.topstep.com/en/articles/8284233-topstep-payout-policy),
"updated today") now shows **tier-scaled** per-payout caps:

| | $50k | $100k | $150k |
|---|---|---|---|
| Standard | $2,000 | $3,000 | **$5,000** |
| Consistency | $3,000 | $4,000 | **$6,000** |

Plus a separate [April-28-2026 cut](https://tradecovex.com/guides/topstep-payout-rules-2026)
to new no-activation-fee $50k/$100k Combine caps. **The repo's flat $5k/$6k is now
only true for the $150k tier.** This is a genuine post-scrape ToS drift the canon
doesn't reflect.

**Does it change the verdict? No — the cap does not bind this book.** The book
earns ~**$7,000–$8,000/yr** at 1 contract (per
`docs/audit/results/2026-06-01-mnq-usdata-capital-fit-v1.md`), ≈ $600/month gross.
That is far below even the $50k's $2,000/payout cap. Tier-scaled caps *would* favor
a bigger account **for a higher-throughput book**, but this book is throughput-
limited by its own edge, not by the payout cap at any tier. So the live drift
sharpens the general picture without flipping this book's conclusion.

### 4. Cross-firm comparison (threads 5–6)

| firm | 50k MLL | 100k MLL | 150k MLL | top split | bots? (`auto_trading`) |
|---|---|---|---|---|---|
| Topstep | $2,000 | $3,000 | $4,500 | 90% | full |
| MyFundedFutures (Rapid) | $2,000 | $3,000 | $4,500 | 90% | full |
| Tradeify | $2,000 | $3,000 | $4,500 | 90% | full |
| Bulenox | $2,500 | $3,000 | $4,500 | 90% | full |
| MFFU Builder | $2,000 / $1,500 add-on | — (50k only) | — | **80%** | full |

**Bot policy (thread 6):** all four firms allow automation. Canon's
`auto_trading="full"` for MFFU is **web-confirmed live**
([article 8444599](https://help.myfundedfutures.com/en/articles/8444599)):
*"Traders may make use of automated trading strategies… High-frequency Trading is
not allowed… Copy trading other traders is forbidden."* Our 3-lane ORB book is not
HFT and does not copy-trade, so it is compliant.

**Was MFFU "the better firm" (thread 5)?** Not for an unattended bot:
- MFFU **Rapid** matches Topstep's 90/10 and MLLs, BUT carries a **forced-Live
  transition trap** — web-confirmed
  ([Understanding Rapid Live](https://help.myfundedfutures.com/en/articles/13134718-understanding-rapid-live)):
  **$10,000 net profit in a single session → automatic Live transition** (profit
  over $10k that day forfeited); **21-day cooldown** after a Live breach; up to
  **$5,000 of sim profit held in Reserve** on transition. A bot can't *choose* not
  to hit the trigger, and forced-Live exposes real-capital semantics (reduced
  4/40 contracts, EOD DD) the operator has not gated.
- Topstep's XFA path has no such forced-promotion; promotion to LFA is the
  operator's choice (and XFA/LFA are mutually exclusive — `prop_profiles.py:13`).

### 5. Builder account (thread 9) + account-cycling logic (thread 10)

**Builder** (`mffu_builder`, web-confirmed
[article 14290805](https://help.myfundedfutures.com/en/articles/14290805-builder-plan-a-comprehensive-guide)):
- **80/20 split** (worse than the 90/10 elsewhere), $50k-only, no activation fee.
- **$2,000 payout cap/cycle, max 5 sim payouts → then FORCED to Live.** Hard
  ceiling: **$10,000 total sim extraction** before forced promotion.
- **Only 1 sim funded account per user** (cannot parallelize Builders).
- Payouts every 48h; 21-day post-breach cooldown; reset ~$84+ (Flex 50k reset
  ~$499 as a reference — Builder-specific reset price not separately confirmed).

**The "hold funds + extract on a cadence + make another when one is useless"
strategy (thread 10) is structurally throttled by design.** The firms engineer
against farming:
- **1 active sim account per user** (Builder, and effectively for serial Rapid
  cycling) ⇒ you cannot run a fleet in parallel; cycles are serialized.
- **Forced-Live funnel** (Builder: after 5th payout; Rapid: at $10k/day) ⇒ you
  cannot stay in the lucrative sim phase indefinitely — the firm *pushes* you to
  real-capital Live, where the favorable sim economics end.
- **21-day cooldown after a Live breach** ⇒ "make another" has a 3-week lockout
  tax per blow.
- So the realistic Builder cycle is: sim-farm ≤ $10k (at 80/20 = $8k take) → forced
  Live → if it breaches, 21-day lockout, then repeat. For *this* ~$600/month book,
  reaching the 5-payout ceiling would take many months; the funnel and the worse
  split make Builder strictly worse than holding a Topstep XFA for an automated
  low-throughput strategy.

**Bottom line on cycling:** it is a real strategy class, but it suits a
high-throughput discretionary trader who can intentionally manage the
sim→live→reset cadence — not an unattended 1-micro ORB bot whose monthly take is
an order of magnitude below the per-cycle caps. No cycling edge for our book.

## Reproduction / Files

- **Trace (read-only, outside repo):** `C:/Users/joshd/c11_matrix/firm_tier_economics_trace.py`
  → run with `PYTHONPATH=<repo> PYTHONIOENCODING=utf-8 python …`. Prints §1–§4
  resolved values from canonical loaders.
- **Canonical sources read:** `trading_app/prop_profiles.py` (`ACCOUNT_TIERS:495+`,
  `PROP_FIRM_SPECS`, `mffu_builder:324+`), `trading_app/prop_firm_policies.py`
  (`PAYOUT_POLICIES:36+`), `trading_app/account_survival.py`
  (`effective_strict_dd_budget:75`, `contracts_per_trade_micro:616`).
- **Book profit rate:** `docs/audit/results/2026-06-01-mnq-usdata-capital-fit-v1.md`
  (~$7–8k/yr at 1 contract).
- **Web verification (Firecrawl NOT connected this session — used WebSearch /
  WebFetch):** Topstep payout policy (help article 8284233 + tradecovex 2026 cap
  guide), MFFU automation (article 8444599), MFFU Rapid forced-Live (article
  13134718), MFFU Builder (article 14290805).

## Limitations / Caveats

- **`UNSUPPORTED until reconciled`:** repo `PAYOUT_POLICIES` flat $5k/$6k caps are
  stale vs the live tier-scaled / April-28-cut caps. Does not change THIS book's
  verdict (cap is non-binding at ~$600/mo), but the canon should be re-scraped and
  reconciled in a separate Tier-A pass before any payout-throughput modeling is
  trusted. Recorded as a debt item, not fixed here (read-only stage).
- **KNOWN DATA GAP — account fees not in canon:** verified there is **no structured
  per-tier eval/reset/activation fee field** in `prop_profiles.py` /
  `prop_firm_policies.py`. This comparison is therefore **gross of account cost**.
  Web-sourced reference points (Builder no activation fee; Flex 50k reset ~$499;
  resets from ~$84) are *web-sourced, not canon*. A precise net-of-fee ranking
  needs the firm pricing pages sourced and a fee field added — out of scope here.
- **Firecrawl not connected:** thread 7 asked for Firecrawl specifically; it is not
  an available MCP this session. Web claims above are WebSearch/WebFetch against
  official help-center URLs. If the operator wants Firecrawl specifically, that is
  a one-line "connect the Firecrawl MCP server" setup — not faked here.
- **Web sources mix official and secondary:** firm help-center articles are
  primary; tradecovex/proptradingvibes are secondary aggregators used only to
  corroborate the April-28 cap-cut date. Treated as unofficial per
  `targeted-grounding.md`.
- **Peer-terminal dependency:** the $2,039 baseline and cap_x0.75/0.80 DD figures
  are from a concurrent peer Claude session in the main checkout (settled by
  execution at commit 9429c540). Cited as peer-provided, not re-run here.
- **`9b3fc530` bracket-parity adversarial-audit gate remains OPEN** — no C11 arming
  follows from this research regardless of tier/firm choice.

## Sources (web)

- [Topstep Payout Policy — help center](https://help.topstep.com/en/articles/8284233-topstep-payout-policy)
- [Topstep Payout Rules 2026 (April-28 cap cut) — tradecovex](https://tradecovex.com/guides/topstep-payout-rules-2026)
- [MFFU Automated Trading Policy — article 8444599](https://help.myfundedfutures.com/en/articles/8444599)
- [MFFU Understanding Rapid Live — article 13134718](https://help.myfundedfutures.com/en/articles/13134718-understanding-rapid-live)
- [MFFU Builder Plan — article 14290805](https://help.myfundedfutures.com/en/articles/14290805-builder-plan-a-comprehensive-guide)
