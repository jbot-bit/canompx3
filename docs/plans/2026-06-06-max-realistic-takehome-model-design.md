# Max-Realistic-Take-Home Model — Design

**Date:** 2026-06-06
**Mode:** plan (approved fork answers captured below; implementation NOT started)
**Author session:** 366cc3d6 (continuation)
**Status:** DESIGN APPROVED IN SHAPE — execute fresh (parent session hit 84% ctx)

---

## Purpose

Produce ONE decision-grade model answering "what is my realistic max take-home,
and on which accounts" where every dollar figure is either (a) canonical repo
data, (b) verified official Topstep documentation with source + scrape date, or
(c) an explicitly labelled assumption with a range. No fabricated point numbers.

The binding insight: the current analysis reports GROSS EDGE the bot generates,
but the real ceiling is the WITHDRAWAL constraint (per-payout cap x frequency x
90/10 split). A bot generating $77k of edge against a payout cap that only lets
out $30k/yr has a real take-home of $30k. The model must make the withdrawal
layer the headline constraint, not the edge layer.

## Operator fork answers (captured)

1. **Rule sourcing:** fetch fresh from Topstep help center with proper
   web-crawl + OCR (we scraped 2026-04-08; sibling note flags a late-Apr-2026
   cut → re-verify). Record each value with source article + date, same
   annotation style as existing `PAYOUT_POLICIES`.
2. **History window:** longest-common window across chosen lanes + stress
   haircut on edge AND drawdown; report window length prominently.

## What is ALREADY verified (do not re-derive)

- Tier MLLs: 50k=$2,000 / 100k=$3,000 / 150k=$4,500 (live `get_account_tier`).
- DD budget = 0.90 x tier MLL (express belt, `STRICT_DD_BUDGET_FRACTION_EXPRESS`).
- Split = 90/10 for accounts opened after 2026-01-12 (article 8284233).
- Sizing chain: per-micro PnL = `pnl_r * risk_dollars`; scales linearly with
  contract count; binding gate = worst rolling-90d historical DD <= budget
  (`account_survival.py`). `contracts_per_trade=1` hardcoded at :407, and
  `contracts_per_trade_micro=1` (:636) is a built-but-unused field.
- Universe: 848 active validated setups; ~18 distinct instrument-session pairs
  diversify (the rest are RR/filter variants = correlated, no diversification).
- Per-account ceiling (SHORT 14-month window, OPTIMISTIC, magnitudes UNRELIABLE):
  50k -> ~$19.5k/yr (16 lanes), 100k -> ~$44k/yr (45 ct), 150k -> ~$77k/yr (76 ct).
  These are annualized off one favorable year — directionally useful only.

## STALE / MISSING official rules (the sourcing pass must fix)

1. **Per-payout caps** — repo flat $5k(Std)/$6k(Consistency); official is
   TIER-SCALED (~50k=$2k, 100k=$3k, 150k=$5k Standard) + a late-Apr-2026 cut.
   MOST IMPORTANT correction. `prop_firm_policies.py:59,93`.
2. **Monthly subscription price per tier** — NOT in repo. Break-even between
   tiers depends entirely on this.
3. **Max concurrent funded accounts per person** — NOT in repo. The whole
   multiply-across-accounts lever rests on this number + copy-trade permission.
4. **Payout frequency** — how often a withdrawal can be requested. cap x
   frequency = real annual withdrawal ceiling, which may bind below the edge.

## The four-layer model

Take-home per account =
  min( edge_after_split , withdrawal_ceiling ) − cost

1. **Edge layer** — per-account gross PnL from canonical trade history, longest
   common window, stress haircut applied.
2. **Survival layer** — DD-budget gate per tier caps contracts/lanes (greedy
   sizer, already prototyped in `/tmp/max_book.py`).
3. **Withdrawal layer (NEW)** — verified tier-scaled per-payout cap x payout
   frequency x 0.90 split. The binding constraint.
4. **Cost layer (NEW)** — minus verified monthly subscription x N accounts +
   activation/reset fees.

Across-accounts answer = per-account take-home x verified-max-concurrent, costs
linear, edge does NOT degrade per account (independent trading) EXCEPT shared
attention/fills (copy-trading addresses, adds its own risk).

## Implementation steps (when executed)

1. Fresh web-crawl + OCR of Topstep help center → source the 4 gaps; annotate
   each with article + scrape date. Unverifiable → labelled assumption + range.
2. Correct stale payout-cap canon → tier-scaled (canonical-data change → gated,
   not silent; keep old values in history with note).
3. Standalone READ-ONLY analysis script (outside capital path): load
   uncorrelated lane set → build book per tier on longest common window →
   survival gate → withdrawal ceiling → costs → print four-layer waterfall per
   tier + per-account-then-multiplied total.
4. RECONCILE GATES (refuse to report if anchors fail): MNQ-subset must reproduce
   DD $2,038.84; existing 3-lane book must reproduce ~$1,005 p50. (Same tripwire
   that caught the earlier $2,798 reconstruction error.)
5. Decision table: per tier — edge-bound take-home, withdrawal-bound take-home,
   which binds, cost, net.

## Validate / failure modes guarded

- Short favorable window → longest-window + haircut + report window length.
- Gross edge mistaken for take-home → withdrawal layer is headline.
- Fabricated unsourced numbers → source-or-label discipline; ranges for unknowns.
- Linear edge assumed at 45–76 contracts → flagged unmodeled slippage risk, NOT
  silently ignored. At that size slippage/liquidity are first-order.
- Assuming N accounts when official cap is lower → that number sourced FIRST,
  bounds the whole multiply.

## Multi-account scaling — the legitimate path (operator question 2026-06-06)

Operator: "is there a logical way around the payout caps / how do people run
multiple accounts? it openly says 5x accounts."

ANSWER (logic, not fabricated specifics — sourcing pass must verify the numbers):
The per-payout cap and the multi-account model solve DIFFERENT problems. Five
accounts is not a contradiction — it is the SANCTIONED way to scale past a single
account's withdrawal ceiling:
- One account = one capped withdrawal channel (~$cap/payout x frequency).
- N accounts = N independent withdrawal ceilings for the SAME strategy.
- COPY-TRADING is the mechanism (repo already has
  `trading_app/live/copy_order_router.py`): one signal mirrored across N
  accounts so attention/fills aren't split. The edge does NOT dilute — each
  account trades the same signal on its own capital; only COST scales linearly.

Why accounts are "disposable" yet stacking works: the firm SELLS accounts
(subscription + reset fees). An expendable account is the firm's revenue, not
your loss, as long as each funded account's expected withdrawal > its cost. The
multi-account arbitrage = replicate one validated edge across N cheap
independent withdrawal channels.

Legitimate levers (NOT manipulation):
1. Multiple accounts + copy-trade (the advertised 5x multiplier).
2. Consistency payout path: higher cap ($6k vs $5k) at cost of the 40%
   largest-day rule (already in canon).
3. Live-funded graduation: up to 100% of balance after 30 winning days
   (uncaps, but full withdrawal closes the account).

FORBIDDEN (account-farming, ToS violation, voids payouts + bans):
disguising copy-trading, faking account independence, detection evasion. The
model does NOT design any of this — it is not needed; stacking + copy-trade is
the legit path. The "x N accounts" line is the real headline, bounded ONLY by
the verified max-concurrent number + copy-trade permission (both still to source).

## Rollback

Analysis script is read-only + disposable. Only durable change = payout-canon
correction (reversible, gated).

## Prior art / memory links

- `project_150k_beats_100k_unclamped_per_lane_sizing_2026_06_05` — per-lane
  sizing breaks the size-moot null; 150k beats 100k unclamped.
- `project_c11_size_moot_at_1micro_and_sim_live_sizing_seam_2026_06_05` — D-2
  payout canon STALE; D-3 sim↔live sizing seam (must close before any
  multi-contract arming); hardcoded 1-micro.
- Hard gates still open: `max_contracts=1` clamp, D-3 seam, bracket audit
  `9b3fc530`, 150k profile creation (Tier B), edge-at-scale slippage UNMEASURED.
