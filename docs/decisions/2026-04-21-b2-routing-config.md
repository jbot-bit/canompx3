# B2 — Order Routing Config — 2026-04-21

**Worktree:** `deploy/live-trading-buildout-v1`
**Stage:** Workstream B sub-phase 5.1b (Task #17)
**Status:** **DRAFT ROUTING TABLE — consumes Stage 1 prop-rules; paper-only until credentials + user-auth clear (workstream A abort still live).**
**Blast radius:** documentation only; no code edits in this sub-phase. Routing constants will be encoded in a follow-on config file in a later sub-phase that's gated on workstream A clearing.
**Authority:** `resources/prop-firm-official-rules.md` (refreshed 2026-04-21), `trading_app/live/broker_factory.py`, `trading_app/live/{rithmic,projectx,tradovate}/`.

---

## Purpose

Define the instrument → venue mapping table the allocator will use to route orders per firm and per account type. Constraints come from Stage 1 prop-rules (workstream E) and from the abort-trigger findings of workstream A.

This doc is a **pre-wire decision**, NOT executable code. The actual routing table will be encoded in `trading_app/live/broker_factory.py` or a sibling config when workstream A unblocks (credential provisioning + user `--live` authorization).

---

## Hard constraints from Stage 1

1. **TopStep Live Funded Account (LFA): ProjectX automation PROHIBITED** (source `https://help.topstep.com/en/articles/10657969-live-funded-account-parameters`, last-updated this week). LFA auto MUST route through **Rithmic** or **Tradovate**. The repo's `trading_app/live/projectx/` integration cannot be used for LFA orders, under any circumstances.
2. **TopStep hedging enforcement staged:** warning → same-day block → permanent closure. Cross-account same-underlying opposite positions prohibited. Applies across Combine / XFA / LFA. Router MUST track "same trader" across slots and veto opposite positions pre-submission.
3. **TopStep copier now ALLOWED on LFA** (reversal from stale 2026-03-16 record). TopstepX, Tradovate, Rithmic, Quantower historically supported.
4. **Bulenox Funded requires 3 successful Master payouts** before promotion. Router starts with Master account slots for Bulenox. Bulenox auto/copier/news rules UNVERIFIED (first-party pages silent; Terms-of-Use PDF binary; WebFetch extraction failed); treat as "assume strict" until ops gets written confirmation.
5. **MFFU fair-play "coordinating identical/opposite strategies across separate accounts"** is AMBIGUOUS vs "copy trading allowed across all account types". Router MUST NOT put the same strategy on MFFU **and** another firm until user gets written MFFU clarification — flagged as abort trigger for Stage 5.1e account binding, NOT resolved here.
6. **MFFU News Trading Policy:** no open positions or orders 2 min before/after data release; Tier 1 events (FOMC, NFP, CPI + EIA/Ag conditional) prohibited for Rapid Sim Funded + Pro Sim Funded. Router MUST enforce a news-embargo gate at order-submission for MFFU accounts.
7. **MFFU Hedging:** E-mini NQ + micro NQ on same account = hedging (same NQ index). Router MUST block opposite positions on same-underlying per MFFU account.

## Code-level constraints from broker base

`trading_app/live/broker_base.py:257` — broker-base hook for F-1 enforcement. Each broker implementation can override. Per `_apply_broker_reality_check` in `session_orchestrator.py`: F-1 auto-disables for TC (Trading Combine) accounts; stays enforced for XFA and LFA profiles.

---

## Routing table — instrument × account type × venue

| Firm | Account type | Instrument family | Allowed venues | Disallowed venues | Notes |
|---|---|---|---|---|---|
| TopStep | Trading Combine (TC) | MNQ / MES / MGC | TopstepX, Tradovate, Rithmic, Quantower | — | F-1 auto-disables for TC per `306d16a0` |
| TopStep | Express Funded (XFA) | MNQ / MES / MGC | TopstepX, Tradovate, Rithmic, Quantower | — | F-1 auto-enforces ladder per XFA size |
| TopStep | Live Funded (LFA) | MNQ / MES / MGC | **Rithmic, Tradovate** | **ProjectX (automation prohibited)**, TopstepX and Quantower presumed allowed but not verified 2026-04-21 | Stage 1 critical finding |
| Bulenox | Master (qualification) | MNQ / MES / MGC | **UNVERIFIED — no first-party source re-verified 2026-04-21** | **UNVERIFIED — all others** | Fetched help pages are silent on auto/copier/news/prohibited; Terms-of-Use PDF fetch failed (binary). NO training-memory fallback per directive. User-owned: obtain written Bulenox support confirmation before routing selection takes effect. |
| Bulenox | Funded (Pro) | MNQ / MES / MGC | **UNVERIFIED** | **UNVERIFIED** | Same as Master, plus requires 3 Master payouts first (first-party confirmed). |
| MFFU | Sim Funded (Rapid / Flex / Pro) | MNQ / MES / MGC | Tradesyncer, Tradovate, Rithmic, "external copier solutions" | — | **CONDITIONAL on Fair Play stack**: (a) NOT exploiting favourable simulated fills, (b) NO HFT, (c) CME guidelines compliance. Also: news-embargo gate (2 min before/after data release; Tier 1 events prohibited for Rapid Sim Funded + Pro Sim Funded) + E-mini/micro-NQ hedging block on same account. |
| Apex | (DEAD for ORB) | — | N/A | All | Not re-fetched 2026-04-21; memory flags dead |
| Tradeify | (DEAD for ORB) | — | N/A | All | Not re-fetched 2026-04-21; memory flags dead |

### Default primary venue per firm

| Firm | Default primary venue | Rationale |
|---|---|---|
| TopStep TC / XFA | **Rithmic** | Already supported per stale copier article + widely used in repo's `trading_app/live/rithmic/` integration |
| TopStep LFA | **Rithmic** | ProjectX disallowed; Rithmic > Tradovate based on repo integration surface |
| Bulenox | **UNVERIFIED** | WebFetch extraction FAILED on Terms-of-Use PDF (binary); help pages silent on auto / copier / news / prohibited conduct. Per directive § guardrails, no training-memory fallback is allowed. Bulenox default is **NOT SELECTED** until ops obtains written Bulenox support confirmation of (a) automation permitted and (b) Rithmic-compatibility. This constitutes a user-owned decision, NOT a preset default. |
| MFFU | **Rithmic** (CONDITIONAL on Fair Play stack) | Copier broadly allowed + repo integration surface. Condition stack: NOT exploiting favourable sim fills + NO HFT + CME-compliant per Fair Play article. Enforcement-critical — violation -> account termination + profit confiscation. |

### Hedging / same-instrument guards (all firms)

Router MUST:
1. Track open positions per trader across all firm accounts.
2. Reject any order that would produce opposite-sign position on same underlying on a different account (TopStep rule; hedging prohibited).
3. Reject any order that would produce opposite-sign position on same underlying on SAME MFFU account (MFFU rule).
4. Reject any order within 2 min before or after a TIER-1 news event if account is MFFU Rapid Sim Funded or MFFU Pro Sim Funded.
5. Emit a hedging-near-breach warning when opposite positions are > 80% confidence of arising (log-only, pre-submission hint).

### TopStep same-trader-tracking

Per Stage 1 extract (`https://help.topstep.com/en/articles/13747047-understanding-hedging` L3492+):
> "Violations are tracked at the Trader level. Violations apply across all accounts involved in hedging."

Router MUST aggregate positions across TopStep TC / XFA / LFA slots for the same trader ID before submitting any order.

---

## N̂ multi-firm correlation gate stub (Stage 5.3 / Task #21 dependency)

Before the router allows a lane to fire on **multiple firms simultaneously**, it MUST compute ρ̂ per Bailey-LdP 2014 Appendix A.3 Eq. 9 across firm-lane pairs:

N̂ = N / (1 + (N − 1) · ρ̄)

If N̂ < N̂ assumed by allocator (currently D=2.41 for the 6-lane MNQ portfolio per `six_lane_portfolio_near_independent.md`), the multi-firm fan-out is refused for that lane.

This gate is NOT live in B2; it's a design marker for Task #21.

---

## MFFU cross-firm coordination ambiguity — FLAGGED

Directive § Phase 5.1e abort trigger: if MFFU fair-play "coordinating identical/opposite strategies across separate accounts" applies cross-firm, the multi-firm scaling plan partially breaks. Router behaviour until clarified:

- **SAFE default:** MFFU gets its own isolated copy of the strategy lane. No cross-firm mirroring to/from MFFU accounts.
- **Consequence:** MFFU adds a parallel but uncorrelated deployment channel; does NOT multiply capital on the same signal as TopStep/Bulenox. Less scaling benefit than memory's multi-firm math assumed.
- **Post-clarification:** if MFFU support confirms in writing that same-edge cross-firm is fine, relax to mirrored fan-out.

---

## What this doc does NOT do

- Does NOT wire Rithmic / Tradovate credentials (workstream A abort active).
- Does NOT flip any profile to `--live`.
- Does NOT edit `trading_app/live/broker_factory.py` or routing code. The table above is a SPEC for the code change that will happen when credentials + `--live` auth clear.
- Does NOT contact MFFU support, Bulenox support, or TopStep support. User-owned ops tasks.

---

## Stage 1 rule-citation index (for auditor cross-check)

Every claim above traces to:

| Claim | Stage 1 source |
|---|---|
| LFA ProjectX ban | `https://help.topstep.com/en/articles/10657969-live-funded-account-parameters` |
| LFA copier allowed | same |
| TopStep hedging staged enforcement | `https://help.topstep.com/en/articles/13747047-understanding-hedging` |
| MFFU news embargo + Tier 1 | `https://help.myfundedfutures.com/en/articles/8230009-news-trading-policy` |
| MFFU fair-play coordination ambiguity | `https://help.myfundedfutures.com/en/articles/8444599-fair-play-and-prohibited-trading-practices` + `https://help.myfundedfutures.com/en/articles/10771500-copy-trading-at-myfundedfutures` |
| MFFU hedging (micro + mini NQ) | `https://help.myfundedfutures.com/en/articles/12011241-hedging-what-you-should-know` |
| Bulenox Funded 3-payout gate | `https://bulenox.com/help/funded-account/` |
| Bulenox auto/copier UNVERIFIED | `https://bulenox.com/help/` + `https://bulenox.com/wa-data/public/site/data/bulenox.com/Terms_of_Use.pdf` (fetch failed) |

All sources are indexed at `resources/prop-firm-official-rules.md` § Provenance index.

---

## Self-review

- [x] Every routing decision cites a Stage 1 rule.
- [x] LFA ProjectX ban is called out as the single most consequential routing constraint.
- [x] UNVERIFIED rules (Bulenox auto/copier/news) default to conservative / opt-out, not permissive.
- [x] No credentials referenced. No code edits.
- [x] Nothing in this doc attempts to resolve the MFFU cross-firm ambiguity on its own — it's flagged, with a safe default, for user + MFFU support to resolve.

**B2 = DRAFT ROUTING SPEC CAPTURED.** Wire implementation is a follow-up task gated on workstream A credentials + `--live` authorization.
