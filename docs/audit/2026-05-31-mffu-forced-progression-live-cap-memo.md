# MFFU Forced Progression + Live Account Cap — Design Memo (Layer C)

**Status (updated 2026-05-31, Option A landed):** Layer C **DATA + drift guard
DONE**. The forced-transition STATE MACHINE / payout-count ledger (Option B) is
still DESIGN ONLY, deferred behind the adversarial-audit gate. Created 2026-05-31
alongside Stage 2 (specs+payout-policy data).

## ✅ Resolution log (2026-05-31 — Option A)

The operator chose **Option A: data-only + drift check** (no state machine, no
ledger, no live-session logic, no account-routing). Findings on re-grounding:

- **The "what the codebase lacks" list below was 2/3 STALE.** Stage 2 already
  encoded `max_live_accounts: 1` AND `forced_live_after_sim_payouts: 5` as DATA
  in `prop_profiles.py` `firm_specific_rules` (Builder, line ~351; Flex covered).
  `PropFirmSpec.firm_specific_rules` (field at line 60) is the carrier — no new
  class field was needed. The only genuine remaining gap was enforcement LOGIC,
  which Option A deliberately does NOT build.
- **The anecdote is RESOLVED against verbatim source — no avoidance path exists.**
  `docs/research-input/mffu/mffu_live_accounts_faq.md:237` (verbatim): "rejecting
  the move to a Live Funded Account is not possible once you are selected." The
  friend's "withdrew many times" is explained by `:89` — Live accounts allow
  **daily** payouts with no cap, i.e. repeated withdrawals are normal POST-promotion
  behavior, exactly as hypothesized. There is nothing to model as "avoidance."
- **`max_live_accounts=1` is the DEFAULT / entry cap, not an absolute structural
  invariant.** Verbatim `mffu_live_accounts_faq.md:243`: "Multiple live accounts
  can be discussed with your risk manager, but are reserved solely for
  high-performing traders that have a long track record." So `1` is the cap a new
  live trader gets; discretionary exceptions exist. The data field encodes the
  default; the discretionary path is not modeled (no consumer).
- **Drift guard shipped:** `check_live_funded_firms_declare_max_live_accounts`
  (`pipeline/check_drift.py`) — every firm backing a `live_funded` PayoutPolicy
  must declare `firm_specific_rules['max_live_accounts']`. Surfaced + closed a real
  gap: the **Topstep** spec had the cap only as prose ("5 Express + 1 Live"); it is
  now structured data, verbatim from `topstep_xfa_parameters.txt:228,235` ("Only 1
  Live Funded Account is permitted"). Tests: `tests/test_pipeline/test_check_drift_live_account_cap.py`.

**Everything below is the ORIGINAL memo, retained for lineage. The "lacks" list and
the "⚠️ UNVERIFIED anecdote" section are SUPERSEDED by the resolution log above.**

---

**Status:** DESIGN ONLY. No code. Gated for separate adversarial-audit sign-off
(capital-path logic). Created 2026-05-31 alongside Stage 2 (specs+payout-policy data).

**Authority:** operator flagged that prop plans "force you to level through to
live funded (which might only be 1 account cap max)." Layers A (specs/tiers) and
B (payout policies) encoded the *data* for this; the *transition logic* and the
*live-account cap* have no field/logic in the codebase and are this memo's scope.

---

## The problem (verbatim-grounded)

MFFU plans force a staged path: **Evaluation → Sim Funded → Live Funded**, and
the live stage caps accounts hard.

- **Builder** (article 14290805, scraped 2026-05-31):
  - @verbatim "After your 5th approved sim payout, you are eligible for promotion
    to a live funded account."
  - @verbatim "Max Live Accounts: 1"
  - @verbatim "Only one Sim Funded account may be active per user at any time."
- **Flex** (Flex 25k/50k articles): 5 winning days → payout; 5 sim payouts /
  $100K total sim cap; same staged path.
- **TopStep** (operator note, NOT yet re-verified this session): also forces a
  progression. The repo already models `topstep_live_funded` as a separate
  `PayoutPolicy` stage and `topstep_scaling_plan.py` for XFA scaling, but there
  is **no explicit forced-transition trigger** encoded.

### What the codebase lacks today

1. **No `max_live_accounts` field** on `AccountProfile` or `PropFirmSpec`.
   `AccountProfile.copies` is sim-account count; nothing collapses it to 1 on
   live promotion.
2. **No stage-transition trigger.** `PayoutPolicy.stage` is a static label
   (`sim_funded` / `live_funded`); nothing fires "5th payout → you are now
   forced into the live stage with its (different) rules."
3. **No payout-count ledger.** The forced-live trigger is payout-count based;
   the survival sim has no notion of cumulative approved payouts.

---

## ⚠️ UNVERIFIED operator anecdote — must resolve before encoding any "avoidance"

Operator: a friend "has withdrawn many times from an account" — suggesting the
forced-live progression **may be avoidable** or not uniformly enforced.

- Treat as an **UNOFFICIAL signal**, not fact (per `.claude/rules/targeted-grounding.md`
  official-vs-unofficial separation).
- The verbatim Builder source says "After your 5th approved sim payout, you are
  **eligible** for promotion" and "Max Sim Payouts: 5 Payouts" — wording is
  *eligible*, and 5 is a *cap on sim payouts*, which could mean: after 5 sim
  payouts you can no longer take sim payouts (must go live to keep withdrawing),
  i.e. it is a soft forcing function, not an automatic conversion. The friend's
  "many withdrawals from one account" may have been on a **Live** account (daily
  payouts, no cap) or a **non-MFFU firm**.
- **ACTION before any avoidance modeling:** re-read the exact MFFU sim-payout-cap
  + promotion wording, and identify which firm/stage the friend's account was on.
  Do NOT encode an "avoid forced live" path on an anecdote.

---

## Proposed design (for sign-off — not yet built)

1. Add `max_live_accounts: int | None = None` to `PropFirmSpec` (optional, like
   `firm_specific_rules`; None = unmodeled). Populate from verbatim
   (`mffu_builder`/`mffu_flex` = 1).
2. Add a `stage_transition` mapping to `firm_specific_rules` (already partially
   present: `forced_live_after_sim_payouts`). Keep it data; the *enforcement*
   belongs in the survival/account-state layer, not the spec.
3. Decide WHERE the trigger fires: `account_survival.py` is read-only Monte Carlo
   over daily scenarios — it has no payout ledger and arguably should not gain
   one. The forced-live transition is a **live-session / account-state** concern
   (`scripts/run_live_session.py` + account HWM tracker), so this likely lands
   there, NOT in the sim. Confirm before building.
4. Drift check: assert every firm with a `live_funded` payout policy declares
   `max_live_accounts`.

## Why deferred (not bundled into Stage 2)

- Adds a field to a frozen dataclass AND new logic on a capital path → triggers
  the adversarial-audit gate (`.claude/rules/institutional-rigor.md` § 2).
- The anecdote above must be resolved first, or we risk encoding the wrong
  forcing semantics.
- Stage 2 (data) is reversible and consumer-free (no active mffu profile); C is
  not — it changes how live promotion/sizing is reasoned about.

## Related

- `docs/runtime/stages/2026-05-31-mffu-builder-prop-rules-stage2.md` — the A+B stage.
- `trading_app/prop_firm_policies.py` — `mffu_builder_sim` / `mffu_flex_sim` (B).
- `trading_app/topstep_scaling_plan.py` — existing TopStep scaling (compare when verifying the TopStep forced-progression claim).
