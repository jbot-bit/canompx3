# C11 Equity-Drawdown Throttle — Implementation Plan (Tier B, capital path)

**Status:** DRAFT — NOT implemented. Requires operator approval + adversarial audit
before any production edit. Validation PASS (OOS + throttle-aware MC); see Evidence.
**Date:** 2026-06-04
**Hypothesis (locked):** `docs/audit/hypotheses/2026-06-04-c11-equity-drawdown-throttle.yaml`
**Validation:** `docs/audit/results/2026-06-04-c11-throttle-validation.md`
**Chosen cell:** `trigger=$800, factor=0.5, recover=$400` (mid of WF-stable band {600,800,1000}).

---

## Purpose

Add a causal equity-drawdown throttle (halve participation when account drawdown-from-peak
≥ $800; restore at ≤ $400) so `topstep_50k_mnq_auto` clears the strict C11 90-day DD gate
($2,038 → $1,459 historical; throttle-aware MC eliminates the residual 0.285% MLL-breach
probability and cuts p95 max-DD by ~$222). The throttle must be enforced **identically** by
the survival gate (`account_survival`) and the live runtime (`SessionOrchestrator`) — one
canonical parameter source, no divergence.

## Source-of-truth chain

1. **Canonical edge:** `orb_outcomes.pnl_r` → `account_survival._load_lane_trade_paths` →
   `_load_profile_daily_scenarios` (DailyScenario per trading day).
2. **Gate metric:** `account_survival._max_observed_rolling_drawdown` (:749), strict gate at
   `evaluate_profile_survival` (:789-798), budget = `dd_limit × STRICT_DD_BUDGET_FRACTION`
   (0.80 × $2,000 = $1,600).
3. **Gate MC:** `account_survival.simulate_survival` (:608-734) — i.i.d. bootstrap of days.
4. **Live DD authority:** `AccountHWMTracker._dd_used()` (account_hwm_tracker.py:527) =
   `max(0, hwm − last_equity)`, polled per-bar in `session_orchestrator` (:1962).
5. **Live sizing path:** `session_orchestrator` ORB sizing (existing `max_orb_size_pts` cap
   at :395 is a *separate* control; throttle is a multiplicative size scale, not a cap).
6. **Throttle constants (NEW canonical source):** to be created — single module both gate
   and live import.

## Canonical throttle parameter source (the parity keystone)

**New module:** `trading_app/throttle_policy.py` (small, no logic dependencies).

```python
# trading_app/throttle_policy.py
from dataclasses import dataclass

@dataclass(frozen=True)
class EquityDrawdownThrottle:
    """Causal equity-drawdown participation throttle. ONE source for gate + live.
    @research-source: docs/audit/hypotheses/2026-06-04-c11-equity-drawdown-throttle.yaml
    @revalidated-for: topstep_50k_mnq_auto (MNQ), OOS WF + throttle-aware MC 2026-06-04
    """
    trigger_dollars: float      # engage when dd_from_peak >= this
    factor: float               # participation scale while engaged (0.5 = half)
    recover_dollars: float      # disengage when dd_from_peak <= this (hysteresis)
    enabled: bool = False       # OFF by default; opt-in per profile

    def scale_for(self, dd_from_peak: float, *, currently_throttled: bool) -> tuple[float, bool]:
        """Return (scale, new_throttled_state). Causal: caller passes dd through t-1."""
        if not self.enabled:
            return 1.0, False
        throttled = currently_throttled
        if not throttled and dd_from_peak >= self.trigger_dollars:
            throttled = True
        elif throttled and dd_from_peak <= self.recover_dollars:
            throttled = False
        return (self.factor if throttled else 1.0), throttled

C11_MNQ_THROTTLE = EquityDrawdownThrottle(
    trigger_dollars=800.0, factor=0.5, recover_dollars=400.0, enabled=False,
)
```

Profile binding: add an optional `throttle: EquityDrawdownThrottle | None` field to the
`AccountProfile` (or a profile→throttle map in `prop_profiles.py`), set for
`topstep_50k_mnq_auto`. Both the gate and live read the throttle off the **resolved
profile**, never a local literal — this is the single source that prevents divergence.

## Gate/live parity mechanism

| Concern | Gate (`account_survival`) | Live (`SessionOrchestrator`) |
|---|---|---|
| Throttle params | `profile.throttle` (resolved) | `profile.throttle` (same resolved object) |
| DD-from-peak input | running cumulative-pnl peak (historical series) and per-MC-path peak | `AccountHWMTracker._dd_used()` (HWM − equity) per-bar |
| Decision function | `throttle.scale_for(...)` | `throttle.scale_for(...)` (identical call) |
| Where applied | scale day pnl/min/max BEFORE `_max_observed_rolling_drawdown` and inside `simulate_survival` per sampled day | scale ORB position size for the next entry while engaged |

**Parity is enforced by:** (1) both sides import `scale_for` from the same module; (2) a new
drift check `check_throttle_gate_live_parity` asserting both the gate path and the live path
reference `throttle_policy.scale_for` and the same profile field (marker + import-graph
assert, mirroring `self-funded-sizing-doctrine` guard style); (3) a parity unit test running
the same dd-from-peak sequence through both call sites and asserting identical scale series.

**OPEN DESIGN DECISION (must resolve before implementation):** DD-basis reconciliation.
The validation/gate uses *cumulative-pnl running peak updated daily*; the live tracker's
`_dd_used()` uses *HWM that for eod_trailing only ratchets at session close*. For a $50k
Topstep eod_trailing account these can differ intraday. Options:
  (a) Throttle on `_dd_used()` live (HWM-from-EOD-close) AND make the gate use the same
      EOD-ratcheting peak (re-validate — DD numbers may shift).
  (b) Throttle on a daily-close cumulative-pnl peak both sides (matches current validation;
      live computes it from session-close equity deltas, not intraday HWM).
  Recommended: (b) — it matches what was validated; (a) requires re-running the whole OOS +
  MC suite on a changed DD basis. This MUST be pinned and (if (a)) re-validated.

## Exact files to change

1. **NEW** `trading_app/throttle_policy.py` — canonical throttle dataclass + `C11_MNQ_THROTTLE`.
2. `trading_app/prop_profiles.py` — bind throttle to `topstep_50k_mnq_auto` (resolved profile field).
3. `trading_app/account_survival.py` — apply `throttle.scale_for` (i) before
   `_max_observed_rolling_drawdown`, (ii) inside `simulate_survival`'s per-day loop
   (scale pnl/min/max; maintain per-path equity peak + throttled state). Port the
   *validated, parity-proven* logic from `C:\Users\joshd\c11_matrix\throttle_mc.py`.
4. `trading_app/live/session_orchestrator.py` — read `profile.throttle`; compute
   dd-from-peak via the chosen basis; apply `scale_for` to next-entry ORB size.
5. `pipeline/check_drift.py` — `check_throttle_gate_live_parity` (new guard).
6. **Tests** (see below).

No dashboard/UI changes. No config flips beyond the per-profile `enabled` (operator-gated).

## Tests

- `tests/test_throttle_policy.py` — `scale_for` truth table (engage/recover/hysteresis,
  disabled→1.0), frozen-dataclass immutability.
- `tests/test_account_survival_throttle.py` — (a) baseline unchanged when `enabled=False`
  (regression guard); (b) with throttle, max90dDD = $1,459 ± rounding; (c) **no-lookahead**:
  permuting day t's pnl does not change day t's scale.
- `tests/test_throttle_gate_live_parity.py` — same dd sequence → identical scale series from
  gate and live call sites.
- Throttle-aware MC reproduction (port `throttle_mc.py` parity assertion into a test):
  factor=1.0 reproduces canonical `simulate_survival` exactly; throttle never regresses
  survival/breach channels.
- `pipeline/check_drift.py` full pass (incl. new parity guard).
- `pytest tests/ -k "survival or session or risk or throttle or hwm"`.
- Live smoke / dry-run proving the orchestrator reads `profile.throttle` from the same
  source (assert object identity or import path in a session-construction test).

## Rollback plan

- The throttle ships `enabled=False` by default. **Rollback = flip `enabled` to False** on
  the profile (one-line, no code revert) — gate and live both revert to current behavior
  instantly because `scale_for` returns `(1.0, False)` when disabled.
- Full revert: `git revert` the implementation commit (additive module + guarded call sites;
  no canonical signature changes, so revert is clean).
- State safety: throttle adds no new persisted state (reuses tracker DD); nothing to migrate
  or clean up on rollback.

## Verification gates before this is "done" (all required)

1. claim hygiene on result + this plan.
2. `account_survival` baseline (enabled=False) byte-identical to pre-change.
3. `account_survival` throttle path → strict gate PASS ($1,459 ≤ $1,600).
4. throttle-aware MC: parity PASS + no survival regression.
5. gate/live parity test PASS.
6. orchestrator dry-run/smoke proves same-source throttle read.
7. `pytest` risk/session/account/throttle/hwm green.
8. no-lookahead test green.
9. `check_drift.py` full pass.
10. adversarial audit (capital path) before arming.

## Remaining risks (carry-forward)

- **DD-basis reconciliation (above)** is unresolved and is the highest-risk design fork.
  Until pinned, do NOT implement the live side.
- Live throttle applies to *next* entry sizing; intra-day already-open positions are not
  resized (matches conservative validation — throttle is a participation scale, not a
  forced-flatten). Confirm this matches operator intent.
- The throttle was validated on `topstep_50k_mnq_auto` only. Other profiles must NOT inherit
  it implicitly (`enabled=False` default protects this).
- Self-funded profiles: per `self-funded-sizing-doctrine.md`, throttle is a *risk* control
  (legitimate for self-funded) but its prop-derived $800 trigger is account-size-specific —
  do not copy to a self-funded tier without re-deriving on that account's risk basis.

## Live-trading status

**Still BLOCKED for live.** This plan does not arm anything. Implementation is gated on:
operator approval of this plan → DD-basis decision → implementation → all 10 verification
gates → adversarial audit. The throttle ships disabled; enabling it for live is a separate,
explicit operator decision after the above.
