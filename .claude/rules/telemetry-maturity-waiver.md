# Telemetry-Maturity Gate — WAIVED for Express-Funded (HARD DOCTRINE)

**Load-policy:** auto-injected when editing `scripts/run_live_session.py`,
`trading_app/live/telemetry_maturity.py`, or `trading_app/prop_profiles.py`.
Read on demand when reasoning about live-launch readiness gates.

**Authority:** operator decision, 2026-06-01 — *"REMOVE telemetry from Topstep
accounts."* Permanent project invariant. The 30-day signal-maturity floor must
NEVER block a funded (Express-Funded) live launch.

---

## The rule

The telemetry-maturity preflight gate (`_check_telemetry_maturity` in
`scripts/run_live_session.py`, gate [10] in the live preflight chain) is
**WAIVED to a clean OK** for any profile with `is_express_funded=True`
(Topstep XFA, Tradeify, Bulenox — every funded-account wrapper).

Verdict matrix (post-waiver):

| Mode / profile class | Verdict |
|---|---|
| `--signal-only` | OK (this path accumulates the count) |
| `--demo` | WARN (advisory; no capital, but not a deliberate waiver) |
| `--live` + Express-Funded (`is_express_funded=True`) | **OK — WAIVED** |
| `--live` + real-capital (`is_express_funded=False`, unknown, or no profile) | **FAIL** (floor still enforced) |

## Why

A funded-account wrapper insulates **real personal capital**: the prop firm's
sim/funded account absorbs drawdown, not the operator's own money. The
30-day telemetry floor exists to protect real capital from launching live on an
unvalidated signal stream — that risk does not apply to a funded wrapper. The
operator has decided the funded path should launch on demand, gated by broker
auth + portfolio + contract + bracket/fill probes (all still enforced), not by
a signal-day count.

## What this does NOT relax

- **Real-capital self-funded profiles keep the gate.** `is_express_funded=False`
  (e.g. `self_funded_tradovate`) still FAILs below the floor. This is consistent
  with `self-funded-sizing-doctrine.md` — real personal capital gets the
  conservative guardrails; the funded wrapper does not.
- Every other preflight gate (auth, portfolio, daily-features freshness,
  contract resolution, notifications/bracket/fill-poller probes, journal health,
  drift gate, strict live-readiness report, project-pulse blocker, copy-trading
  resolution, shadow-copy loss protection) is untouched.

## Enforcement

The waiver lives in the `_check_telemetry_maturity` branch logic, keyed on the
canonical `AccountProfile.is_express_funded` flag (`trading_app/prop_profiles.py`)
— no hardcoded profile-id list, so new funded profiles inherit the waiver
automatically and new real-capital profiles inherit the FAIL automatically.

## Related

- `scripts/run_live_session.py` § `_check_telemetry_maturity` — the implementation.
- `.claude/rules/self-funded-sizing-doctrine.md` — sibling doctrine; real
  personal capital is risk-first and keeps conservative guardrails.
- `trading_app/prop_profiles.py` § `is_express_funded` — the canonical classifier.
- `docs/governance/decisions/2026-06-01-telemetry-waiver-express-funded.md` —
  the dated decision record.
