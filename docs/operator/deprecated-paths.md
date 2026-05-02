# Deprecated Operator Paths

These references remain in the repo for history or generic broker support, but
they are not approved implementation guidance for the current TopstepX-first
operator architecture.

## Active/current references annotated in place

- `docs/handoffs/2026-05-02-topstep-operator-v3-handover.md`
  - `DEPRECATED – blocked by Topstep policy as of 2026-05-02`
  - reason: same-account `TopstepX + Quantower shell` lifecycle is dead
- `scripts/run_live_session.py`
  - `DEPRECATED – blocked by Topstep policy as of 2026-05-02`
  - reason: generic multi-account copy-routing comments could be misread as
    approval for Topstep Live Funded copy use
- `trading_app/live/session_orchestrator.py`
  - `DEPRECATED – blocked by Topstep policy as of 2026-05-02`
  - reason: generic `CopyOrderRouter` comments and dashboard metadata could be
    misread as approval for Topstep Live Funded copy use
- `trading_app/prop_profiles.py`
  - `DEPRECATED – blocked by Topstep policy as of 2026-05-02`
  - reason: Topstep `auto_trading="full"` and legacy MNQ automation planning
    must not be read as approval for Live Funded ProjectX-API automation

## Boundary

- Generic copy-routing infrastructure is **not** globally deprecated.
- ProjectX execution infrastructure is **not** globally deprecated.
- The deprecation applies specifically to:
  - same-account TopstepX external-platform shells
  - Topstep Live Funded copy-trading interpretation
  - Topstep Live Funded ProjectX-API automation interpretation

## Exact references

- `docs/handoffs/2026-05-02-topstep-operator-v3-handover.md:66`
- `scripts/run_live_session.py:567`
- `trading_app/live/session_orchestrator.py:543`
- `trading_app/live/session_orchestrator.py:1220`
- `trading_app/prop_profiles.py:258`
- `trading_app/prop_profiles.py:484`
