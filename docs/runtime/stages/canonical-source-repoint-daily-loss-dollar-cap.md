task: Repair dangling @canonical-source pointers — peer sweep (5768c882) deleted docs/runtime/stages/2026-05-26-daily-loss-dollar-cap-wiring.md while prop_profiles.py:116 and risk_manager.py:30 still cite it, breaking drift Check 114 (red on origin/main CI). Relocate the canonical content to docs/specs/daily_loss_dollar_cap.md and re-point both annotations there.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/risk_manager.py
  - docs/specs/daily_loss_dollar_cap.md

## Blast Radius
- docs/specs/daily_loss_dollar_cap.md — NEW canonical home for the daily-loss dollar cap, recovered verbatim from the swept stage file (git show be4400d8). Proper durable location per the stage-gate canonical anti-pattern rule (canonical content must NOT live in docs/runtime/stages/). Zero code coupling — pure doc.
- trading_app/prop_profiles.py:116 — COMMENT-ONLY change: @canonical-source pointer retargeted from the swept stage file to the new spec. No logic, no field, no behavior change. `daily_loss_dollars` field untouched.
- trading_app/risk_manager.py:30 — COMMENT-ONLY change: same pointer retarget. `max_daily_loss_dollars` field untouched.
- Reads: none. Writes: none. No DB, no schema, no live arm, no logic.
- Root cause: Source-of-Truth Chain Rule — the canonical source was deleted, two downstream annotations dangled. Fix re-establishes the chain by relocating the source, not by patching downstream.

## Acceptance
- drift Check 114 (@canonical-source annotations point to existing files) PASSES — both pointers resolve.
- CI=true drift run: the 2 violations drop to 0 (Check 114 was the sole real failure; confirm no others).
- No behavior change: `daily_loss_dollars` / `max_daily_loss_dollars` field defaults and semantics identical (comment-only edits).
- Spec content faithfully preserves the swept stage's grounding (Carver Table 20, 2026 Monte Carlo calibration, audit history).
