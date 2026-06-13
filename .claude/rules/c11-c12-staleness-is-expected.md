---
paths:
  - "trading_app/derived_state.py"
  - "trading_app/account_survival.py"
  - "trading_app/lifecycle_state.py"
  - "scripts/tools/project_pulse.py"
  - "scripts/tools/refresh_control_state.py"
---

# C11/C12 Control-State Staleness Is EXPECTED — Regen, Don't Debug (HARD DOCTRINE)

**Load-policy:** auto-injected (via the `paths:` frontmatter above) when editing the
C11 (survival) / C12 (SR-monitor) control-state machinery. Read on demand when a
fingerprint/identity mismatch surfaces on a live control surface.

**Authority:** 2026-06-13 — codified after a fresh terminal burned a debug→memory-recall
cycle on a bare `BLOCKED: Criterion 11 state db identity mismatch` message that *read*
like a bug. Origin lesson:
`memory/feedback_c11_c12_fingerprint_mismatch_is_expected_fail_closed_regen_dont_debug_2026_06_13.md`.

---

## The rule

A C11/C12 control-state **fingerprint/identity mismatch is EXPECTED fail-closed
staleness, NOT a bug.** The fingerprint is *designed* to invalidate whenever a
canonical input drifts:

- **DB rebuild** (`db identity mismatch`) — gold.db regenerated/backfilled.
- **Lane narrow / profile change** (`lane_ids mismatch`, `profile mismatch`,
  `profile fingerprint mismatch`) — deployed lanes or profile definition changed.
- **Code change** (`code fingerprint mismatch`) — a `_criterion11_code_paths` /
  C12-fingerprint member was edited (e.g. `derived_state.py`, `account_survival.py`,
  `sr_monitor.py`).
- **Age** (`stale state: Nd old > Md`) — the freshness window elapsed.
- **No state yet** (`missing`) — the report has never been generated for this profile.

**Remedy:** regenerate, do NOT debug —

```bash
python scripts/tools/refresh_control_state.py --profile <profile_id>
```

`refresh_control_state` regenerates BOTH C11 (`evaluate_profile_survival`) and C12
(`run_monitor`), is operator-GO-gated, and re-reads after writing — so it **fails
closed if the state is still invalid** after regen.

## The one exception — DEFECT class

Reasons in the DEFECT class are NOT routine staleness: `legacy state: missing …`,
`wrong state_type: …`, `wrong schema_version: …`, `invalid canonical_inputs`,
`invalid freshness[ metadata]`, `unreadable: …`, `invalid state payload`,
`legacy state: missing versioned envelope`, `invalid state envelope`, and any
unrecognized/`None` reason.

For these the safe flow is **regen FIRST** (refresh_control_state runs anyway and
fails closed on a persistent defect) — **but if it STILL blocks after a clean regen,
it is a real bug: investigate.** This is the discriminating nuance: blanket-stamping
every mismatch "EXPECTED regen" would mask a genuine schema/corruption defect as
routine staleness (a false-PASS in the diagnostic sense).

## Self-classifying message (the structural fix)

The operator/terminal should never need this doctrine to decode the message — the
message classifies itself. `trading_app.derived_state.classify_state_reason(reason)`
returns `(klass, guidance)` where `klass ∈ {EXPECTED_STALE, DEFECT}` and `guidance` is
the human-readable remedy. It is the **single source** consumed by every display
surface (C11 gate messages in `account_survival.py`; the C11 survival + C12 branches in
`project_pulse.py`) — one classifier, both criteria, zero inline-copy parity risk.

The `klass` is **advisory to the human only** — nothing gates the regen tool on it
(`refresh_control_state._needs_refresh` regenerates on ANY not-valid state regardless
of class). The classifier informs; it never decides whether to regen.

## What this does NOT relax

- **Fail-closed stays.** A stale/invalid gate genuinely DOES block live until regen —
  `category`/`severity` in the pulse remain `broken`/`high`. Only the message TEXT
  self-classifies.
- **DEFECT still warrants investigation if it survives a regen.**

## Related

- `memory/feedback_c11_c12_fingerprint_mismatch_is_expected_fail_closed_regen_dont_debug_2026_06_13.md`
  — the origin lesson.
- `trading_app/derived_state.py` § `classify_state_reason` — the canonical classifier.
- `trading_app/account_survival.py` § `check_survival_report_gate` — C11 gate messages.
- `scripts/tools/refresh_control_state.py` — the regen entrypoint (C11 + C12).
- `.claude/rules/branch-flip-protection.md` — companion fail-closed-staleness pattern.
