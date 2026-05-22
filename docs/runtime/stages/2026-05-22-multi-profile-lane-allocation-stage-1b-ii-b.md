---
task: Stage 1b-ii.b — migrate trading_app/live/session_orchestrator.py allocator-block gate to resolve_allocation_json (live-broker arming path; HIGH severity)
mode: CLOSED
closed_date: 2026-05-22
closed_note: |
  Acceptance: zero `lane_allocation.json` literals in session_orchestrator.py;
  allowlist shrunk 1→0 trading_app entries; 155 PASSED drift (844 pre-existing
  strict-IS carry-over orthogonal); 120 targeted tests pass; evidence-auditor
  surfaced profile_id-mismatch operator-confusion finding which was fixed in
  the same patch (explicit mismatch error vs generic "requires allocation
  file"). Parent stage 1b sub-stage progress checklist ticked.
original_mode: IMPLEMENTATION
slug: 2026-05-22-multi-profile-lane-allocation-stage-1b-ii-b
parent_stage: docs/runtime/stages/2026-05-21-multi-profile-lane-allocation-stage-1b.md
parent_commit: 9331c194
doctrine_anchors:
  - .claude/rules/institutional-rigor.md § 4 (delegate to canonical sources, never re-encode)
  - .claude/rules/adversarial-audit-gate.md (live-broker path requires evidence-auditor on diff before merge)
  - docs/specs/lane_allocation_schema.md § 4 (resolver contract)
scope_lock:
  - trading_app/live/session_orchestrator.py
  - pipeline/check_drift.py
---

## Blast Radius

- `trading_app/live/session_orchestrator.py` — ONE call-site (lines ~420-453, the "Allocator block gate" block). Replaces:
  - direct construction of `_alloc_path = Path(__file__).resolve().parents[2] / "docs" / "runtime" / "lane_allocation.json"`
  - existence probe (`_alloc_path.exists()`)
  - corruption probe (`json.loads(_alloc_path.read_text())`)
  - call to `load_paused_strategy_ids(_alloc_path)` with positional legacy path
  - log strings that name the legacy path
  with a single call to `resolve_allocation_json(profile_id)` followed by the equivalent fail-closed checks against `AllocationRead.source`.
- `profile_id` is already locally available at line 360 (extracted from `portfolio.name.removeprefix("profile_")`); for non-`profile_*` accounts it's None.
- Behavior preservation contract (FAIL-CLOSED must be byte-equivalent for `profile_*` accounts):
  - `profile_*` + no allocation found (`source == "missing"`) → `RuntimeError("FAIL-CLOSED: profile account requires lane_allocation.json ...")`. Today reads the file's path; after migration the message must include the resolver's `result.path` if available, else the expected legacy path (for the message-string operator-runbook stability).
  - `profile_*` + corrupt JSON → previously raised `RuntimeError("FAIL-CLOSED: profile account requires parseable lane_allocation.json: <error>")` from the probe-parse. The resolver silently returns `source="missing"` on `JSONDecodeError`/`OSError`. To preserve the corruption-vs-missing distinction, do a residual corruption probe ONLY when `source == "missing"` AND `_is_profile` (re-test the legacy path, since that's where the operator-runbook expects to see the file).
  - non-`profile_*` (paper/signal) → fail-open: empty `_regime_paused` set is acceptable. Log strings preserved.
- `load_paused_strategy_ids(allocation_path=...)` continues to be the consumer of the resolved data path. Calling it with `profile_id=profile_id` (when non-None) uses the resolver internally; calling with just `allocation_path=...` keeps legacy semantics. The chosen approach: call with `profile_id=profile_id` (or `None`) so the resolver semantics flow through.
- `pipeline/check_drift.py` — single edit: remove `Path("trading_app/live/session_orchestrator.py")` from `_LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST`. The grep-gate (Check #170-ish) then PASSES because the literal is gone; the dead-allowlist-entry sub-check would FAIL if I left the entry in after removing the literal.
- Reads: `gold.db` not touched. Writes: none. No DDL.
- Tests touched: zero net new tests; existing live-orchestrator test suite must still pass. (Per parent stage, a fixture sweep applies — but the orchestrator's tests don't write `lane_allocation.json` directly; they use shared fixtures already exercised by Stages 1b-ii.a-1 and 1b-ii.a-2.)
- Drift: 844 pre-existing strict-IS carry-over violations remain on this branch (orthogonal, per HANDOFF). PASS count must stay at 155 (i.e., the grep-gate continues to pass after the allowlist shrinks 1→0).
- Adversarial-audit-gate: HIGH severity (live-broker arming + kill-switch path). Per `.claude/rules/adversarial-audit-gate.md` the diff must go through fresh-context `evidence-auditor` before merge. Will fire after green-tests.

## Verification

1. `pytest tests/test_trading_app/ tests/test_pipeline/ -q -k "session_orchestrator or prop_profiles or lane_allocation"` — all pass.
2. `python pipeline/check_drift.py` — 155 PASSED (unchanged), grep-gate passes, dead-allowlist-entry sub-check passes (allowlist entry was removed in lockstep).
3. `grep -n "lane_allocation\.json" trading_app/live/session_orchestrator.py` — returns ZERO matches (literal fully purged).
4. Self-review walk through the three branches: (a) `profile_*` + missing → RuntimeError with operator-runbook message; (b) `profile_*` + corrupt → RuntimeError with parse-error suffix; (c) paper/signal account or no portfolio → empty set, no raise, log preserved.
5. Adversarial-audit-gate evidence-auditor dispatch on the diff (fresh-context, capital-class).

## Acceptance criteria — Stage done when

- [ ] `lane_allocation.json` literal absent from `trading_app/live/session_orchestrator.py`
- [ ] `_LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST` no longer contains `session_orchestrator.py`
- [ ] `resolve_allocation_json(profile_id)` is the sole path-resolution surface used by the allocator-block gate
- [ ] FAIL-CLOSED preserved for `profile_*`: missing → RuntimeError (named path), corrupt → RuntimeError (parse error)
- [ ] Fail-open preserved for paper/signal accounts: empty set, "No allocation file — regime gate disabled" log when legacy file absent (literal `lane_allocation.json` purged from log strings to satisfy grep-gate)
- [ ] `python pipeline/check_drift.py` reports 155 PASSED with grep-gate PASS and dead-allowlist-entry PASS
- [ ] Targeted pytest slice passes
- [ ] `evidence-auditor` adversarial-audit pass on the diff (per adversarial-audit-gate.md)
- [ ] Sub-stage progress checklist in parent stage file ticks 1b-ii.b
- [ ] Commit pushed on `session/joshd-multi-profile-lane-allocation`

## Notes

- `profile_id` extraction at line 360 happens BEFORE the allocator-block gate at line ~420, so it's in scope. Non-`profile_*` accounts get `profile_id = None`, which the resolver+loader handle as legacy-only semantics.
- The corruption-vs-missing distinction is preserved by a residual probe ONLY for `profile_*` accounts. The probe is the same JSON parse the operator's runbook expects to fail loudly on, not a re-implementation of resolver semantics.
