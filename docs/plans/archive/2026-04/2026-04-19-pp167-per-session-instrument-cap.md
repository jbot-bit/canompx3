---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# PP-167 — Per-(session, instrument) ORB Cap Design Proposal

**Created:** 2026-04-19
**Status:** IMPLEMENTED — landed on `main` 2026-04-23 after autonomous follow-through on the repo cleanup queue.
**Owner:** canompx3
**Campaign:** 2026-04-19 MAX-EV extraction campaign, Phase 4.1
**Authority:** `.claude/rules/institutional-rigor.md` § Design Proposal Gate; `.claude/rules/branch-discipline.md`

## Implementation outcome

The recommended option **(a)** was implemented on `2026-04-23`:

- `get_lane_registry()` now returns `dict[tuple[str, str], dict]` keyed by `(orb_label, instrument)`.
- `trading_app/live/session_orchestrator.py` now loads and checks ORB caps with the same `(session, instrument)` key.
- The remaining runtime registry consumers (`scripts/tools/slippage_scenario.py` and `scripts/tools/forward_monitor.py`) were updated in the same change so the key-shape migration landed closed.
- Targeted verification passed:
  - `./.venv-wsl/bin/pytest tests/test_trading_app/test_prop_profiles.py tests/test_trading_app/test_session_orchestrator.py -q`
  - `./.venv-wsl/bin/python pipeline/check_drift.py`

`self_funded_tradovate.active` was intentionally left unchanged. Repo truth still says activation is gated on opening the account and passing API tests (`trading_app/prop_profiles.py`), and the readiness/deployment audits still classify that surface as not currently deployable (`docs/audit/results/2026-04-19-dormant-profile-activation-readiness-scan.md`, `docs/audit/results/2026-04-19-validated-shelf-vs-live-deployment-audit.md`). This change closes the false schema conflict without promoting an unready profile.

---

## Problem

`trading_app/prop_profiles.py:970 get_lane_registry` raises `ValueError` on the `self_funded_tradovate` profile because two of its sessions contain lanes for different instruments with different `max_orb_size_pts` caps:

- `EUROPE_FLOW`: `MNQ_*` cap=150.0, `MGC_*` cap=30.0
- `NYSE_OPEN`: `MNQ_*` cap=150.0, `MES_*` cap=60.0

The `get_lane_registry` docstring (lines 971-984) asserts: *"The ORB cap is a session-level attribute — a function of the session's volatility profile, not of the specific strategy — so every lane on the same session must share the same cap."* That assertion is false when a profile has multi-instrument sessions. Micro-gold's volatility profile is genuinely different from micro-equity-index volatility at the same wall-clock session; a shared cap cannot represent both.

Current mitigation: `self_funded_tradovate` has `active=False` and `resolve_profile_id(..., exclude_self_funded=True)` is the default. The full 10-lane Tradovate diversification plan is blocked.

## Non-goals for this proposal

- No changes to `validated_setups`, `lane_allocator`, or `sr_monitor`.
- No change to per-lane `max_orb_size_pts` values on `DailyLaneSpec` instances.
- No change to the `min()` of session caps as a safety default — caller logic only.

## Consumer inventory (`max_orb_size_pts`)

Scoped to canonical tree (`trading_app/`, `scripts/`, `tests/`) — worktree scratch copies excluded.

| File | Line | Usage | Shape |
|---|---:|---|---|
| `trading_app/prop_profiles.py` | 78 | Schema: `DailyLaneSpec.max_orb_size_pts: float \| None` | per-lane |
| `trading_app/prop_profiles.py` | 299, 367, 368, 369, 403, 512, 575, 576, 577, 578, 613-14-15-16, 652-56, 691-95, 730+, ~60 more | Literal values per `DailyLaneSpec` | per-lane |
| `trading_app/prop_profiles.py` | 970-1009 | `get_lane_registry` — fails on conflict | session-keyed read |
| `trading_app/live/session_orchestrator.py` | 229-233 | Builds `_orb_caps: dict[str, float]` keyed by `orb_label` | session-keyed read (needs fix) |
| `trading_app/pre_session_check.py` | 597 | `orb_cap = lane.get("max_orb_size_pts")` per-lane loop | per-lane read (already correct) |
| `trading_app/derived_state.py` | 43 | Exports per-lane into derived state dict | per-lane read (already correct) |
| `scripts/tools/generate_trade_sheet.py` | 514 | Reads `lane.max_orb_size_pts` per-lane | per-lane read (already correct) |
| `scripts/tools/build_optimal_profiles.py`, `generate_profile_lanes.py` | various | Generators writing `max_orb_size_pts=` literals | write-side (no runtime impact) |
| `tests/test_trading_app/test_prop_profiles.py` | 112, 199-230+ | Asserts per-lane + session-registry consistency | needs test update under any option |

**Blast summary:** 1 session-keyed reader (`session_orchestrator`), 1 session-keyed fail-closed validator (`get_lane_registry`). Everything else is already per-lane or write-side.

## Option (a) — Per-(session, instrument) registry [RECOMMENDED]

### Shape

Change `get_lane_registry` return type from `dict[str, dict]` (session-keyed) to `dict[tuple[str, str], dict]` keyed by `(orb_label, instrument)`. Callers that currently ask "what's the cap for session X?" must now ask "what's the cap for session X on instrument Y?"

### `get_lane_registry` becomes

```python
def get_lane_registry(profile_id: str | None = None) -> dict[tuple[str, str], dict]:
    """Return a (session, instrument)-keyed lane map.

    ORB cap (``max_orb_size_pts``) is a per-(session, instrument) attribute —
    each instrument has its own volatility profile; a single session cap
    cannot represent multiple instruments. Fails closed if two lanes share
    the same (session, instrument) pair with different caps (that's a real
    conflict).
    """
    registry: dict[tuple[str, str], dict] = {}
    conflicts: dict[tuple[str, str], set[float | None]] = {}
    for lane in get_profile_lane_definitions(profile_id):
        key = (lane["orb_label"], lane["instrument"])
        cap = lane.get("max_orb_size_pts")
        if key not in registry:
            registry[key] = lane
            continue
        existing = registry[key].get("max_orb_size_pts")
        if cap != existing:
            conflicts.setdefault(key, set()).update([existing, cap])
    if conflicts:
        raise ValueError(
            "Profile has inconsistent max_orb_size_pts across lanes on the "
            f"same (session, instrument): {...}. Reconcile DailyLaneSpec entries."
        )
    return registry
```

### Consumer changes required

1. **`session_orchestrator.py:220-237`:** `self._orb_caps` becomes `dict[tuple[str, str], float]`. Cap lookup at entry time becomes `self._orb_caps.get((label, instrument))`. Requires passing instrument at the ORB-cap check site.
2. **`pre_session_check.py:597`:** already per-lane; no change needed (reads from lane dict which carries its own `max_orb_size_pts`).
3. **`derived_state.py:43`:** already per-lane; no change.
4. **`scripts/tools/generate_trade_sheet.py`:** already per-lane; no change.
5. **Tests:** update `test_get_lane_registry_session_consistency`-style assertions to `(session, instrument)` keying. Add multi-instrument-session fixture covering `self_funded_tradovate` case.

### Migration

Single PR on named branch. No data migration (schema of `DailyLaneSpec` unchanged). No DB migration. No worktree rebases beyond the usual.

### Unlocks

- `self_funded_tradovate.active = True` after consumer changes land and tests pass. 10 lanes (5 MNQ + 2 MGC + 2 MES + 1 self-funded-specific combo per `prop_profiles.py:855-883`).
- Future multi-instrument profiles don't hit this wall.

### Risks

- **Consumer sweep:** one hot path (`session_orchestrator`) needs code edit. Missing any call site → silent KeyError or wrong cap application. Mitigation: the fail-closed `ValueError` on `get_lane_registry` moves to runtime tuple-keyed; a missed consumer would fail on `dict[(label, instrument)]` access immediately in tests.
- **Tests:** existing tests that assume `get_lane_registry()[label]` break immediately. Good — forces the consumer sweep.
- **Blast radius:** ~3 production files, 1 test file. Small.

## Option (b) — `min()` reducer on conflict

### Shape

Keep `get_lane_registry` session-keyed (`dict[str, dict]`). On conflict, emit `min(caps)` across instruments, carrying a warning log rather than raising.

### Trade-offs

- **Pro:** zero consumer changes. Single-file fix in `prop_profiles.py`.
- **Con:** MGC's 30pt cap on `EUROPE_FLOW` would veto MNQ's 150pt cap → MNQ trade entries on `EUROPE_FLOW` would be gated at `risk_points >= 30`, blocking most legitimate MNQ entries. Silently mis-caps MNQ.
- **Con:** The docstring claim "cap is a session-level attribute" remains false; we'd be encoding a false invariant with a papered-over reducer.
- **Con:** Violates `.claude/rules/institutional-rigor.md` § 4 (delegate to canonical sources) — the per-instrument cap data exists on `DailyLaneSpec`; collapsing it to `min()` is information loss.

**Verdict:** rejected — lowest-cost but highest-misfire cost.

## Option (c) — Composite cap object, session-keyed

### Shape

`get_lane_registry` returns `dict[str, CapBundle]` where `CapBundle` is a mapping `instrument -> cap`. Session-keyed outer; per-instrument inner. Consumers navigate the inner map.

### Trade-offs

- **Pro:** Preserves session as primary key. Minimal change to call sites that operate session-first.
- **Con:** Introduces a new schema concept (CapBundle). More code than (a).
- **Con:** Ultimately equivalent to (a) in semantics — (session, instrument) pair is still the lookup key, just with an extra layer.

**Verdict:** viable but strictly dominated by (a). Reject.

## Recommendation

**Option (a).** Reasons:
1. Smallest honest schema change — (session, instrument) IS the natural key; session-only keying was an incorrect invariant.
2. Smallest blast radius of any option that preserves information correctness (3 files + tests).
3. Fails closed at the right layer — real conflicts (two lanes on same `(session, instrument)` with different caps) still raise `ValueError`; only the false conflict (different instruments sharing a session) disappears.
4. Unlocks `self_funded_tradovate` 10-lane profile without silently distorting MNQ entries.

## Acceptance criteria (when implementation lands)

1. `get_lane_registry` returns `dict[tuple[str, str], dict]` signature + docstring updated.
2. `session_orchestrator._orb_caps` is `dict[tuple[str, str], float]`; ORB-cap check at entry time passes `(label, instrument)`.
3. `self_funded_tradovate.active = True` in the same commit that lands the above (not before — gated by consumer sweep).
4. New test covering multi-instrument-session profile returns 2 distinct `(label, instrument)` entries without raising.
5. Existing tests updated to tuple keying; all green.
6. `python pipeline/check_drift.py` passes with no new violations.
7. `grep -rn "get_lane_registry" trading_app/ scripts/ tests/` returns every call site using the new signature.

## Out-of-scope / deferred

- Actually populating the 3 non-MNQ lanes of `self_funded_tradovate` with Mode-A-recomputed ExpR. That's Phase 3.2 (allocator baseline refresh) territory.
- Cross-instrument cap aggregation for reporting / fitness (e.g., the dashboard might still want "session-worst cap" summary). Add as a helper, don't conflate with the primary registry.
- Refactoring the 60+ `DailyLaneSpec(..., max_orb_size_pts=...)` literals in `prop_profiles.py`. Values stay as-is per Phase 4.1 non-goals.

## Verification protocol for implementation phase

Per the campaign plan and `.claude/rules/institutional-rigor.md`:

- Drift check passes (`python pipeline/check_drift.py`).
- Full `tests/test_trading_app/test_prop_profiles.py` green.
- `tests/test_trading_app/test_session_orchestrator*` green.
- Dead-code sweep: `grep -rn "get_lane_registry" .` — every call site uses new signature.
- `self_funded_tradovate` turned `active=True` only after all of the above.
- Self-review per `.claude/rules/institutional-rigor.md` § 1 before declaring done.

## Blast radius map (concise, per `.claude/rules/institutional-rigor.md`)

**Will edit:**
- `trading_app/prop_profiles.py` — `get_lane_registry` signature + docstring + conflict message
- `trading_app/live/session_orchestrator.py` — `_orb_caps` typing + lookup site(s)
- `tests/test_trading_app/test_prop_profiles.py` — tuple-keyed assertions + multi-instrument fixture
- `docs/` — this proposal referenced in the amendment

**Won't edit:**
- `trading_app/pre_session_check.py` (already per-lane)
- `trading_app/derived_state.py` (already per-lane)
- `scripts/tools/generate_trade_sheet.py` (already per-lane)
- Any DB schema or data
- `lane_allocator`, `sr_monitor`, `strategy_validator`, `check_drift`

**Will touch transitively (tests):**
- Any test that calls `get_lane_registry()[label]` directly — search: `grep -rn "get_lane_registry" tests/`

---

## Sign-off

Pick option **(a) / (b) / (c)** or request amendment. On approval, this moves to a named implementation branch with a separate stage-gate file. No code edits authorized until sign-off.

**End of proposal.**
