# Stage D — Cherry-pick loop audit fixes (post-review FAIL on Stage A + silent gap on Stage B)

task: Fix the two evidence-auditor findings on commits 81da1099 and b3bb9bdf. (1) Stage A FAIL: `compute_oos_power_readiness` misuses two-sample Welch helper by passing IS trade count as OOS group A — replace with canonical `one_sample_power`. (2) Stage B silent gap: bridge propagates source-scope dict unchanged, could leak a `theory_citation` key if a future source carries one — explicit allowlist of scope fields.

mode: IMPLEMENTATION

scope_lock:
  - scripts/research/cherry_pick_ranker.py
  - scripts/research/fast_lane_to_heavyweight_bridge.py
  - tests/test_research/test_cherry_pick_ranker.py
  - tests/test_research/test_fast_lane_to_heavyweight_bridge.py

## Blast Radius

- scripts/research/cherry_pick_ranker.py — `compute_oos_power_readiness` rewritten to call `research.oos_power.one_sample_power(d=cohen_d, n=oos_n)`. cohen_d still reverse-engineered from pooled t-stat. Public API unchanged.
- scripts/research/fast_lane_to_heavyweight_bridge.py — `build_heavyweight_prereg` switches `"scope": dict(scope)` to an explicit allowlist of canonical scope fields. Source-leak attack surface closed.
- tests/test_research/test_cherry_pick_ranker.py — add 2 new tests: (a) one-sample power values match canonical helper at known inputs, (b) large-IS-N + small-OOS-N case no longer inflates power above a sanity ceiling.
- tests/test_research/test_fast_lane_to_heavyweight_bridge.py — add 1 new test: source YAML with synthetic `scope.theory_citation` does NOT leak into the emitted draft.
- Drift checks #160 + #161: unaffected (both target constants, not function bodies).
- Reads: research/oos_power.py (canonical helper, no modification).
- Capital-class? No. Bug fix on ranker scoring + defensive guard on bridge.

## Acceptance criteria

1. `pytest tests/test_research/test_cherry_pick_ranker.py tests/test_research/test_fast_lane_to_heavyweight_bridge.py -v` PASS including 3 new tests.
2. `python pipeline/check_drift.py` count unchanged at 161, all checks still PASS (only pre-existing MGC orthogonal violation).
3. Re-run iteration 1 ranker → real QUEUED entry's score recomputes. Expect score to fall slightly (correcting the inflation). Existing skip_recommended=Y verdict preserved (no behavior reversal).
4. Bridge dry-run against the real QUEUED entry result MD → emitted YAML still has no theory_citation key AND new explicit allowlist preserves all currently-used scope fields (instrument, strategy_id, session, orb_minutes, entry_model, confirm_bars, rr_target, direction, filter_type, filter_source, out_of_scope).
