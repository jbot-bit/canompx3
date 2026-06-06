task: Build SR-watchlist Part C (report-only) — enroll the 719 promotable-FIT candidates into SR monitoring, score from canonical forward outcomes, write a SEPARATE sr_watchlist_state.json the allocator/rebalance NEVER reads. Diagnostic deliverable: CONTINUE/ALARM/NO_DATA split to learn whether the SR gate or forward-sample is the real allocation ceiling.
mode: DESIGN
status: NOT STARTED — design complete + 2nd-pass-audited; awaiting IMPLEMENTATION session

## Scope Lock
- trading_app/sr_monitor.py
- trading_app/prop_profiles.py
- pipeline/check_drift.py
- (new) sr_watchlist_state.json

## Blast Radius
- trading_app/sr_monitor.py — add a watchlist universe + _build_watchlist_lanes(); write sr_watchlist_state.json via canonical derived-state envelope. Touches a capital-adjacent runtime module -> Tier B, adversarial-audit gate required before merge despite report-only output.
- trading_app/prop_profiles.py — add get_promotable_watchlist_lanes() helper returning canonical lane defs (reuse parse_strategy_id + existing lane-def shape; institutional-rigor §4 delegate-not-reencode). Reads validated_setups/fitness; writes nothing.
- pipeline/check_drift.py — +1 check: sr_watchlist_state.json is report-only (no allocation/rebalance consumer reads it).
- scripts/tools/rebalance_lanes.py — NOT touched in Part C (this is where --strict-live-clean gate pauses; Part E target).
- trading_app/lane_allocator.py — NOT touched in Part C (Part E only).
- Reads: gold.db (read-only, canonical orb_outcomes + paper_trades + validated_setups). Writes: sr_watchlist_state.json only (report-only, allocator-invisible). Zero capital-affecting writes.

## Design reference
Full design: docs/plans/sr-watchlist-part-c-report-only.md (commit 49772a3a + 2nd-pass amendments).
Memory: project_allocation_bottleneck_is_sr_unknown_circular_gate_2026_06_03.md.

## Approved decisions (do not re-litigate)
- Part C ONLY, then reassess with real numbers. Do NOT build Part E (capital path) in the same session.
- Honest NO_DATA — no capital moves on NO_DATA; surface the sample-starved gap as the primary deliverable.

## Keystone facts (2nd-pass-verified — do not trust 1st-pass cites)
- PAUSE decision: scripts/tools/rebalance_lanes.py:128-138, flag-gated by --strict-live-clean. Pauses sr_status != "CONTINUE" -> NO_DATA is paused IDENTICALLY to UNKNOWN. Unlock is bounded by CONTINUE-eligible candidates, NOT all 719.
- sr_status populated by lane_allocator.enrich_scores_with_liveness (lane_allocator.py:625) from sr_state.json.
- Circular source: sr_monitor._build_lanes (sr_monitor.py:111) iterates get_profile_lane_definitions = current allocation only.
- SR verdict (sr_monitor.py:289-302): NO_DATA iff zero forward trades; CONTINUE iff >=1 forward trade and no alarm; NO minimum-N floor.
- prepare_monitor_inputs (sr_monitor.py:157-208): 3-tier baseline preference. Watchlist candidates MUST land in tier 3 (canonical_forward) -> ASSERT paper_trades empty per candidate; surface exceptions.
- BASELINE_WINDOW = 50 (sr_monitor.py:67).
- Live numbers (2026-05-30 rebalance): active=3, paused=845, scored=848; 782 paused on SR-UNKNOWN gate. promotable-FIT=719.

## Open questions for IMPLEMENTATION (see design § 9)
- Half-size/shadow/_S075 lane identity: how to enroll candidates validated at a different stop_multiplier (do NOT silently inherit active profile's). 
- Watchlist universe = promotable-FIT MINUS already-allocated, no aperture double-count.
- Bulk-load forward outcomes once per (instrument, orb_minutes) — avoid 719 connects + the 2026-05-21 SIGSEGV class.

## Acceptance criteria (design § 10) — done = all 6 true
1. sr_watchlist_state.json exists, envelope-valid (drift #124/#125 pass).
2. New drift check proves no allocation/rebalance consumer reads it.
3. Reports CONTINUE/ALARM/NO_DATA counts + forward-N distribution + paper_trades-non-empty exceptions for the 719.
4. Allocator + rebalance behavior byte-identical to pre-change.
5. check_drift.py green; full-set scoring runs without SIGSEGV.
6. Self-review + evidence-auditor (capital-adjacent path) pass before merge.

## Preflight for next session
- Branch session/joshd-maximise-ops-fix has 3 unpushed commits (71056eb2 guard-msg fix, 49772a3a plan, e85dd06c stage-close) + this stage's design. Decide push before starting Part C.
- Re-verify the live histogram still shows the SR-UNKNOWN gate dominant (flag-conditional — design § 9).
