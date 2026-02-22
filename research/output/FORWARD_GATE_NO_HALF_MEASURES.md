# FORWARD GATE — NO HALF MEASURES

Status: ACTIVE
Date: 2026-02-22

## Scope (freeze discovery)
No new strategy discovery until current KEEP set passes/fails forward gate.

Candidates in gate:
- A0, A1, A2, A3
- B1 (optional)
- B2 (probation/weak)

## Hard pass criteria (all required)
1) Forward avgR > 0
2) Forward uplift vs baseline > 0
3) Forward max DD not worse than baseline by > 10%
4) Minimum forward sample met
   - gain-first candidates: N >= 60
   - frequency candidates: N >= 100
5) No retuning during forward window

## Auto-kill criteria
- Forward uplift <= 0 after minimum sample
- Forward avgR <= 0 after minimum sample
- Stability break (3 consecutive weekly negative deltas)

## Process
- Freeze rule definitions + thresholds now
- Run forward-only shadow logging
- Weekly scorecard only (no parameter edits)
- At gate end: PROMOTE or KILL only (no WATCH limbo)

## Automation (active)
- Manual run: `research\\run_forward_gate.cmd`
- Python updater: `python research/update_forward_gate_tracker.py`
- Latest status file: `research/output/forward_gate_status_latest.md`
- Daily scheduled task: `canompx3-forward-gate` (Windows Task Scheduler)
