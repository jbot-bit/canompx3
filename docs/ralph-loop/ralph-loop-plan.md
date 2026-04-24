## Iteration: 170
## Target: trading_app/outcome_builder.py:_compute_outcomes_all_rr (line 217)
## Finding: `break_ts` is a dead parameter in `_compute_outcomes_all_rr` — accepted in the signature but never read in the function body. Two production callsites and two test callsites pass it unnecessarily. Institutional-rigor.md § 5: "Dead parameters passed for 'future use' are drift bait."
## Classification: [mechanical]
## Blast Radius: 3 files (trading_app/outcome_builder.py, tests/test_trading_app/test_stress_hardcore.py — no other files call _compute_outcomes_all_rr)
## Invariants:
## 1. _compute_outcomes_all_rr behavior is identical — only the signature changes (break_ts removed)
## 2. All tests pass with unchanged assertions
## 3. build_outcomes production behavior unchanged
## Diff estimate: 5 lines changed (1 signature + 2 production callsites + 2 test callsites)
## Doctrine cited: institutional-rigor.md § 5 (no dead parameters — drift bait)
