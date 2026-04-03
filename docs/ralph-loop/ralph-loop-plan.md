## Iteration: 138
## Target: pipeline/build_daily_features.py:1086
## Finding: Wrong comment "~200 5m bars ≈ ~3.5 days" contradicts the correct comment at line 664 "200 bars ≈ 16.7 hours of 5m bars" in the same file. 200 × 5min = 1000 min = 16.7 hours, not 3.5 days. Someone adjusting the lookback based on the wrong comment could reduce days=10 to ~4 and silently break RSI warm-up.
## Classification: [mechanical]
## Blast Radius: 0 callers affected (comment-only change), test_build_daily_features.py is companion test
## Invariants:
##   1. days=10 value MUST NOT change — it's conservatively correct
##   2. No logic changes — comment fix only
##   3. Drift checks must still pass
## Diff estimate: 1 line
