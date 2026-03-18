## Iteration: 131
## Target: trading_app/ml/predict_live.py:307
## Finding: Aggressive RR mismatch (trade skip, take=False) logged at logger.debug() — invisible at INFO level; aperture mismatch (take=True) correctly logged at INFO
## Classification: [mechanical]
## Blast Radius: 1 file (predict_live.py only — log level change)
## Invariants:
  1. take=False behavior MUST remain unchanged (only log level changes)
  2. rr_mismatch_count counter MUST still be incremented
  3. Cached MLPrediction result MUST be identical
## Diff estimate: 1 line
