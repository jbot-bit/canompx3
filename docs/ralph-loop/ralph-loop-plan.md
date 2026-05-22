## Iteration: 196
## Target: trading_app/lane_correlation.py:29-30
## Finding: RHO_REJECT_THRESHOLD (0.70) and SUBSET_REJECT_THRESHOLD (0.80) lack @research-source annotations on capital-gate constants
## Classification: [mechanical]
## Blast Radius: 1 file (lane_correlation.py:29-30 only — adding comments, no logic change)
## Invariants: Threshold values 0.70 and 0.80 must NOT change; only annotations added; all callers unaffected
## Diff estimate: 4 lines added (comments)
## Doctrine cited: integrity-guardian.md § 8 (Never inline research stats without @research-source), research-truth-protocol.md
