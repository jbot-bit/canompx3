## Iteration: 199
## Target: trading_app/lane_allocator.py:1136-1139
## Finding: Hysteresis block silently drops a deployable lane when best_prior was demoted by an upstream gate (status not in DEPLOY/RESUME/PROVISIONAL) — neither the new candidate nor the prior lane gets selected.
## Classification: [judgment]
## Blast Radius: 1 production file, 1 test file
## Invariants: [gates apply before ranking; hysteresis never rescues a demoted prior; slot count must not silently decrease]
## Diff estimate: 6 lines production + 25 lines test
## Doctrine cited: integrity-guardian.md § 6 (no silent failures — silent slot drop in capital-class path)
