---
task: Commit 14 fresh fast-lane scanner trial-ledger appends from 2026-05-20 scanner runs (append-only generated state; precedent commit 772c27e7). Originally drafted at 4 entries; sibling-session scanner re-runs while .gitignore over-match was hiding the working tree pushed the count to 14 by the time the gitignore anchor fix (4bd288c4) surfaced these to git status.
mode: TRIVIAL
updated: 2026-05-21
scope_lock:
  - docs/runtime/fast_lane_trial_ledger.yaml
acceptance:
  - git diff is append-only (490 insertions, 0 deletions; verified)
  - 14 new run_id entries from 2026-05-20 scanner runs (verified; 7 scanner-1f726fa8 pairs across 7 timestamps T154559Z through T160600Z)
  - python pipeline/check_drift.py passes (modulo pre-existing MGC carry-over)
  - committed direct to main per project default (non-capital, append-only audit data)
---

## Context

Working tree carried 14 unstaged scanner trial-ledger appends from Stage 2A.3 scanner smoke runs on 2026-05-20 (7 timestamps × 2 preregs each):
- T154559Z, T154847Z, T155325Z, T155338Z, T155756Z, T160217Z, T160600Z
- Each timestamp produced one entry per prereg:
  - scanner-1f726fa8f078c6e0 (MNQ COMEX_SETTLE E2 RR2.0 ORB_VOL_16K pooled-O5)
  - scanner-64156e75c55701b6 (MNQ US_DATA_1000 E1 RR1.0 PD_CLEAR_LONG O30)
- The .gitignore over-match (`runtime/` unanchored, fixed in 4bd288c4) was concealing these appends from `git status`; they only surfaced after the anchor.

Append-only YAML; mirrors precedent commit `772c27e7` ("chore(fast-lane): append scanner trial-ledger entries from 2026-05-20 runs").

## Blast Radius

Zero. File is canonical trial-universe accounting (Bailey-López de Prado 2014 § 3); checks #169/#170/#173 read it but only require schema integrity which append_entry() enforces at write time.
