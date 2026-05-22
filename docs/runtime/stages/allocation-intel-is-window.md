---
task: Narrow allocation_intel.py session-regime seed window to strict-IS (anchor on HOLDOUT_SACRED_FROM, not CURRENT_DATE)
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/allocation_intel.py
blast_radius: |
  Single-file change to scripts/tools/allocation_intel.py. Replaces `CURRENT_DATE - INTERVAL 180 DAY` with `HOLDOUT_SACRED_FROM - INTERVAL 180 DAY` in `_compute_session_regimes()`. Imports `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`. Tool is read-only, called manually by operator. No callers in production code (verified via grep). Effect: HOT/COLD/FLAT classifications shift; cells whose positive avg_r lived in the post-2026-01-01 holdout slice (e.g. MGC NYSE_OPEN per docs/audit/results/2026-05-21-mgc-adjacency-preflight-park.md) will reclassify away from HOT. Verify by re-running the script and confirming the MGC NYSE_OPEN line either disappears or shows regime != HOT.
---

# Stage — allocation_intel IS-only seed window

## Why
`docs/audit/results/2026-05-21-mgc-adjacency-preflight-park.md` (commit 8d182a5b, PR #305) established that `allocation_intel.py § 4` HOT-undermapped scan surfaced MGC NYSE_OPEN and MGC SINGAPORE_OPEN as adjacency candidates **because** the 6-month rolling window straddled `HOLDOUT_SACRED_FROM` (2026-01-01). The entire +0.19R seed signal lived in the 5-month post-holdout slice; strict IS replay gave naive-t = +0.027 vs hurdle 3.00.

This same trap will surface any IS-flat / OOS-positive cell on any instrument. The fix is to anchor the seed window on `HOLDOUT_SACRED_FROM` instead of `CURRENT_DATE`, so the regime read uses only canonical IS data.

## Change
In `_compute_session_regimes` (lines 97-132):
- Import `from trading_app.holdout_policy import HOLDOUT_SACRED_FROM`
- Replace `o.trading_day >= CURRENT_DATE - INTERVAL 180 DAY` with `o.trading_day >= ? - INTERVAL 180 DAY AND o.trading_day < ?` and pass `HOLDOUT_SACRED_FROM` twice in params
- Update docstring: "6-month trailing window ending at HOLDOUT_SACRED_FROM, strict IS-only"

## Verify
1. `git diff` is one file, ≤25 lines net
2. `python pipeline/check_drift.py` passes
3. `python scripts/tools/allocation_intel.py` runs without error
4. MGC NYSE_OPEN no longer appears with regime=HOT in § 4 output (or, if it does, IS replay independently confirms HOT)
5. Output contains "## 4. HOT sessions with thin validated coverage" section as before
