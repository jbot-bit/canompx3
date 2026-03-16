## Iteration: 114
## Target: trading_app/ml/features.py:84
## Finding: `_backfill_global_features` uses `GLOBAL_FEATURES[0]` ("atr_20") as sole proxy for post-backfill NaN count, but early-exit check iterates ALL features — asymmetry could silently miss warning if atr_20 is backfilled while overnight_range (the #1 ML feature, 6.5% avg importance) is not
## Classification: [mechanical]
## Blast Radius: 1 file (private function, 3 internal call sites, 0 external callers); companion test checks data not warning — unaffected
## Invariants:
##   1. Return value of _backfill_global_features unchanged
##   2. n_still_missing is now max missing count across ALL GLOBAL_FEATURES (consistent with pre-backfill check)
##   3. Warning fires if ANY global feature still has NaN — mirrors lines 56-59 logic
## Diff estimate: ~4 lines
