---
paths:
  - "research/**"
  - "trading_app/strategy_*"
  - "docs/audit/**"
---
# Quant Agent Identity — Seven-Sins Bias-Class Index

**Narrowed 2026-05-17:** this file auto-loads ONLY on research / strategy_* / audit edits — the paths where Seven-Sins bias defense actually applies. Production pipeline + scripts edits no longer trigger this rule; they pick up canonical-source + fail-closed discipline via `integrity-guardian.md` and the Seven-Sins reminder via `institutional-rigor.md` § 12.

## Seven Sins quick-table

| Sin | What to watch for |
|-----|-------------------|
| **Look-ahead bias** | Future data as predictor. Banned columns + E2 break-bar suffixes → see `backtesting-methodology.md` § 1.1 + § 6.3 |
| **Data snooping** | Claiming significance after testing N+ hypotheses without BH FDR correction at the appropriate K framing |
| **Overfitting** | High Sharpe with N < 30, or passing only one year. MinBTL ceiling from Bailey 2013 |
| **Survivorship bias** | Dropping dead instruments (MCL/SIL/M6E/MBT/M2K) or purged entry models (E0) from base rates |
| **Storytelling bias** | Narrative around noise. p > 0.05 → observation, not finding |
| **Outlier distortion** | One extreme day driving aggregate. Year-by-year breakdown required |
| **Transaction cost illusion** | Always use `COST_SPECS` from `pipeline/cost_model.py` |

## Data Snooping Quarantine

The AI agent is itself a vector for data leakage:
- Never optimize parameters against the OOS / holdout window
- Never cherry-pick strategies by peeking at OOS, then retroactively justify IS
- "Which strategy should I trade?" → answer with FDR-validated FIT/WATCH only

## Related authority

- `institutional-rigor.md` § 9 — Seven-Sins awareness as a behavioral check on every change (canonical home of the reminder)
- `integrity-guardian.md` § 5 (Evidence Over Assertion) — generation is not validation; reading code ≠ verifying code
- `backtesting-methodology.md` — operational rules; this file is the *bias-class index*
