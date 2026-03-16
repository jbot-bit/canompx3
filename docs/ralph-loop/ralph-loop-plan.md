## Iteration: 107
## Target: research/research_vol_regime_switching.py:192,784
## Finding: (1) hardcoded IN ('E1','E2') in load_data() SQL; (2) unused datetime import + hardcoded static date in output
## Classification: [mechanical]
## Blast Radius: 1 file, 0 production callers
## Invariants: (1) SQL filtering logic unchanged — only the IN clause literal replaced with runtime expression; (2) date in output reflects today; (3) no analysis logic changes
## Diff estimate: 2 lines changed

### VS-05 (MEDIUM): research_vol_regime_switching.py:192
- `AND o.entry_model IN ('E1', 'E2')` hardcoded — use ENTRY_MODELS (same fix already applied to get_validated_sessions)

### VS-04 (LOW): research_vol_regime_switching.py:784 + orphan import
- `datetime` imported but never used; hardcoded date 2026-03-01 in output
- Fix: use datetime.date.today().isoformat()
