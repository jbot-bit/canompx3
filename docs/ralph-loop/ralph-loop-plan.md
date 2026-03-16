## Iteration: 106
## Target: research/research_zt_event_viability.py:358
## Finding: Hardcoded variation count "18" in report output — should be computed dynamically from EVENT_FAMILIES structure
## Classification: [mechanical]
## Blast Radius: 1 file, 0 external callers (standalone research script, no importers)
## Invariants: (1) report structure and analysis logic unchanged; (2) computed count equals 18 with current EVENT_FAMILIES; (3) no changes to EVENT_FAMILIES or any analysis code
## Diff estimate: 4 lines (1 new compute expression, 1 string line changed)

### ZT-01 (LOW): research_zt_event_viability.py:358
- Hardcoded `"18"` directional cells in markdown report template
- Derived from: 3 EVENT_FAMILIES × 3 follow_windows × 2 models = 18
- If EVENT_FAMILIES structure changes, the report count silently diverges from actual tested count
- Fix: compute `variation_count` dynamically and use f-string interpolation in report
