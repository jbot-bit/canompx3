## Iteration: 94
## Target: scripts/tools/gen_playbook.py:18-30 (SESSION_ORDER)
## Finding: EUROPE_FLOW session missing from SESSION_ORDER — any EUROPE_FLOW validated strategies silently omitted from MARKET_PLAYBOOK.md
## Classification: [mechanical]
## Blast Radius: 1 file (gen_playbook.py only — standalone script, no importers)
## Invariants: [1] SESSION_ORDER display order must match Quick Reference table header; [2] INSTRUMENTS assert must stay; [3] tuple format (session_name, brisbane_winter, brisbane_summer, event_label) preserved
## Diff estimate: 1 line (add EUROPE_FLOW tuple to SESSION_ORDER between SINGAPORE_OPEN and LONDON_METALS)
