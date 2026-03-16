## Iteration: 116
## Target: trading_app/strategy_discovery.py:1082-1088
## Finding: Fail-open fallback — empty get_enabled_sessions() silently falls back to all ORB_LABELS instead of raising; misconfigured instrument would run discovery for wrong sessions
## Classification: [judgment]
## Blast Radius: 1 file (strategy_discovery.py); callers: pipeline_status.py, scripts/infra/parallel_rebuild.py, run_full_pipeline.py — all pass instrument arg from config, not empty
## Invariants: run_discovery() public signature unchanged; fallback behaviour for valid instruments (non-empty sessions) unchanged; raise path only reached on misconfiguration
## Diff estimate: 4 lines
