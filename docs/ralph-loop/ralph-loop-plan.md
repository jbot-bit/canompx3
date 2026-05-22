## Iteration: 198
## Target: trading_app/chordia.py:350-362
## Finding: `load_chordia_audit_log` catches only `yaml.YAMLError` — `OSError` / `UnicodeDecodeError` from `p.read_text()` bypass the fail-closed contract, propagating to the capital-class allocator gate.
## Classification: [judgment]
## Blast Radius: 1 production file (chordia.py), callers: lane_allocator.py:315 + lane_allocator.py:735 + check_drift.py:9372 (all benefit automatically)
## Invariants:
##   1. Fail-closed: missing/unreadable YAML → empty log with default_has_theory=False
##   2. Operator-visible: all failure paths log at WARNING with the exception
##   3. No change to the PASS path (well-formed YAML returns the full parsed log)
## Diff estimate: 3 lines changed
## Doctrine cited: integrity-guardian.md § 3 (fail-closed mindset) + § 6 (no silent failures — every except must log)
