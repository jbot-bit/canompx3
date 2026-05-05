## Iteration: 181
## Target: pipeline/system_context.py:355
## Finding: read_claim() catches bare Exception and returns None silently, making the parallel-session blocker fail-open on corrupt/unexpected claim files
## Classification: [judgment]
## Blast Radius: 1 production file, callers in scripts/ (session_preflight.py, checkpoint_guard.py, session_router.py, project_pulse.py) — all read-only consumers of read_claim result; behavior unchanged for valid/missing files
## Invariants: [1] valid claim files still return SessionClaim; [2] missing files still return None; [3] no new exceptions propagate to callers — unexpected errors log warning + return None; [4] OSError + JSONDecodeError + ValidationError remain silent (non-corrupt parse errors are expected)
## Diff estimate: ~8 lines
## Doctrine cited: integrity-guardian.md § 3 (never catch Exception and return success in health/audit paths) + § 6 (no silent failures)
