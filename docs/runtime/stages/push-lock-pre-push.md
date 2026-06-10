---
task: Add a push-serialization lock to .githooks/pre-push, reusing commit_serialize.py (parameterized lock filename), so concurrent HEAD:main pushes don't each burn ~3min drift and lose to a stale-ref reject. Operator-approved fix; exit-instantly behavior (not wait/poll).
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/commit_serialize.py
  - .githooks/pre-push
  - .githooks/pre-commit
  - tests/test_tools/test_commit_serialize.py
---

## Scope Lock
- scripts/tools/commit_serialize.py
- .githooks/pre-push
- .githooks/pre-commit
- tests/test_tools/test_commit_serialize.py

## Scope expansion (informed mid-implementation, anti-creep §)
Live-repro (not unit tests) surfaced a PRE-EXISTING liveness bug in the serializer
that the push lock would inherit: `_holder_is_dead_or_stale` keys liveness on the
HELPER subprocess pid, which dies the instant `acquire()` returns. During the
~3-min drift the lock therefore holds a DEAD pid → a 2nd concurrent gate steals
it instead of blocking. PROVEN: `rc=0` (steal) against a live-shell-held lock.
Naive `ppid` fix is INERT on Git Bash (MSYS pid ≠ native Windows pid; `_pid_alive`
returns False for it — PROVEN). Correct mechanism: hook exports its NATIVE pid via
`/proc/$$/winpid` (resolvable by `_pid_alive`, PROVEN True), helper records it as
the liveness pid. Operator chose the real fix → both pre-commit + pre-push now in
scope. Capital path → adversarial-audit gate before done.

## Blast Radius
- scripts/tools/commit_serialize.py — (a) generalize hardcoded LOCK_FILENAME into an optional CLI arg; default `.commit-in-progress.lock` preserved. (b) Add a NATIVE-pid liveness field (`live_pid`) recorded from a hook-exported env var; `_holder_is_dead_or_stale` checks it (falls back to legacy `pid` when the env var is absent → in-process tests + old locks still work).
- .githooks/pre-commit — export `CANOMPX3_HOOK_NATIVE_PID=$(cat /proc/$$/winpid 2>/dev/null || echo $$)` before the existing step-0 acquire, so the commit serializer's liveness keys on the live hook shell. No other change.
- .githooks/pre-push — new step 0: export the same native-pid env var, acquire `.push-in-progress.lock` BEFORE the ~3min drift; `trap release EXIT`. Fail-OPEN on the lock; drift gate stays fail-CLOSED + untouched.
- tests/test_tools/test_commit_serialize.py — lock-name arg coverage + default preserved + NEW: live-pid env recorded & gates staleness (dead live_pid → steal; live live_pid → block), legacy-lock fallback.
- Reads: git common dir + /proc/$$/winpid (read-only). Writes: `<git-common-dir>/.{commit,push}-in-progress.lock` (inside .git/, never tracked). No schema, no trading_app/ touch. CAPITAL-ADJACENT: changes a fail-closed-gate's serializer → adversarial-audit gate applies.
