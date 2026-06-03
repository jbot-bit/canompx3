# Condition-Based Waiting — poll the condition, never guess the delay

**Load-policy:** referenced from CLAUDE.md § Backtesting Methodology / test
hygiene. Read on demand when writing or fixing a test, hook, or background-task
poll that waits on async state (DB readiness, file appearance, subprocess exit,
MCP handle release).

**Authority:** ported and tailored from the superpowers `condition-based-waiting`
technique (2026-05-31 plugin-absorption pass) and ratified by this repo's own
incident history — `time.sleep()`/`Start-Sleep` race patterns have bitten us
(gold.db lock-churn waits, MCP-handle release polling, pre-commit backfill
timing). The Bash tool guidance already says *"do not retry failing commands in
a sleep loop"* and *"use a check command rather than sleeping first"*; this rule
is the test/code-side companion to that operator-side stance.

---

## The rule

**Wait for the actual condition you care about, not a guess about how long it
takes.** An arbitrary `time.sleep(N)` passes on a fast box and flakes under CI
load or a busy gold.db writer — the worst failure class, because it's
intermittent and reads as "passed."

```python
# ❌ BEFORE — guessing at timing
time.sleep(0.5)
assert build_daily_features_done()          # flakes when the build runs long

# ✅ AFTER — polling the real condition
wait_for(build_daily_features_done, "daily_features build complete", timeout_s=30)
assert build_daily_features_done()
```

### Canonical helper (Python idiom)

```python
import time

def wait_for(condition, description, timeout_s=10.0, poll_s=0.05):
    """Poll `condition()` until truthy or timeout. Returns the truthy value.
    Raises TimeoutError with a clear message — never hangs forever."""
    deadline = time.monotonic() + timeout_s
    while True:
        result = condition()          # call INSIDE the loop — fresh data each poll
        if result:
            return result
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {description} after {timeout_s}s")
        time.sleep(poll_s)            # 50ms: not a busy-spin, not a guess
```

### canompx3 wait-on patterns

| Waiting for | Condition |
|---|---|
| DuckDB write-lock released | a `read_only=False` connect succeeds (try/except, retry) |
| MCP handle reaped | `reap_stale_claude_processes.py` reports 0 holders |
| Generated artifact | `Path(html).exists() and Path(html).stat().st_size > 0` |
| Background `run_in_background` task | the task-notification fires — **don't poll it**, the harness re-invokes you |
| daily_features rebuilt | row exists for `(trading_day, symbol, orb_minutes)` in gold.db |

---

## When an arbitrary sleep IS correct

Only when you are testing *timed behavior itself* (a debounce, a heartbeat
interval, a 10-minute peer-session staleness window). Then:

1. First `wait_for` the triggering condition.
2. Then sleep a duration derived from **known** timing, not a guess.
3. Comment WHY: `time.sleep(0.2)  # 2 heartbeat ticks @ 100ms — verifying partial emit`.

A bare `time.sleep()` with no triggering-condition wait and no WHY comment is the
anti-pattern this rule exists to kill.

---

## Common mistakes

- **Polling too fast** (`poll_s=0.001`) — wastes CPU, starves the thing you wait on. 50ms is the floor.
- **No timeout** — a condition that never comes hangs the test/CI forever. Always bound it.
- **Stale read** — caching the state before the loop instead of calling the getter inside it. Re-check fresh each poll.
- **Sleeping to "wait for" a harness-tracked background task** — you'll be re-invoked on completion; polling it just burns turns (see Bash tool guidance + `ScheduleWakeup` notes).

---

## Related

- `CLAUDE.md` § Shell / Bash tool guidance — operator-side "no sleep-retry loops".
- `.claude/skills/quant-debug/SKILL.md` § Step 4.5 — defense-in-depth (a flaky
  test is a missing condition-guard).
- `memory/feedback_worktree_stale_lease_pid_roll_and_golddb_lock_churn_2026_05_30.md`
  — a real gold.db-lock timing incident this rule guards against.
