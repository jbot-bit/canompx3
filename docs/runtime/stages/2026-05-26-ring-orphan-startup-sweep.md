---
task: |
  HARDEN — `data/live_bars/*.json` ring files orphaned on session crash.
  Audit-fix #2 (2026-05-22 commit 6d5c248b) preserves the ring on
  flush_to_db failure intentionally (data-loss prevention). But ANY
  crash before the shutdown hook means the ring file survives to the
  next session. Observed today (2026-05-26): pre-existing `MES.json`
  on disk after orchestrator restart, even though no MES session ran.

  The audit-fix's `is_stale + heartbeat cross-check` (bot_dashboard.py
  `_bars_watcher`) correctly refuses SSE pushes on a stale ring, so
  the operator dashboard does NOT show ghost bars. But:
    1. Stale rings accumulate forever — disk usage drift.
    2. PID metadata in the stale ring could mislead post-mortem
       analysis (looks like writer is alive when it's not).
    3. START_BOT.bat:49 already clears `bot_state.json` on launch —
       the symmetric ring-cleanup is missing.

  FIX: add a startup-side ring-sweep that:
    (a) Reads PID metadata from each `data/live_bars/<SYMBOL>.json`.
    (b) Checks if PID is alive (Windows: `tasklist /FI "PID eq <n>"`;
        POSIX: `os.kill(pid, 0)`).
    (c) If PID dead → delete ring file with INFO log.
    (d) If PID alive → leave alone (concurrent session legitimately
        writing).

  Add to `START_BOT.bat` as Step 2b (between bot_state clear and
  data-freshness check), OR — better — as an idempotent helper
  `scripts/tools/sweep_orphan_rings.py` called from BOTH:
    - START_BOT.bat:49 area
    - `scripts/run_live_session.py` early-startup (before
      `acquire_instance_lock` at line 930)

  Justification for double-call: START_BOT.bat is the operator path;
  `run_live_session.py` is also invoked by Codex/WSL and other
  launchers that don't go through .bat. Single helper, called from
  both = canonical-source-delegation per `institutional-rigor.md` § 4.

mode: IMPLEMENTATION
status: IMPLEMENTED_PENDING_OPERATOR_LIVE_REPRO_2026_05_25
priority: P2
deferred_reason: |
  Touches `trading_app/live/` and `scripts/run_live_session.py` — the
  same files the active 2026-05-26 smoke session has imported. Editing
  during live debut violates `adversarial-audit-gate.md` (HIGH severity
  on live-trading path). Land post-session.

scope_lock:
  - scripts/tools/sweep_orphan_rings.py  # NEW
  - tests/test_scripts/test_sweep_orphan_rings.py  # NEW
  - START_BOT.bat  # add Step 2b
  - scripts/run_live_session.py  # call helper before instance_lock
  - trading_app/live/bar_ring.py  # add `get_writer_pid_from_ring` helper if not present

agent: claude (opus 4.7)
---

## Blast Radius

- WRITES NEW: `scripts/tools/sweep_orphan_rings.py` (~80 lines). Pure-function `sweep(ring_dir: Path) -> list[Path]` returning deleted-file paths for testability. CLI wrapper with `--dry-run` flag.
- WRITES NEW: 5 unit tests covering: (a) empty dir, (b) ring file with live PID — preserved, (c) ring file with dead PID — deleted, (d) corrupt ring file — log + skip, (e) ring file missing PID field — log + skip (don't delete on ambiguous).
- WRITES (modify): `START_BOT.bat` — add `.venv\Scripts\python.exe scripts\tools\sweep_orphan_rings.py >nul 2>&1` after line 49.
- WRITES (modify): `scripts/run_live_session.py` — call `sweep_orphan_rings.sweep(LIVE_BARS_DIR)` before `acquire_instance_lock`.
- WRITES (modify): `trading_app/live/bar_ring.py` — IF `get_writer_pid_from_ring()` doesn't already exist, add it (read JSON, return `payload.get("writer_pid")`).
- LIVE-IMPACT: zero during running sessions (helper only deletes ring files where PID is dead = no writer). Adds ~50ms to startup.
- Idempotency: pure function, safe to call N times.
- Rollback: revert helper + 2 lines from .bat + 1 line from run_live_session.py.

## Acceptance

1. 5 tests pass.
2. Manual: kill orchestrator with `taskkill /F /PID <n>`, verify ring file present, run sweep, verify deleted.
3. Manual: start orchestrator, run sweep mid-session, verify ring file PRESERVED (PID alive check).
4. `pipeline/check_drift.py` passes.
5. New `--dry-run` mode prints what would be deleted without acting.

## Doctrine references

- `institutional-rigor.md` § 4 (canonical-source delegation — single helper, multiple callers)
- 2026-05-22 commit `6d5c248b` audit-fix #2 (preserves ring on flush failure — this stage RESPECTS that contract; only deletes when writer PID is provably dead)
- `feedback_n3_same_class_doctrine_threshold.md` (n=1 disk-orphan observation today; defer mechanical enforcement until n=2)

## Sources

- Windows PID-alive check: `tasklist /FI "PID eq <n>"` is the documented Windows API per Microsoft Learn (tasklist command reference).
- POSIX equivalent: `os.kill(pid, 0)` documented per Python `os` module docs (signal 0 = existence-test only, no signal sent).
- Cross-platform: `psutil.pid_exists(pid)` is the canonical Python wrapper if psutil is already a dependency — verify in `pyproject.toml` before adding.

## Codex Implementation Pass

Status: IMPLEMENTED_PENDING_OPERATOR_LIVE_REPRO. The startup-side sweep landed
without starting a broker or live session. Automated verification is green,
including a controlled temp-dir dead/live PID repro; the operator-only
Windows/live-session repro remains pending because this bounded task explicitly
forbade starting a live session.

What landed:

- Added [`scripts/tools/sweep_orphan_rings.py`](/home/joshd/canompx3/scripts/tools/sweep_orphan_rings.py)
  with a conservative tri-state PID probe:
  positive integer PID + provably dead => delete; live, corrupt JSON,
  missing PID, boolean/non-positive/non-int PID, or unknown PID status =>
  preserve.
- Added focused coverage in
  [`tests/test_scripts/test_sweep_orphan_rings.py`](/home/joshd/canompx3/tests/test_scripts/test_sweep_orphan_rings.py)
  for empty-dir, dead-PID delete, live-PID preserve, corrupt JSON preserve,
  missing PID preserve, ambiguous/non-positive PID preserve, and CLI
  `--ring-dir --dry-run`.
- Wired startup calls in [`START_BOT.bat`](/home/joshd/canompx3/START_BOT.bat)
  and [`scripts/run_live_session.py`](/home/joshd/canompx3/scripts/run_live_session.py)
  before instance-lock acquisition / normal live startup.
- No change was needed in
  [`trading_app/live/bar_ring.py`](/home/joshd/canompx3/trading_app/live/bar_ring.py);
  existing `writer_pid` metadata was already sufficient.

Verification:

- RED proof:
  `./.venv-wsl/bin/python -m pytest -q tests/test_scripts/test_sweep_orphan_rings.py`
  -> 7 failed with `ModuleNotFoundError: No module named 'scripts.tools.sweep_orphan_rings'`.
- GREEN:
  `./.venv-wsl/bin/python -m pytest -q tests/test_scripts/test_sweep_orphan_rings.py`
  -> 10 passed.
- Regression slice:
  `./.venv-wsl/bin/python -m pytest -q tests/test_scripts/test_sweep_orphan_rings.py tests/test_scripts/test_run_live_session_account_selection.py tests/test_scripts/test_run_live_session_account_id_sentinel.py tests/test_scripts/test_run_live_session_preflight.py tests/test_scripts/test_run_live_session_telemetry_maturity.py`
  -> 57 passed.
- Launcher slice:
  `./.venv-wsl/bin/python -m pytest -q tests/test_tools/test_codex_launcher_scripts.py`
  -> 6 passed.
- Ruff:
  `./.venv-wsl/bin/python -m ruff check scripts/tools/sweep_orphan_rings.py scripts/run_live_session.py tests/test_scripts/test_sweep_orphan_rings.py`
  -> pass.
- Py compile:
  `./.venv-wsl/bin/python -m py_compile scripts/tools/sweep_orphan_rings.py scripts/run_live_session.py`
  -> pass.
- Controlled temp-dir PID repro:
  actual POSIX PID probe deleted `DEAD.json` for pid `99999999` and preserved
  `LIVE.json` for `os.getpid()` without starting a broker/live session.
- Drift:
  `./.venv-wsl/bin/python pipeline/check_drift.py --quiet`
  -> clean, 164 passed, 20 advisory.

Skipped manual acceptance:

- SKIPPED — kill/restart orchestrator dead-PID repro — bounded task explicitly
  said "Do not start a broker/live session." Residual risk narrowed by the
  controlled POSIX dead/live PID repro; Windows `tasklist` branch still needs
  operator confirmation on the real path.
- SKIPPED — mid-session live-PID preserve smoke — same reason. Residual risk:
  startup hook could still need a platform-specific manual confirmation on a
  real Windows operator path, although the implementation preserves on any
  non-dead or unprovable PID status.
