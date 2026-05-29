"""Reap provably-stale Claude / MCP / hook processes for session-start hygiene.

PC + Claude responsiveness degrades over a working day because abandoned prior
sessions leave behind:
  - orphaned ``multiprocessing`` fork workers whose parent (a post-edit hook
    invocation) has already exited, and
  - duplicate MCP server generations (repo_state / research_catalog /
    strategy_lab / code-review-graph) started by earlier Claude sessions that
    did not shut down cleanly.

Stale MCP servers hold read-only ``gold.db`` handles and contend for the
single-writer DuckDB lock — the same root cause behind the recurring
"commit/drift is slow because a sibling holds the DB lock" incident class
(see memory/feedback_shared_index_db_lock_precommit_race_2026_05_28.md and
memory/feedback_stale_mcp_node_process_accumulation_slows_session_2026_05_29.md).

Safety contract (institutional-rigor.md §§ 3, 6 — fail-closed, no silent
failure):
  - DRY-RUN by default. Killing requires the explicit ``--apply`` flag.
  - A process is a kill candidate ONLY when it is *provably* stale:
      (a) a fork worker whose parent PID is dead, OR
      (b) a project process started BEFORE the current session lock's
          ``iso_started`` (a prior session's leftover).
  - Capital-path processes (live bot, dashboard, broker/execution, session
    launcher) are HARD-EXCLUDED from candidacy — matched by command-line
    signature — so this tool can never touch live trading.
  - The current process and its own ancestry are never candidates.
  - Fail-open: any enumeration / parse / kill error logs and continues; the
    tool never blocks a commit or a session start.

The decision logic is pure and isolated from OS calls so it is unit-tested
against a synthetic process table (no real process is killed in tests).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Command-line signatures that mark a process as CAPITAL-PATH — never a kill
# candidate under any circumstance. Matched case-insensitively as substrings.
CAPITAL_PATH_SIGNATURES = (
    "webhook_server",
    "bot_dashboard",
    "--demo",
    "--live",
    "--signal-only",
    "launch",
    "broker",
    "execution_engine",
    "session_orchestrator",
)

# Command-line signatures that mark a process as a PROJECT process eligible for
# staleness evaluation. A process must match one of these AND be provably stale
# to become a candidate. Matched case-insensitively as substrings.
PROJECT_PROCESS_SIGNATURES = (
    "repo_state_mcp_server",
    "research_catalog_mcp_server",
    "strategy_lab_mcp_server",
    "mcp_server.py",
    "code-review-graph",
    "pyright-langserver",
)

# Marker substring identifying a multiprocessing fork worker (orphan-detectable
# via its parent PID liveness).
FORK_WORKER_MARKER = "multiprocessing-fork"


@dataclass(frozen=True)
class ProcInfo:
    """A single process row. ``started`` is timezone-aware UTC or None if unknown."""

    pid: int
    ppid: int
    cmdline: str
    started: datetime | None


@dataclass(frozen=True)
class ReapDecision:
    """The verdict for one process."""

    proc: ProcInfo
    kill: bool
    reason: str


def _cmd_matches(cmdline: str, signatures: tuple[str, ...]) -> bool:
    low = cmdline.lower()
    return any(sig.lower() in low for sig in signatures)


def read_session_start(lock_path: Path) -> datetime | None:
    """Return the current session lock's ``iso_started`` as aware UTC, or None.

    None means "no readable lock" — under the fail-closed contract, callers
    treat None as "cannot prove anything is older than the session", so
    age-based candidacy (rule b) is disabled and only dead-parent orphans
    (rule a) remain killable.
    """
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except Exception:
        return None
    iso = ""
    try:
        iso = json.loads(raw).get("iso_started", "")
    except Exception:
        # session-start.py writes the lock with an unescaped Windows ``worktree``
        # path (e.g. "C:\\Users\\...") which is invalid JSON — the iso_started
        # field itself is well-formed (digits, ':', '-', '+', '.', 'T') and
        # never contains a backslash, so recover it by regex rather than failing
        # closed on every Windows session. (Lock-writer bug flagged separately;
        # see docs/runtime/stages/2026-05-29-stale-process-reaper.md.)
        m = re.search(r'"iso_started"\s*:\s*"([^"\\]+)"', raw)
        if m:
            iso = m.group(1)
    if not iso:
        return None
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts.astimezone(UTC)
    except Exception:
        return None


def decide(
    procs: list[ProcInfo],
    session_start: datetime | None,
    self_pid: int,
    self_ancestry: frozenset[int],
) -> list[ReapDecision]:
    """Pure decision function — no OS calls. Unit-tested directly.

    A process is killed only when ALL safety gates pass and exactly one
    staleness rule fires. The order of checks is fail-closed: any exclusion
    short-circuits to ``kill=False`` before staleness is even considered.
    """
    pids_alive = {p.pid for p in procs}
    decisions: list[ReapDecision] = []

    for p in procs:
        # Gate 0 — never touch ourselves or our own ancestry.
        if p.pid == self_pid or p.pid in self_ancestry:
            decisions.append(ReapDecision(p, False, "self/ancestry — excluded"))
            continue

        # Gate 1 — capital-path processes are untouchable, full stop.
        if _cmd_matches(p.cmdline, CAPITAL_PATH_SIGNATURES):
            decisions.append(ReapDecision(p, False, "capital-path signature — excluded"))
            continue

        # Rule (a) — orphan fork worker: parent PID is not in the live set.
        if FORK_WORKER_MARKER in p.cmdline.lower():
            if p.ppid not in pids_alive:
                decisions.append(ReapDecision(p, True, f"orphan fork worker (parent {p.ppid} dead)"))
            else:
                decisions.append(ReapDecision(p, False, f"fork worker, parent {p.ppid} alive — keep"))
            continue

        # Only project-signature processes are eligible past this point.
        if not _cmd_matches(p.cmdline, PROJECT_PROCESS_SIGNATURES):
            decisions.append(ReapDecision(p, False, "not a project process — ignored"))
            continue

        # Rule (b) — prior-session leftover: started before the current lock.
        if session_start is None:
            decisions.append(ReapDecision(p, False, "no session lock — age rule disabled (fail-closed)"))
            continue
        if p.started is None:
            decisions.append(ReapDecision(p, False, "start time unknown — cannot prove stale (fail-closed)"))
            continue
        if p.started < session_start:
            age_min = (session_start - p.started).total_seconds() / 60.0
            decisions.append(
                ReapDecision(p, True, f"prior-session MCP/helper (started {age_min:.0f} min before session)")
            )
        else:
            decisions.append(ReapDecision(p, False, "started after session — current session's own — keep"))

    return decisions


def _enumerate_processes() -> list[ProcInfo]:  # pragma: no cover - OS boundary
    """Enumerate processes via OS-native tools. Isolated so ``decide`` stays pure.

    Windows: PowerShell CIM (Win32_Process) emitting JSON.
    POSIX:   ``ps`` with pid,ppid,lstart,args.
    Any failure returns [] (fail-open — the reaper then finds no candidates).
    """
    if os.name == "nt":
        return _enumerate_windows()
    return _enumerate_posix()


def _enumerate_windows() -> list[ProcInfo]:  # pragma: no cover - OS boundary
    ps_script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python|node|code-review' } | "
        "ForEach-Object { [PSCustomObject]@{ pid=$_.ProcessId; ppid=$_.ParentProcessId; "
        "cmd=$_.CommandLine; start=$_.CreationDate } } | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return []
        raw = json.loads(out.stdout)
        if isinstance(raw, dict):
            raw = [raw]
        procs: list[ProcInfo] = []
        for r in raw:
            procs.append(
                ProcInfo(
                    pid=int(r.get("pid") or 0),
                    ppid=int(r.get("ppid") or 0),
                    cmdline=str(r.get("cmd") or ""),
                    started=_parse_wmi_date(r.get("start")),
                )
            )
        return procs
    except Exception:
        return []


def _parse_wmi_date(val: object) -> datetime | None:  # pragma: no cover - OS boundary
    """Parse a CIM ``CreationDate`` (ISO via ConvertTo-Json: '/Date(ms)/' or ISO string)."""
    if not val:
        return None
    s = str(val)
    try:
        if s.startswith("/Date(") and s.endswith(")/"):
            ms = int(s[6:-2].split("+")[0].split("-")[0])
            return datetime.fromtimestamp(ms / 1000.0, tz=UTC)
        ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return ts.astimezone(UTC) if ts.tzinfo else ts.replace(tzinfo=UTC)
    except Exception:
        return None


def _enumerate_posix() -> list[ProcInfo]:  # pragma: no cover - OS boundary
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,lstart=,args="],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode != 0:
            return []
        procs: list[ProcInfo] = []
        for line in out.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, ppid, rest = parts
            # lstart is a fixed 5-token date; args is everything after.
            rest_tokens = rest.split(None, 5)
            started = None
            cmdline = rest
            if len(rest_tokens) >= 6:
                date_str = " ".join(rest_tokens[:5])
                cmdline = rest_tokens[5]
                try:
                    started = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y").replace(tzinfo=UTC)
                except Exception:
                    started = None
            try:
                procs.append(ProcInfo(int(pid), int(ppid), cmdline, started))
            except ValueError:
                continue
        return procs
    except Exception:
        return []


def _self_ancestry(self_pid: int, procs: list[ProcInfo]) -> frozenset[int]:
    """Walk parent links up from ``self_pid`` so the live session is never reaped."""
    by_pid = {p.pid: p for p in procs}
    chain: set[int] = set()
    cur = self_pid
    seen: set[int] = set()
    while cur in by_pid and cur not in seen:
        seen.add(cur)
        parent = by_pid[cur].ppid
        chain.add(parent)
        cur = parent
    return frozenset(chain)


def _kill(pid: int) -> bool:  # pragma: no cover - OS boundary
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
        else:
            os.kill(pid, 9)
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reap provably-stale Claude/MCP/hook processes (fail-open).")
    parser.add_argument("--apply", action="store_true", help="Actually kill candidates (default: dry-run).")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary line.")
    args = parser.parse_args(argv)

    lock_path = PROJECT_ROOT / ".git" / ".claude.pid"
    session_start = read_session_start(lock_path)
    procs = _enumerate_processes()
    self_pid = os.getpid()
    ancestry = _self_ancestry(self_pid, procs)

    decisions = decide(procs, session_start, self_pid, ancestry)
    candidates = [d for d in decisions if d.kill]

    if not args.quiet:
        if session_start is None:
            print("WARN: no readable session lock — age-based reaping disabled, only dead-parent orphans eligible.")
        for d in candidates:
            print(f"  [{'KILL' if args.apply else 'DRY '}] pid={d.proc.pid} ppid={d.proc.ppid} :: {d.reason}")
            print(f"         {d.proc.cmdline[:120]}")

    killed = 0
    if args.apply:
        for d in candidates:
            if _kill(d.proc.pid):
                killed += 1

    action = f"killed {killed}" if args.apply else f"{len(candidates)} candidate(s) (dry-run — pass --apply to kill)"
    print(f"reap_stale_claude_processes: {len(procs)} procs scanned, {action}.")
    return 0  # fail-open: always succeed so we never block a hook/commit.


if __name__ == "__main__":
    sys.exit(main())
