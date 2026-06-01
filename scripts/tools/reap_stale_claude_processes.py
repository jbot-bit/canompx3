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
        # closed on every Windows session. (Root cause: session-start.py writes
        # the lock with an unescaped Windows worktree path; tracked separately.)
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
    reap_duplicates: bool = False,
    current_launcher_pid: int | None = None,
    launcher_of: dict[int, int] | None = None,
    launcher_started: dict[int, datetime] | None = None,
) -> list[ReapDecision]:
    """Pure decision function — no OS calls. Unit-tested directly.

    A process is killed only when ALL safety gates pass and exactly one
    staleness rule fires. The order of checks is fail-closed: any exclusion
    short-circuits to ``kill=False`` before staleness is even considered.

    ``reap_duplicates`` gates the aggressive, lock-independent Rule (c). It is
    OFF by default because "newest pair" is not provably "the current session's
    pair" when the session lock is stale — auto/session-start use stays
    conservative (orphans + lock-proven-stale only); operators opt in for the
    aggressive duplicate sweep.

    ``current_launcher_pid`` is the live session's launcher PID; ``launcher_of``
    maps each project process PID -> its ancestral launcher PID; ``launcher_started``
    maps a launcher PID -> its start time. Together they drive Rule (d): an MCP
    server whose launcher is a DIFFERENT launcher than ours AND whose launcher
    *predates our session lock* is a prior-session leftover (the launcher never
    exited, so Rules a/b miss it). Rule (d) is fail-closed on TWO axes:
      - disabled entirely when ``current_launcher_pid`` is unknown (we cannot
        prove a server belongs to a prior session without knowing which is ours);
      - a different launcher is reaped ONLY when it provably predates the session
        lock. A different launcher that started AFTER our lock is a CONCURRENT
        LIVE PEER (another active session / worktree) — NEVER reaped. This is the
        multi-session safety the lease-incident class demands (see
        memory/feedback_dead_pid_fresh_heartbeat_lease_trust_heartbeat_not_pid_*).
    The root cause this targets: abandoned Claude sessions (observed 14-65h old)
    keep full MCP server sets alive as children, so every child reports
    ``parent alive`` and evades Rules a/b.
    """
    pids_alive = {p.pid for p in procs}
    launcher_of = launcher_of or {}
    launcher_started = launcher_started or {}
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

        # Rule (d) — prior-session launcher: the MCP server's ancestral launcher
        # is a DIFFERENT launcher than ours AND that launcher predates our session
        # lock. This is the dominant leak (abandoned sessions keep MCP children
        # "non-orphan"). Fail-closed on two axes: we must know our own launcher,
        # AND a different launcher is only reaped when provably lock-stale — a
        # newer different launcher is a concurrent LIVE PEER and is left alone.
        if current_launcher_pid is not None:
            owner = launcher_of.get(p.pid)
            if owner is not None and owner == current_launcher_pid:
                decisions.append(ReapDecision(p, False, "current session launcher — keep"))
                continue
            if owner is not None and owner != current_launcher_pid:
                owner_start = launcher_started.get(owner)
                if session_start is not None and owner_start is not None and owner_start < session_start:
                    age_min = (session_start - owner_start).total_seconds() / 60.0
                    decisions.append(
                        ReapDecision(
                            p,
                            True,
                            f"prior-session launcher {owner} (started {age_min:.0f} min before session) != current {current_launcher_pid}",
                        )
                    )
                    continue
                if session_start is not None and owner_start is not None and owner_start >= session_start:
                    decisions.append(
                        ReapDecision(p, False, f"concurrent live-peer launcher {owner} (newer than session) — keep")
                    )
                    continue
                # Owner start unknown or no lock — cannot prove the peer is stale.
                # Fall through to lock-based Rule (b) rather than kill on a guess.
            # owner is None: launcher unresolved — fall through to lock-based Rule (b).

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

    # Rule (c) — duplicate-generation reaping (lock-independent hardening).
    # A correctly-running MCP server is at most one stub→base PID pair (see
    # memory/feedback_venv_stub_base_interpreter_pid_pairs_not_a_bug_2026_05_27.md).
    # When a server signature has MORE than one pair alive, earlier generations
    # are leftovers from prior sessions that a stale session lock cannot date.
    # Keep the newest EXPECTED_PAIR processes per signature; mark the rest kill.
    # This runs only on processes Rule (b) left as keep — exclusions already won.
    # Opt-in: see ``reap_duplicates`` docstring (avoids killing a live-connected
    # sibling MCP server when the session lock is stale).
    if reap_duplicates:
        _apply_duplicate_generation_rule(decisions)
    return decisions


# A single logical MCP/LSP server presents as a stub→base interpreter PID pair.
EXPECTED_PAIR = 2


def _apply_duplicate_generation_rule(decisions: list[ReapDecision]) -> None:
    """Mutate ``decisions`` in place: among still-kept project processes, keep
    the newest ``EXPECTED_PAIR`` per server signature, mark older ones kill.

    Processes with an unknown start time are conservatively kept (cannot rank).
    """
    # Each entry: (epoch_start, decision_index). epoch captured here so the
    # sort key carries a known float, not an Optional member access.
    by_sig: dict[str, list[tuple[float, int]]] = {}
    for i, d in enumerate(decisions):
        if d.kill or d.proc.started is None:
            continue
        sig = _server_signature(d.proc.cmdline)
        if sig is None:
            continue
        by_sig.setdefault(sig, []).append((d.proc.started.timestamp(), i))

    for sig, entries in by_sig.items():
        if len(entries) <= EXPECTED_PAIR:
            continue
        # Newest first; keep the first EXPECTED_PAIR, reap the rest.
        entries.sort(reverse=True)
        for _epoch, stale_i in entries[EXPECTED_PAIR:]:
            p = decisions[stale_i].proc
            decisions[stale_i] = ReapDecision(p, True, f"duplicate generation of '{sig}' (newer pair alive)")


def _server_signature(cmdline: str) -> str | None:
    """The PROJECT_PROCESS signature this cmdline matches, for grouping pairs."""
    low = cmdline.lower()
    for sig in PROJECT_PROCESS_SIGNATURES:
        if sig.lower() in low:
            return sig
    return None


# cmdline markers for the Claude Code CLI launcher process (the parent that owns
# an MCP server tree). Matched case-insensitively as substrings. Windows packs
# it as ``claude.exe``; POSIX installs expose ``.../bin/claude`` (a node shim) —
# both must match so Rule (d) works cross-platform. ``claude.exe`` is checked as
# a distinct token so the Windows path is unambiguous; ``/claude`` and the
# trailing-``claude`` forms catch the POSIX bin shim without matching unrelated
# words like "claudette" mid-cmdline.
CLAUDE_LAUNCHER_MARKERS = ("claude.exe", "/bin/claude", "\\bin\\claude")


def _is_claude_launcher(cmdline: str) -> bool:
    """True if ``cmdline`` looks like the Claude Code CLI launcher process."""
    low = cmdline.lower()
    if any(m in low for m in CLAUDE_LAUNCHER_MARKERS):
        return True
    # Bare-invocation fallback: the launcher is the executable token "claude"
    # (e.g. a PATH-resolved ``claude`` with no directory). Guard against
    # substring false-positives ("claudette") by requiring a word boundary.
    return bool(re.search(r"(?:^|[\s/\\\"'])claude(?:\.exe)?(?:[\s\"']|$)", low))


def _claude_launcher_pid(pid: int, procs: list[ProcInfo]) -> int | None:
    """Walk parent links up from ``pid`` to the nearest Claude launcher ancestor.

    Returns that launcher's PID, or None if no Claude launcher is found in the
    ancestry (e.g. an MCP server orphaned from its launcher, or a non-Claude
    parent tree). Cycle-safe and bounded by the process count.
    """
    by_pid = {p.pid: p for p in procs}
    cur = pid
    seen: set[int] = set()
    while cur in by_pid and cur not in seen:
        seen.add(cur)
        proc = by_pid[cur]
        if _is_claude_launcher(proc.cmdline):
            return proc.pid
        cur = proc.ppid
    return None


def _build_launcher_map(
    procs: list[ProcInfo],
) -> tuple[int | None, dict[int, int], dict[int, datetime]]:
    """Resolve (current_launcher_pid, launcher_of, launcher_started).

    ``current_launcher_pid`` is the Claude launcher ancestor of THIS process
    (``os.getpid()``) — the live session's launcher. ``launcher_of`` maps every
    project-signature MCP process to its ancestral launcher PID, so ``decide``
    can tell our own MCP children from a prior session's. ``launcher_started``
    maps each owning launcher PID to its start time, so Rule (d) can require a
    different launcher to be lock-stale before reaping its children (never a
    concurrent live peer). Pure-ish: takes the enumerated table; the only OS
    call is ``os.getpid()``.
    """
    self_pid = os.getpid()
    current_launcher = _claude_launcher_pid(self_pid, procs)
    by_pid = {p.pid: p for p in procs}
    launcher_of: dict[int, int] = {}
    launcher_started: dict[int, datetime] = {}
    for p in procs:
        if _cmd_matches(p.cmdline, PROJECT_PROCESS_SIGNATURES):
            owner = _claude_launcher_pid(p.pid, procs)
            if owner is not None:
                launcher_of[p.pid] = owner
                owner_proc = by_pid.get(owner)
                if owner_proc is not None and owner_proc.started is not None:
                    launcher_started[owner] = owner_proc.started
    return current_launcher, launcher_of, launcher_started


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
    # 'claude' MUST be in the filter: Rule (d) walks parent links to the
    # claude.exe launcher ancestor, so the launcher process must appear in the
    # enumerated table or _claude_launcher_pid always returns None (Rule d dead).
    ps_script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python|node|code-review|claude' } | "
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
    parser.add_argument(
        "--reap-duplicates",
        action="store_true",
        help=(
            "Also reap older duplicate MCP/LSP generations (keep newest pair per server). "
            "OFF by default — may kill a live-connected sibling when the session lock is stale; "
            "use for a manual deep clean, not session-start auto-reaping."
        ),
    )
    args = parser.parse_args(argv)

    lock_path = PROJECT_ROOT / ".git" / ".claude.pid"
    session_start = read_session_start(lock_path)
    procs = _enumerate_processes()
    self_pid = os.getpid()
    ancestry = _self_ancestry(self_pid, procs)
    current_launcher_pid, launcher_of, launcher_started = _build_launcher_map(procs)

    decisions = decide(
        procs,
        session_start,
        self_pid,
        ancestry,
        reap_duplicates=args.reap_duplicates,
        current_launcher_pid=current_launcher_pid,
        launcher_of=launcher_of,
        launcher_started=launcher_started,
    )
    candidates = [d for d in decisions if d.kill]

    if not args.quiet:
        if session_start is None:
            print("WARN: no readable session lock — age-based reaping disabled, only dead-parent orphans eligible.")
        if current_launcher_pid is None:
            print(
                "WARN: could not resolve current claude.exe launcher — Rule (d) prior-session-launcher reaping disabled (fail-closed)."
            )
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
