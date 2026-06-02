"""Tests for `scripts/tools/worktree_guard.py` (the canonical lease module).

Strategy: every test runs against a synthetic git worktree under `tmp_path`
so the developer's real `<git-dir>/.claude.worktree.lock` is never touched.
We `subprocess.run(["git", "init", "-q"])` once per test (fast on Windows
because tmp_path is per-test).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))

import worktree_guard as wg  # noqa: E402  # type: ignore[import-not-found]


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _release_all_for_cwd(cwd: Path) -> None:
    """Belt-and-braces cleanup of any FileLock we still hold for this cwd."""
    lk = wg.lock_path(cwd)
    if lk is None:
        return
    held = wg._LIVE_LOCKS.pop(str(lk), None)
    if held is not None:
        try:
            held.release(force=True)
        except OSError:
            pass


@pytest.fixture
def repo(tmp_path: Path):
    repo_path = _init_repo(tmp_path)
    yield repo_path
    _release_all_for_cwd(repo_path)


def test_skipped_outside_git_repo(tmp_path: Path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    status_str, _, _ = wg.acquire(not_a_repo)
    assert status_str == "skipped"


def test_acquire_fresh_writes_sidecar(repo: Path):
    status_str, payload, _ = wg.acquire(repo)
    assert status_str == "acquired"
    assert payload is not None and payload["pid"] == os.getpid()
    ls = wg.lease_path(repo)
    assert ls is not None and ls.exists()
    data = json.loads(ls.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["worktree"] == str(repo.resolve())


def test_acquire_then_refresh_same_pid(repo: Path):
    s1, _, _ = wg.acquire(repo)
    s2, payload, _ = wg.acquire(repo)
    assert s1 == "acquired"
    assert s2 == "refreshed"
    assert payload is not None and payload["pid"] == os.getpid()


def test_blocked_by_live_peer(repo: Path):
    """A DIFFERENT session with a fresh heartbeat AND a LIVE ppid blocks.

    The peer's liveness anchor is its recorded ppid; we pin it to THIS test
    process's pid (definitely alive) so the block path is deterministic.
    """
    s1, _, _ = wg.acquire(repo, pid=99991, session_id="peerA", ppid=os.getpid())
    assert s1 == "acquired"

    s2, payload, msg = wg.acquire(repo, pid=99992, session_id="callerB", ppid=os.getpid())
    assert s2 == "blocked", msg
    assert payload is not None
    assert payload["session_id"] == "peerA"


def test_reclaims_lease_with_dead_ppid(repo: Path):
    """A peer lease whose recorded ppid is dead is reclaimed, not blocking.

    This is the /clear-restart safety property: the prior session's Claude
    process is gone, so its ppid reads dead and the new session takes over.
    """
    wg.acquire(repo, pid=99991, session_id="deadPeer", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # pid in a range no real process will hold
    ls.write_text(json.dumps(data), encoding="utf-8")

    status_str, payload, msg = wg.acquire(repo, pid=99992, session_id="freshSession", ppid=os.getpid())
    assert status_str == "reclaimed", msg
    assert payload is not None and payload["session_id"] == "freshSession"


def test_stale_heartbeat_with_live_ppid_still_blocks(repo: Path):
    """A stale heartbeat does NOT reclaim while the holder's ppid is alive.

    The heartbeat only refreshes on PreToolUse; a single long-running tool call
    (e.g. check_drift.py at ~180s, past the 90s window) legitimately stops
    refreshing it. ppid-liveness is authoritative — a provably-alive session
    keeps its lease, so a peer must BLOCK, not steal it mid-call. (Adversarial
    audit MED finding, 2026-05-30.)
    """
    wg.acquire(repo, pid=99991, session_id="busyPeer", ppid=os.getpid())  # alive ppid
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 30)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    status_str, payload, msg = wg.acquire(repo, pid=99992, session_id="peerB", ppid=os.getpid())
    assert status_str == "blocked", msg
    assert payload is not None and payload["session_id"] == "busyPeer"


def test_reclaims_lease_stale_heartbeat_and_dead_ppid(repo: Path):
    """Reclaim requires BOTH: heartbeat stale AND ppid dead (holder truly gone)."""
    wg.acquire(repo, pid=99991, session_id="gonePeer", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 30)
    data["iso_heartbeat"] = backdated.isoformat()
    data["ppid"] = 2147480000  # dead
    ls.write_text(json.dumps(data), encoding="utf-8")

    status_str, _, msg = wg.acquire(repo, pid=99992, session_id="freshSession", ppid=os.getpid())
    assert status_str == "reclaimed", msg


def test_same_session_refreshes_not_blocks(repo: Path):
    """The SAME session_id re-acquiring is a refresh, never a block."""
    s1, _, _ = wg.acquire(repo, pid=11111, session_id="sameSession", ppid=os.getpid())
    s2, payload, msg = wg.acquire(repo, pid=22222, session_id="sameSession", ppid=os.getpid())
    assert s1 == "acquired"
    assert s2 == "refreshed", msg
    assert payload is not None and payload["session_id"] == "sameSession"


def test_status_shows_lease_state(repo: Path):
    wg.acquire(repo)
    snap = wg.status(repo)
    assert snap["in_git_repo"] is True
    assert snap["lease_present"] is True
    assert snap["holder_pid"] == os.getpid()
    assert snap["current_is_holder"] is True


def test_status_no_lease(repo: Path):
    snap = wg.status(repo)
    assert snap["in_git_repo"] is True
    assert snap["lease_present"] is False
    # The OS-lock `is_locked` field is gone in the heartbeat model — ownership
    # is the sidecar, not a phantom subprocess lock. No lease → no holder.
    assert "holder_pid" not in snap


def test_release_by_current_pid(repo: Path):
    wg.acquire(repo)
    status_str, msg = wg.release(repo)
    assert status_str == "released", msg
    assert wg.read_lease(repo) is None


def test_release_refuses_other_pid(repo: Path):
    wg.acquire(repo, pid=42424)
    status_str, msg = wg.release(repo, pid=99999, force=False)
    assert status_str == "refused", msg


def test_force_release_always_clears(repo: Path):
    wg.acquire(repo, pid=42424)
    status_str, msg = wg.release(repo, pid=99999, force=True)
    assert status_str == "released", msg
    assert wg.read_lease(repo) is None


def test_release_when_not_held(repo: Path):
    status_str, _ = wg.release(repo)
    assert status_str == "not-held"


def test_sidecar_stale_heartbeat_reported(repo: Path):
    wg.acquire(repo)
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    # Backdate the heartbeat past the staleness threshold.
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 60)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    snap = wg.status(repo)
    assert snap["sidecar_stale"] is True
    assert snap["heartbeat_age_seconds"] >= wg.STALE_HEARTBEAT_SECONDS


def test_cli_status_default(repo: Path, capsys: pytest.CaptureFixture):
    rc = wg.main(["--cwd", str(repo), "--status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "worktree-guard:" in out


def test_cli_acquire_then_status_then_release(repo: Path, capsys: pytest.CaptureFixture):
    rc1 = wg.main(["--cwd", str(repo), "--acquire"])
    rc2 = wg.main(["--cwd", str(repo), "--status"])
    rc3 = wg.main(["--cwd", str(repo), "--force-release"])
    assert rc1 == 0
    assert rc2 == 0
    assert rc3 == 0
    out = capsys.readouterr().out
    assert "acquired" in out or "refreshed" in out
    assert "released" in out


def test_cli_json_output(repo: Path, capsys: pytest.CaptureFixture):
    wg.main(["--cwd", str(repo), "--acquire", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["action"] == "acquire"
    assert data["status"] in ("acquired", "refreshed")


def test_iso_age_seconds_parses_z_suffix():
    # Both `+00:00` and `Z` should parse to the same age.
    base = datetime.now(UTC) - timedelta(seconds=120)
    iso_offset = base.isoformat()
    iso_z = base.isoformat().replace("+00:00", "Z")
    a = wg._iso_age_seconds(iso_offset)
    b = wg._iso_age_seconds(iso_z)
    assert a is not None and b is not None
    assert abs(a - b) < 1


def test_iso_age_seconds_handles_garbage():
    assert wg._iso_age_seconds("") is None
    assert wg._iso_age_seconds("not-a-date") is None


def test_lease_path_isolation_across_worktrees(tmp_path: Path):
    """Two separate git repos must have separate lease paths."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=a, check=True)
    subprocess.run(["git", "init", "-q"], cwd=b, check=True)
    pa = wg.lease_path(a)
    pb = wg.lease_path(b)
    assert pa is not None and pb is not None
    assert pa != pb


def test_write_lease_is_atomic_torn_read_safe(repo: Path) -> None:
    """_write_lease must use atomic os.replace so a concurrent reader never
    sees partial JSON (CRITICAL fix, adversarial audit 2026-05-30)."""
    ls = wg.lease_path(repo)
    assert ls is not None

    # Write a valid lease via the public API.
    wg.acquire(repo, session_id="test-session")
    assert ls.exists()

    # Verify the tmp file is gone after write (atomic replace completed).
    tmp = ls.with_suffix(".json.tmp")
    assert not tmp.exists(), "tmp file must be cleaned up by os.replace"

    # Content must be valid JSON (not truncated).
    data = json.loads(ls.read_text(encoding="utf-8"))
    assert "session_id" in data
    assert data["schema"] >= 3  # schema 3+ includes ppid_create_time on Windows


def test_schema3_ppid_create_time_written_on_windows(repo: Path) -> None:
    """On Windows, ppid_create_time must be stored in the lease (schema 3 fix)."""
    if os.name != "nt":
        pytest.skip("Windows-only — PID reuse detection via GetProcessTimes")

    wg.acquire(repo, session_id="test-win-session")
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    assert "ppid_create_time" in data, "ppid_create_time must be stored on Windows"
    assert isinstance(data["ppid_create_time"], int)
    assert data["ppid_create_time"] > 0


# --------------------------------------------------------------------------- #
# Heartbeat-marries-lease: a fresh peer BEAT overrides a false-dead ppid so the
# single lease is BLOCKED, not reclaimed session-to-session. This is the Windows
# concurrency-leak fix (3 live terminals slipping past the lease gate because
# OpenProcess momentarily read the holder's ppid as dead).
# --------------------------------------------------------------------------- #
def _write_beat(repo: Path, session_id: str, *, cwd: Path | None = None, age_s: float = 0.0) -> Path:
    """Stamp a `<session_id>.beat` into the repo's git-common-dir heartbeat dir,
    mirroring `.claude/hooks/session-heartbeat.py`. `age_s` backdates the mtime."""
    import time

    common = wg._git_common_dir(repo)
    assert common is not None
    beat_dir = common / wg.HEARTBEAT_DIRNAME
    beat_dir.mkdir(parents=True, exist_ok=True)
    bf = beat_dir / f"{session_id}.beat"
    bf.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "cwd": str((cwd or repo).resolve()),
                "branch": "main",
                "pid": 4242,
                "ts": time.time() - age_s,
            }
        ),
        encoding="utf-8",
    )
    if age_s:
        old = time.time() - age_s
        os.utime(bf, (old, old))
    return bf


def test_fresh_peer_beat_blocks_reclaim_when_ppid_false_dead(repo: Path):
    """THE FIX: holder ppid reads dead, but a DIFFERENT session is beating in
    this tree right now → block, not reclaim. Without the heartbeat cross-check
    this returned 'reclaimed' and three live sessions coexisted (the leak)."""
    wg.acquire(repo, pid=99991, session_id="holderA", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead pid → ppid probe says holder gone
    ls.write_text(json.dumps(data), encoding="utf-8")

    # A peer (NOT the caller) is heartbeating in this exact tree, seconds old.
    _write_beat(repo, "holderA", cwd=repo, age_s=2.0)

    status_str, payload, msg = wg.acquire(repo, pid=99992, session_id="callerB", ppid=os.getpid())
    assert status_str == "blocked", msg
    assert payload is not None and payload["session_id"] == "holderA"


def test_self_beat_only_still_reclaims(repo: Path):
    """No false self-block: if the ONLY fresh beat is the caller's own session,
    a dead-ppid stale lease must still reclaim (the /clear-restart path)."""
    wg.acquire(repo, pid=99991, session_id="oldSelf", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 30)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    # Only the CALLER's own beat exists — must be excluded, so reclaim proceeds.
    _write_beat(repo, "freshSelf", cwd=repo, age_s=1.0)

    status_str, _, msg = wg.acquire(repo, pid=99992, session_id="freshSelf", ppid=os.getpid())
    assert status_str == "reclaimed", msg


def test_stale_peer_beat_does_not_block_reclaim(repo: Path):
    """A peer beat OLDER than the live window is a crashed session — it must not
    keep the lease alive. Dead ppid + stale heartbeat + stale beat → reclaim."""
    wg.acquire(repo, pid=99991, session_id="crashedPeer", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 30)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    _write_beat(repo, "crashedPeer", cwd=repo, age_s=wg.HEARTBEAT_LIVE_WINDOW_SECONDS + 120)

    status_str, _, msg = wg.acquire(repo, pid=99992, session_id="newSession", ppid=os.getpid())
    assert status_str == "reclaimed", msg


def test_sibling_worktree_beat_does_not_block(repo: Path, tmp_path: Path):
    """A fresh beat whose cwd is a DIFFERENT worktree is sanctioned parallel
    work, not a same-tree collision — it must not block a reclaim here."""
    wg.acquire(repo, pid=99991, session_id="holderC", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 30)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    sibling = tmp_path / "sibling-tree"
    sibling.mkdir()
    _write_beat(repo, "siblingPeer", cwd=sibling, age_s=2.0)  # fresh, but elsewhere

    status_str, _, msg = wg.acquire(repo, pid=99992, session_id="newSession", ppid=os.getpid())
    assert status_str == "reclaimed", msg


def test_fresh_peer_beat_surfaces_in_status(repo: Path):
    """status() must report peer_live=True from a fresh peer beat even when the
    holder ppid reads dead — the exact signal that exposed the live leak."""
    wg.acquire(repo, pid=99991, session_id="holderD", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead
    ls.write_text(json.dumps(data), encoding="utf-8")
    _write_beat(repo, "holderD", cwd=repo, age_s=2.0)

    snap = wg.status(repo)
    assert snap["fresh_peer_heartbeat"] is True
    assert snap["peer_live"] is True


# ---------------------------------------------------------------------------
# Blocking-window vs observability-window split (n=1 false-block 2026-06-02).
# A session that ended <10min ago (the `/clear` restart) left a beat the 600s
# window counted as a "live peer", false-blocking the fresh restart against its
# own dead predecessor. The blocking path now uses BLOCKING_HEARTBEAT_WINDOW_
# SECONDS (90s); status() keeps the 600s observability window.
# ---------------------------------------------------------------------------


def test_blocking_window_is_tighter_than_observability_window():
    """The two windows are deliberately different. If they ever converge, the
    /clear false-block returns — this guards the constant relationship."""
    assert wg.BLOCKING_HEARTBEAT_WINDOW_SECONDS == wg.STALE_HEARTBEAT_SECONDS
    assert wg.BLOCKING_HEARTBEAT_WINDOW_SECONDS < wg.HEARTBEAT_LIVE_WINDOW_SECONDS


def test_just_cleared_predecessor_beat_does_not_false_block(repo: Path):
    """THE FIX. A DIFFERENT session whose beat is in the 90-600s band (ended a
    few minutes ago, the /clear case) with a dead-ppid stale lease must RECLAIM,
    not block. Before the window split this returned 'blocked' every /clear."""
    wg.acquire(repo, pid=99991, session_id="oldClearedSelf", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead — holder process gone
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 30)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    # Predecessor's residual beat: older than the 90s blocking window but inside
    # the 600s observability window — exactly the /clear-restart signature.
    mid_age = wg.BLOCKING_HEARTBEAT_WINDOW_SECONDS + 60
    assert mid_age < wg.HEARTBEAT_LIVE_WINDOW_SECONDS  # genuinely in the band
    _write_beat(repo, "oldClearedSelf", cwd=repo, age_s=mid_age)

    status_str, _, msg = wg.acquire(repo, pid=99992, session_id="newSession", ppid=os.getpid())
    assert status_str == "reclaimed", msg


def test_genuinely_live_peer_within_90s_still_blocks(repo: Path):
    """The fix must NOT weaken real two-session protection: a peer beating within
    the 90s blocking window (live sessions re-beat every <=20s) still blocks."""
    wg.acquire(repo, pid=99991, session_id="livePeer", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead ppid → must rely on the beat to block
    ls.write_text(json.dumps(data), encoding="utf-8")

    # A live peer's newest beat is seconds old (throttle is 20s) — inside 90s.
    _write_beat(repo, "livePeer", cwd=repo, age_s=15.0)

    status_str, payload, msg = wg.acquire(repo, pid=99992, session_id="callerX", ppid=os.getpid())
    assert status_str == "blocked", msg
    assert payload is not None and payload["session_id"] == "livePeer"


def test_status_keeps_600s_observability_window(repo: Path):
    """status() must still SURFACE a 90-600s peer beat (peer_live True) — the
    operator-visible signal is broader than the block trigger on purpose."""
    wg.acquire(repo, pid=99991, session_id="recentPeer", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead
    ls.write_text(json.dumps(data), encoding="utf-8")
    # Beat in the 90-600s band: below the block window, inside observability.
    _write_beat(repo, "recentPeer", cwd=repo, age_s=wg.BLOCKING_HEARTBEAT_WINDOW_SECONDS + 60)

    snap = wg.status(repo)
    assert snap["fresh_peer_heartbeat"] is True  # 600s window still sees it


def test_reclaim_prunes_dead_holder_beat(repo: Path):
    """On reclaim, the dead holder's residual beat is deleted so it can't trip
    the blocking window on the NEXT /clear (self-cleaning, belt-and-braces)."""
    wg.acquire(repo, pid=99991, session_id="deadHolder", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 30)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    beat = _write_beat(repo, "deadHolder", cwd=repo, age_s=wg.BLOCKING_HEARTBEAT_WINDOW_SECONDS + 60)
    assert beat.exists()

    status_str, _, msg = wg.acquire(repo, pid=99992, session_id="newSession", ppid=os.getpid())
    assert status_str == "reclaimed", msg
    assert not beat.exists(), "dead holder's beat should be pruned on reclaim"


def test_reclaim_does_not_prune_a_live_callers_own_beat(repo: Path):
    """Prune targets ONLY the reclaimed holder's beat — never the caller's own,
    even if the caller already has a beat on disk."""
    wg.acquire(repo, pid=99991, session_id="goneHolder", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # dead
    backdated = datetime.now(UTC) - timedelta(seconds=wg.STALE_HEARTBEAT_SECONDS + 30)
    data["iso_heartbeat"] = backdated.isoformat()
    ls.write_text(json.dumps(data), encoding="utf-8")

    holder_beat = _write_beat(repo, "goneHolder", cwd=repo, age_s=wg.BLOCKING_HEARTBEAT_WINDOW_SECONDS + 60)
    caller_beat = _write_beat(repo, "newSession", cwd=repo, age_s=5.0)

    status_str, _, _ = wg.acquire(repo, pid=99992, session_id="newSession", ppid=os.getpid())
    assert status_str == "reclaimed"
    assert not holder_beat.exists(), "holder beat pruned"
    assert caller_beat.exists(), "caller's own beat must survive"
