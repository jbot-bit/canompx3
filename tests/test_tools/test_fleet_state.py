"""Class-matrix + delegation tests for the canonical fleet-state resolver.

Covers all 6 classifications and proves fleet_state DELEGATES liveness to the
canonical worktree_guard path rather than re-encoding it. Git and liveness are
monkeypatched so the tests are deterministic and independent of the live fleet.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from scripts.tools import active_plan as ap
from scripts.tools import fleet_state as fs
from scripts.tools import worktree_guard as wg

# ── _count_dirty: real vs churn ──────────────────────────────────────────────


def test_count_dirty_excludes_churn():
    porcelain = " M HANDOFF.md\n M live_journal.db\n M pipeline/dst.py\n M docs/runtime/active_plan.md"
    total, real, real_nondel = fs._count_dirty(porcelain)
    assert total == 4
    assert real == 1  # only pipeline/dst.py is real work
    assert real_nondel == 1  # the M is a non-deletion edit


def test_count_dirty_rename_counts_destination():
    porcelain = "R  old.py -> trading_app/new.py"
    total, real, real_nondel = fs._count_dirty(porcelain)
    assert total == 1
    assert real == 1
    assert real_nondel == 1  # a rename is a non-deletion


def test_count_dirty_deletions_count_real_but_not_real_nondel():
    # The load-bearing distinction: a hollow tree's deletions inflate `real` but
    # are NOT work-at-risk (real_nondel). A genuine M among them IS.
    porcelain = "\n".join([" D gone_a.py", " D gone_b.py", " M kept.py"])
    total, real, real_nondel = fs._count_dirty(porcelain)
    assert total == 3
    assert real == 3  # all three are non-churn changes
    assert real_nondel == 1  # only kept.py is a non-deletion (work-at-risk)


def test_count_dirty_untracked_scaffolding_is_not_work_at_risk():
    # The poisoning-tree case (live 2026-06-06): a gutted tree's remaining
    # non-deletion lines are UNTRACKED runtime dirs (`?? .claude/`), NOT lost
    # work. They must NOT count as real_nondel, else the reapable tree flips to
    # NEEDS_FINISH and Stage 2 can't clean it up.
    porcelain = "\n".join([" D gone.py", "?? .claude/", "?? .codex/", "?? .canompx3-runtime/"])
    total, real, real_nondel = fs._count_dirty(porcelain)
    assert total == 4
    assert real == 4  # all non-churn changes (observability)
    assert real_nondel == 0  # untracked scaffolding is NOT work-at-risk


def test_count_dirty_tracked_edit_is_work_at_risk_but_untracked_is_not():
    # A TRACKED modification IS work-at-risk; an untracked file is not.
    porcelain = "\n".join([" M tracked_edit.py", "?? brand_new_untracked.py"])
    total, real, real_nondel = fs._count_dirty(porcelain)
    assert total == 2
    assert real == 2
    assert real_nondel == 1  # only the tracked ` M` counts


def test_count_dirty_substring_churn_is_not_churn():
    # A real source path that merely CONTAINS a churn name must NOT be churn
    # (segment match, not substring) — else a genuine edit is silently dropped.
    porcelain = " M tests/test_live_journal.db_helpers.py"
    total, real, real_nondel = fs._count_dirty(porcelain)
    assert total == 1
    assert real == 1  # NOT swallowed by the live_journal.db churn entry
    assert real_nondel == 1


def test_count_dirty_empty():
    assert fs._count_dirty("") == (0, 0, 0)


# ── _classify: the precedence ladder ─────────────────────────────────────────


def test_classify_live_wins_over_everything():
    cls, reasons = fs._classify(
        live=True, hollow=True, merged=True, real_dirty=5, real_nondel_dirty=5, unpushed=3, behind=0
    )
    assert cls == fs.CLASS_LIVE
    assert any("live" in r for r in reasons)


def test_classify_hollow_before_needs_finish():
    # Hollow tree with deletion-noise must NOT be read as work-at-risk.
    cls, _ = fs._classify(
        live=False, hollow=True, merged=False, real_dirty=0, real_nondel_dirty=0, unpushed=0, behind=0
    )
    assert cls == fs.CLASS_HOLLOW


def test_classify_unpushed_is_needs_finish():
    cls, reasons = fs._classify(
        live=False, hollow=False, merged=True, real_dirty=0, real_nondel_dirty=0, unpushed=4, behind=0
    )
    # Unpushed work outranks merged — merged-but-unpushed still needs finishing.
    assert cls == fs.CLASS_NEEDS_FINISH
    assert any("unpushed" in r for r in reasons)


def test_classify_real_dirty_is_needs_finish():
    cls, _ = fs._classify(
        live=False, hollow=False, merged=False, real_dirty=2, real_nondel_dirty=2, unpushed=0, behind=0
    )
    assert cls == fs.CLASS_NEEDS_FINISH


def test_classify_merged_clean_is_mergeable():
    cls, reasons = fs._classify(
        live=False, hollow=False, merged=True, real_dirty=0, real_nondel_dirty=0, unpushed=0, behind=0
    )
    assert cls == fs.CLASS_MERGED
    assert any("prune" in r for r in reasons)


def test_classify_clean_current_is_healthy():
    cls, _ = fs._classify(
        live=False, hollow=False, merged=False, real_dirty=0, real_nondel_dirty=0, unpushed=0, behind=5
    )
    assert cls == fs.CLASS_HEALTHY


def test_classify_clean_far_behind_is_stale():
    cls, reasons = fs._classify(
        live=False, hollow=False, merged=False, real_dirty=0, real_nondel_dirty=0, unpushed=0, behind=500
    )
    assert cls == fs.CLASS_STALE
    assert any("behind" in r for r in reasons)


# ── fleet_state end-to-end with monkeypatched git/liveness ──────────────────


class _FakeInfo:
    def __init__(self, path, branch, head):
        self.path = path
        self.branch = branch
        self.head = head


def _patch_fleet(monkeypatch, *, infos, porcelains, live_map, merged_map, divergence):
    """Wire fleet_state's collaborators to fixtures.

    porcelains: {path -> porcelain str}
    live_map:   {path -> bool}
    merged_map: {branch -> bool}
    divergence: {branch -> (ahead, behind, unpushed)}
    """

    # Key fixtures by the worktree's POSIX path so Windows backslash-stringifying
    # of Path("/wt/x") doesn't break lookups (the resolver passes a Path through).
    def _key(p):
        return Path(p).as_posix()

    monkeypatch.setattr(fs._wm, "list_worktrees", lambda root: infos)
    monkeypatch.setattr(fs, "_porcelain", lambda p: porcelains.get(_key(p), ""))
    monkeypatch.setattr(fs, "_peer_live_in", lambda p, sid: live_map.get(_key(p), False))
    monkeypatch.setattr(fs, "_is_merged", lambda b, base: merged_map.get(b, False))
    monkeypatch.setattr(fs, "_ahead_behind_unpushed", lambda b, base: divergence.get(b, (0, 0, 0)))


def test_fleet_state_full_class_matrix(monkeypatch):
    """One fixture fleet covering live / hollow / merged / needs-finish / healthy."""
    infos = [
        _FakeInfo("/wt/live", "refs/heads/session/live", "aaa"),
        _FakeInfo("/wt/hollow", "refs/heads/session/hollow", "bbb"),
        _FakeInfo("/wt/merged", "refs/heads/session/merged", "ccc"),
        _FakeInfo("/wt/needs", "refs/heads/session/needs", "ddd"),
        _FakeInfo("/wt/healthy", "refs/heads/session/healthy", "eee"),
    ]
    porcelains = {
        "/wt/live": " M pipeline/x.py",
        "/wt/hollow": "\n".join(f" D gone_{i}.py" for i in range(300)),
        "/wt/merged": "",
        "/wt/needs": " M trading_app/y.py",
        "/wt/healthy": "",
    }
    live_map = {"/wt/live": True}
    merged_map = {"session/merged": True}
    divergence = {"session/needs": (2, 0, 2)}
    _patch_fleet(
        monkeypatch,
        infos=infos,
        porcelains=porcelains,
        live_map=live_map,
        merged_map=merged_map,
        divergence=divergence,
    )

    states = fs.fleet_state(root=Path("/repo"))
    # Key by branch — path is .resolve()'d (Windows turns /wt/x into C:\wt\x).
    by_branch = {s.branch: s for s in states}

    assert by_branch["session/live"].classification == fs.CLASS_LIVE
    assert by_branch["session/hollow"].classification == fs.CLASS_HOLLOW
    assert by_branch["session/merged"].classification == fs.CLASS_MERGED
    assert by_branch["session/needs"].classification == fs.CLASS_NEEDS_FINISH
    assert by_branch["session/healthy"].classification == fs.CLASS_HEALTHY


def test_fleet_state_current_tree_never_live(monkeypatch):
    """The tree fleet_state runs from is never classified as a live PEER."""
    repo = Path("/repo").resolve()
    infos = [_FakeInfo(str(repo), "refs/heads/main", "fff")]
    # Even if the liveness oracle would say True, is_current short-circuits it.
    _patch_fleet(
        monkeypatch,
        infos=infos,
        porcelains={str(repo): ""},
        live_map={str(repo): True},
        merged_map={},
        divergence={},
    )
    states = fs.fleet_state(root=repo)
    assert len(states) == 1
    assert states[0].is_current is True
    assert states[0].live is False
    assert states[0].classification != fs.CLASS_LIVE


def test_fleet_state_delegates_liveness_to_worktree_guard(monkeypatch):
    """PROVE delegation: _peer_live_in calls worktree_guard._peer_is_live, not a
    re-encoded copy. We spy on the canonical function and assert it was hit."""
    calls = {"n": 0}

    def _spy(lease, cwd=None, exclude_session_id=""):
        calls["n"] += 1
        return True

    # read_lease returns a truthy lease so _peer_is_live (the canonical path) runs.
    monkeypatch.setattr(fs._wg, "read_lease", lambda p: {"ppid": 999})
    monkeypatch.setattr(fs._wg, "_peer_is_live", _spy)

    result = fs._peer_live_in(Path("/wt/peer"), exclude_session_id="me")
    assert result is True
    assert calls["n"] == 1  # the canonical guard function WAS called


def test_fleet_state_empty_when_manager_unavailable(monkeypatch):
    """If worktree enumeration is impossible, return [] — never guess."""
    monkeypatch.setattr(fs._wm, "list_worktrees", lambda root: (_ for _ in ()).throw(RuntimeError("boom")))
    assert fs.fleet_state(root=Path("/repo")) == []


# ── fresh-heartbeat → LIVE integration (the load-bearing TRUE-branch test) ──
#
# Every prior live run read live=False (no peer was mid-tool-call), so the path
# that returns live=True / classifies LIVE never fired. These tests stage a REAL
# beat in a REAL git repo so the canonical _fresh_peer_heartbeat read path (beat
# file, mtime window, cwd-match) executes for real — only the OS-PID probe is
# stubbed when needed (operator decision: real beat + PID-stub fallback).


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _write_beat(repo: Path, session_id: str, *, cwd: Path, age_s: float, anchor_pid: int | None):
    """Stamp a `<session_id>.beat` into the repo's git-common-dir heartbeat dir,
    mirroring `tests/test_tools/test_worktree_guard.py::_write_beat`."""
    common = wg._git_common_dir(repo)
    assert common is not None
    beat_dir = common / wg.HEARTBEAT_DIRNAME
    beat_dir.mkdir(parents=True, exist_ok=True)
    bf = beat_dir / f"{session_id}.beat"
    payload = {
        "session_id": session_id,
        "cwd": str(cwd.resolve()),
        "branch": "main",
        "pid": 4242,
        "ts": time.time() - age_s,
    }
    if anchor_pid is not None:
        payload["anchor_pid"] = anchor_pid
    bf.write_text(json.dumps(payload), encoding="utf-8")
    if age_s:
        old = time.time() - age_s
        os.utime(bf, (old, old))
    return bf


def test_peer_live_in_true_branch_fires_on_fresh_beat(tmp_path):
    """TRUE branch: a fresh peer beat with a LIVE anchor in this tree → live=True.

    Exercises the real _fresh_peer_heartbeat read path; anchor_pid is THIS test
    process (genuinely alive) so no PID stub is needed for the happy path."""
    repo = _init_repo(tmp_path)
    # Peer 'peer', cwd == this tree, fresh, anchor = our own (live) pid.
    _write_beat(repo, "peer", cwd=repo, age_s=2.0, anchor_pid=os.getpid())

    assert fs._peer_live_in(repo, exclude_session_id="me") is True


def test_peer_live_in_true_branch_with_pid_stub_fallback(tmp_path, monkeypatch):
    """Same TRUE branch, but the OS-PID probe is stubbed (the CI-flaky fallback
    the operator sanctioned). We stub ONLY worktree_guard._pid_is_alive → True;
    the beat read path (file, mtime window, cwd-match) is still fully real."""
    repo = _init_repo(tmp_path)
    monkeypatch.setattr(wg, "_pid_is_alive", lambda *a, **k: True)
    # Use an arbitrary anchor_pid — the stub makes the OS probe deterministic.
    _write_beat(repo, "peer", cwd=repo, age_s=2.0, anchor_pid=123456)

    assert fs._peer_live_in(repo, exclude_session_id="me") is True


def test_fleet_state_classifies_fresh_beat_tree_as_live(tmp_path, monkeypatch):
    """End-to-end: a full fleet_state run classifies the beating tree LIVE.

    Enumeration is patched to point at the real repo (so we don't need a real
    `git worktree add`), but liveness runs for REAL against the staged beat —
    the path under test. anchor_pid is our live pid; no liveness stub."""
    repo = _init_repo(tmp_path)
    _write_beat(repo, "peer", cwd=repo, age_s=2.0, anchor_pid=os.getpid())

    # fleet_state self-skips the CURRENT tree (is_current → live=False). Point
    # enumeration at the repo but invoke from a DIFFERENT root so the repo is a
    # peer, not the current tree.
    infos = [_FakeInfo(str(repo), "refs/heads/session/peer", "aaa")]
    monkeypatch.setattr(fs._wm, "list_worktrees", lambda root: infos)
    monkeypatch.setattr(fs, "_ahead_behind_unpushed", lambda b, base: (0, 0, 0))
    monkeypatch.setattr(fs, "_is_merged", lambda b, base: False)

    states = fs.fleet_state(root=tmp_path / "elsewhere", exclude_session_id="me")
    assert len(states) == 1
    assert states[0].live is True
    assert states[0].classification == fs.CLASS_LIVE


def test_fleet_state_stale_beat_is_not_live(tmp_path, monkeypatch):
    """Negative: a beat older than the blocking window → live=False (not LIVE)."""
    repo = _init_repo(tmp_path)
    # Age beyond BLOCKING_HEARTBEAT_WINDOW_SECONDS → outside the live window.
    stale_age = wg.BLOCKING_HEARTBEAT_WINDOW_SECONDS + 60
    _write_beat(repo, "peer", cwd=repo, age_s=stale_age, anchor_pid=os.getpid())

    infos = [_FakeInfo(str(repo), "refs/heads/session/peer", "aaa")]
    monkeypatch.setattr(fs._wm, "list_worktrees", lambda root: infos)
    monkeypatch.setattr(fs, "_ahead_behind_unpushed", lambda b, base: (0, 0, 0))
    monkeypatch.setattr(fs, "_is_merged", lambda b, base: False)

    states = fs.fleet_state(root=tmp_path / "elsewhere", exclude_session_id="me")
    assert len(states) == 1
    assert states[0].live is False
    assert states[0].classification != fs.CLASS_LIVE


# ── HOLLOW + work-at-risk safety (a gutted tree that still holds real work) ──
#
# A HOLLOW tree that ALSO carries (a) unpushed commits or (b) a genuine
# non-deletion edit among the deletions must NOT be silently reap-eligible in
# Stage 2 — reaping it would lose unrecoverable work. The ladder must class it
# NEEDS_FINISH so Stage 2's reaper never deletes it. NOTE the work-at-risk signal
# for (b) is real_NONDEL_dirty, NOT real_dirty: a hollow tree's mass deletions
# ARE counted in real_dirty (a `D` line is a real change), so gating on real_dirty
# would route EVERY hollow tree to NEEDS_FINISH and defeat the class.


def test_classify_hollow_with_unpushed_is_needs_finish_not_hollow():
    cls, reasons = fs._classify(
        live=False, hollow=True, merged=False, real_dirty=0, real_nondel_dirty=0, unpushed=3, behind=0
    )
    # The safety invariant: unpushed commits outrank hollow.
    assert cls == fs.CLASS_NEEDS_FINISH
    assert cls != fs.CLASS_HOLLOW
    assert any("unpushed" in r for r in reasons)


def test_classify_hollow_with_real_nondel_edit_is_needs_finish():
    # Finding-2 fix: a gutted tree carrying a genuine non-deletion edit (e.g. one
    # `M src.py` among 300 deletions) is work-at-risk → NEEDS_FINISH, NOT reapable.
    cls, reasons = fs._classify(
        live=False, hollow=True, merged=False, real_dirty=301, real_nondel_dirty=1, unpushed=0, behind=0
    )
    assert cls == fs.CLASS_NEEDS_FINISH
    assert any("non-deletion" in r for r in reasons)


def test_classify_hollow_pure_deletions_stays_hollow_despite_high_real_dirty():
    # The trap the auditor's literal fix would have hit: real_dirty=300 (all
    # deletions), real_nondel_dirty=0. Must stay HOLLOW (reapable), NOT NEEDS_FINISH.
    cls, _ = fs._classify(
        live=False, hollow=True, merged=False, real_dirty=300, real_nondel_dirty=0, unpushed=0, behind=0
    )
    assert cls == fs.CLASS_HOLLOW


def test_classify_hollow_without_workatrisk_stays_hollow():
    # Control: a gutted tree with NO unpushed commits and NO real edits is reapable.
    cls, _ = fs._classify(
        live=False, hollow=True, merged=False, real_dirty=0, real_nondel_dirty=0, unpushed=0, behind=0
    )
    assert cls == fs.CLASS_HOLLOW


def test_classify_live_still_wins_over_hollow_workatrisk():
    # LIVE must still outrank the new HOLLOW+work-at-risk branch.
    cls, _ = fs._classify(live=True, hollow=True, merged=False, real_dirty=0, real_nondel_dirty=2, unpushed=3, behind=0)
    assert cls == fs.CLASS_LIVE


def test_fleet_state_hollow_unpushed_tree_not_reapable(monkeypatch):
    """End-to-end: a gutted tree with unpushed commits classes NEEDS_FINISH."""
    infos = [_FakeInfo("/wt/gutted", "refs/heads/session/gutted", "ggg")]
    porcelains = {"/wt/gutted": "\n".join(f" D gone_{i}.py" for i in range(300))}
    _patch_fleet(
        monkeypatch,
        infos=infos,
        porcelains=porcelains,
        live_map={},
        merged_map={},
        divergence={"session/gutted": (5, 0, 5)},  # 5 unpushed commits
    )
    states = fs.fleet_state(root=Path("/repo"))
    by_branch = {s.branch: s for s in states}
    assert by_branch["session/gutted"].classification == fs.CLASS_NEEDS_FINISH
    assert by_branch["session/gutted"].hollow is True  # still flagged hollow…
    assert by_branch["session/gutted"].unpushed == 5  # …but NOT reap-eligible


def test_fleet_state_hollow_with_real_edit_not_reapable(monkeypatch):
    """End-to-end: a gutted tree with one real non-deletion edit among the mass
    deletions classes NEEDS_FINISH (real_nondel_dirty>0), NOT reapable HOLLOW.
    Runs the REAL _count_dirty over the porcelain (only enumeration is patched)."""
    infos = [_FakeInfo("/wt/gutted2", "refs/heads/session/gutted2", "hhh")]
    deletions = [f" D gone_{i}.py" for i in range(300)]
    porcelain = "\n".join([*deletions, " M still_working.py"])  # one real edit
    _patch_fleet(
        monkeypatch,
        infos=infos,
        porcelains={"/wt/gutted2": porcelain},
        live_map={},
        merged_map={},
        divergence={"session/gutted2": (0, 0, 0)},  # NO unpushed — only the edit
    )
    states = fs.fleet_state(root=Path("/repo"))
    by_branch = {s.branch: s for s in states}
    s = by_branch["session/gutted2"]
    assert s.hollow is True  # 300 del + 1 nondel still hollow (nondel<=10)
    assert s.unpushed == 0  # no commit protection here…
    assert s.classification == fs.CLASS_NEEDS_FINISH  # …the real edit protects it


# ── active_plan anchor round-trip ────────────────────────────────────────────


def test_active_plan_roundtrip(tmp_path):
    path = tmp_path / "active_plan.md"
    plan = ap.ActivePlan(
        goal="Build the fleet-state brain",
        plan_file="/plans/x.md",
        current_stage=1,
        stages=4,
        status="IN_PROGRESS",
        updated="2026-06-06",
        unfinished=["Stage 2", "Stage 3"],
        body="# Notes\nsome body",
    )
    ap.write_active_plan(plan, path=path)
    loaded = ap.read_active_plan(path=path)
    assert loaded.exists
    assert loaded.goal == "Build the fleet-state brain"
    assert loaded.current_stage == 1
    assert loaded.stages == 4
    assert loaded.unfinished == ["Stage 2", "Stage 3"]
    assert "some body" in loaded.body


def test_active_plan_absent_is_empty(tmp_path):
    loaded = ap.read_active_plan(path=tmp_path / "nope.md")
    assert loaded.exists is False
    assert ap.summary_line(loaded) == ""


def test_active_plan_summary_line(tmp_path):
    plan = ap.ActivePlan(goal="G", current_stage=2, stages=4, status="IN_PROGRESS")
    line = ap.summary_line(plan)
    assert line.startswith("ACTIVE PLAN: G")
    assert "stage 2/4" in line


def test_active_plan_malformed_does_not_raise(tmp_path):
    path = tmp_path / "active_plan.md"
    path.write_text("---\ngoal without colon\ngarbage\n", encoding="utf-8")
    # Must not raise — fail-open for the SessionStart hook.
    loaded = ap.read_active_plan(path=path)
    assert isinstance(loaded, ap.ActivePlan)
