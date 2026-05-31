"""Tests for `scripts/tools/worktree_launch_preflight.py` (read-only classifier).

Covers compute_wt_path, classify (NEW / REUSE_CLEAN / REFUSE_HOT), dirty-check,
lease-hot delegation + fail-open, and the --json CLI shape. Filesystem stays
under tmp_path; the canonical worktree_guard subprocess is mocked so tests are
deterministic and never touch the developer's real lease.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))

import worktree_launch_preflight as wlp  # noqa: E402  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# compute_wt_path
# --------------------------------------------------------------------------- #
def test_compute_wt_path_mirrors_new_session(tmp_path):
    root = tmp_path / "canompx3"
    root.mkdir()
    wt, branch = wlp.compute_wt_path("hwm-fix", repo_root=root)
    assert wt == tmp_path / "canompx3-hwm-fix"
    assert branch.startswith("session/")
    assert branch.endswith("-hwm-fix")


def test_branch_format(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _, branch = wlp.compute_wt_path("x", repo_root=root)
    parts = branch.split("/", 1)
    assert parts[0] == "session"
    assert parts[1].endswith("-x")


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #
def test_classify_new_when_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert wlp.classify(missing) == wlp.NEW


def test_classify_reuse_clean(tmp_path):
    wt = tmp_path / "clean"
    wt.mkdir()
    with patch.object(wlp, "_is_dirty", return_value=False), patch.object(wlp, "_lease_hot", return_value=False):
        assert wlp.classify(wt) == wlp.REUSE_CLEAN


def test_classify_refuse_when_dirty(tmp_path):
    wt = tmp_path / "dirty"
    wt.mkdir()
    with patch.object(wlp, "_is_dirty", return_value=True), patch.object(wlp, "_lease_hot", return_value=False):
        assert wlp.classify(wt) == wlp.REFUSE_HOT


def test_classify_refuse_when_lease_hot(tmp_path):
    wt = tmp_path / "hot"
    wt.mkdir()
    with patch.object(wlp, "_is_dirty", return_value=False), patch.object(wlp, "_lease_hot", return_value=True):
        assert wlp.classify(wt) == wlp.REFUSE_HOT


# --------------------------------------------------------------------------- #
# _lease_hot delegation
# --------------------------------------------------------------------------- #
def _fake_run(stdout, returncode=0):
    def _run(*a, **k):
        return subprocess.CompletedProcess(args=a, returncode=returncode, stdout=stdout, stderr="")

    return _run


def test_lease_hot_true_when_present_and_peer_live(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    payload = json.dumps({"lease_present": True, "peer_live": True})
    with (
        patch.object(wlp, "_repo_root", return_value=PROJECT_ROOT),
        patch.object(Path, "exists", return_value=True),
        patch.object(subprocess, "run", _fake_run(payload)),
    ):
        assert wlp._lease_hot(wt) is True


def test_lease_hot_false_when_present_but_no_peer(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    payload = json.dumps({"lease_present": True, "peer_live": False})
    with (
        patch.object(wlp, "_repo_root", return_value=PROJECT_ROOT),
        patch.object(Path, "exists", return_value=True),
        patch.object(subprocess, "run", _fake_run(payload)),
    ):
        assert wlp._lease_hot(wt) is False


def test_lease_hot_failopen_on_nonzero_rc(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    with (
        patch.object(wlp, "_repo_root", return_value=PROJECT_ROOT),
        patch.object(Path, "exists", return_value=True),
        patch.object(subprocess, "run", _fake_run("", returncode=1)),
    ):
        assert wlp._lease_hot(wt) is False


def test_lease_hot_failopen_on_bad_json(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    with (
        patch.object(wlp, "_repo_root", return_value=PROJECT_ROOT),
        patch.object(Path, "exists", return_value=True),
        patch.object(subprocess, "run", _fake_run("not json{{{")),
    ):
        assert wlp._lease_hot(wt) is False


def test_lease_hot_failopen_on_subprocess_error(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=8)

    with (
        patch.object(wlp, "_repo_root", return_value=PROJECT_ROOT),
        patch.object(Path, "exists", return_value=True),
        patch.object(subprocess, "run", _boom),
    ):
        assert wlp._lease_hot(wt) is False


def test_lease_hot_false_when_guard_missing(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    fake_root = tmp_path / "no-guard-here"
    fake_root.mkdir()
    with patch.object(wlp, "_repo_root", return_value=fake_root):
        # guard path under fake_root does not exist -> False without subprocess
        assert wlp._lease_hot(wt) is False


# --------------------------------------------------------------------------- #
# _is_dirty
# --------------------------------------------------------------------------- #
def test_is_dirty_true_on_porcelain_output(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    with patch.object(subprocess, "run", _fake_run(" M file.py\n")):
        assert wlp._is_dirty(wt) is True


def test_is_dirty_false_on_clean(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    with patch.object(subprocess, "run", _fake_run("")):
        assert wlp._is_dirty(wt) is False


def test_is_dirty_failopen_on_git_error(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    with patch.object(subprocess, "run", _fake_run("", returncode=128)):
        assert wlp._is_dirty(wt) is False


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def test_main_json_shape_and_returns_zero(tmp_path, capsys):
    root = tmp_path / "canompx3"
    root.mkdir()
    with patch.object(wlp, "_repo_root", return_value=root):
        rc = wlp.main(["--descriptor", "smoke", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["descriptor"] == "smoke"
    assert out["decision"] in (wlp.NEW, wlp.REUSE_CLEAN, wlp.REFUSE_HOT)
    assert out["worktree_path"].endswith("canompx3-smoke")
    assert out["branch"].endswith("-smoke")


def test_main_text_shape_parseable_by_bat(tmp_path, capsys):
    root = tmp_path / "canompx3"
    root.mkdir()
    with patch.object(wlp, "_repo_root", return_value=root):
        rc = wlp.main(["--descriptor", "smoke"])
    assert rc == 0
    lines = {ln.split("=", 1)[0]: ln.split("=", 1)[1] for ln in capsys.readouterr().out.splitlines() if "=" in ln}
    assert lines["DECISION"] in (wlp.NEW, wlp.REUSE_CLEAN, wlp.REFUSE_HOT)
    assert "WTPATH" in lines
    assert "BRANCH" in lines


def test_main_classification_never_errors_on_missing(tmp_path, capsys):
    # A brand-new descriptor always classifies NEW -> rc 0.
    root = tmp_path / "canompx3"
    root.mkdir()
    with patch.object(wlp, "_repo_root", return_value=root):
        rc = wlp.main(["--descriptor", "totally-fresh-xyz", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["decision"] == wlp.NEW
