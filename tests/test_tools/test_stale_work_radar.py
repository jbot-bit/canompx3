"""Tests for scripts.tools.stale_work_radar."""

import json
import subprocess
from unittest.mock import patch

from scripts.tools import stale_work_radar as radar


def _cp(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout=stdout, stderr=stderr)


class TestAheadBehindParsing:
    def test_left_is_behind_right_is_ahead(self) -> None:
        # `rev-list --left-right --count base...branch` => "<behind>\t<ahead>"
        with patch.object(radar, "_run_git", return_value=_cp("106\t9\n")):
            ahead, behind = radar.ahead_behind("session/x", "origin/main")
        assert ahead == 9
        assert behind == 106

    def test_failure_returns_zero_zero(self) -> None:
        with patch.object(radar, "_run_git", return_value=_cp(returncode=1)):
            assert radar.ahead_behind("x", "origin/main") == (0, 0)

    def test_malformed_returns_zero_zero(self) -> None:
        with patch.object(radar, "_run_git", return_value=_cp("garbage\n")):
            assert radar.ahead_behind("x", "origin/main") == (0, 0)


class TestUnpushed:
    def test_no_remote_tip_means_local_only(self) -> None:
        # rev-parse origin/<branch> fails => no remote
        with patch.object(radar, "_run_git", return_value=_cp(returncode=1)):
            count, has_remote = radar.unpushed_count("session/local-only")
        assert count == 0
        assert has_remote is False

    def test_remote_tip_counts_unpushed(self) -> None:
        calls = [_cp("deadbeef\n"), _cp("3\n")]  # remote_tip, then rev-list count
        with patch.object(radar, "_run_git", side_effect=calls):
            count, has_remote = radar.unpushed_count("session/x")
        assert count == 3
        assert has_remote is True


class TestWorktreeMap:
    def test_parses_porcelain(self) -> None:
        output = "\n".join(
            [
                "worktree /repo/main",
                "HEAD abc",
                "branch refs/heads/main",
                "",
                "worktree /repo/wt-a",
                "HEAD def",
                "branch refs/heads/session/feature-a",
                "",
                "worktree /repo/detached",
                "HEAD 999",
                "detached",
                "",
            ]
        )
        with patch.object(radar, "_run_git", return_value=_cp(output)):
            mapping = radar.worktree_for_branch()
        assert mapping["main"] == "/repo/main"
        assert mapping["session/feature-a"] == "/repo/wt-a"
        assert "detached" not in mapping  # detached HEAD has no branch line


class TestScore:
    def _report(self, **kw) -> radar.BranchReport:
        defaults = dict(
            branch="b",
            ahead=0,
            behind=0,
            unpushed=0,
            has_remote=True,
            upstream_gone=False,
            last_commit_age_days=0,
            last_commit_date="2026-06-03",
            worktree_path=None,
            dirty_lines=0,
            merged_into_base=False,
        )
        defaults.update(kw)
        return radar.BranchReport(**defaults)

    def test_local_only_uses_ahead_as_unpushed(self) -> None:
        r = self._report(has_remote=False, ahead=9)
        radar.score(r)
        assert r.risk_breakdown["unpushed"] == 9 * radar.W_UNPUSHED_COMMIT
        assert any("LOCAL-ONLY" in f for f in r.flags)

    def test_remote_branch_uses_precise_unpushed(self) -> None:
        r = self._report(has_remote=True, unpushed=2, ahead=2)
        radar.score(r)
        assert r.risk_breakdown["unpushed"] == 2 * radar.W_UNPUSHED_COMMIT
        assert all("LOCAL-ONLY" not in f for f in r.flags)

    def test_blob_trap_flagged(self) -> None:
        r = self._report(dirty_lines=13392)
        radar.score(r)
        assert any("artifact-blob trap" in f for f in r.flags)
        assert r.risk_breakdown["dirty"] == radar.W_DIRTY_FLAG

    def test_rebase_debt_only_past_threshold(self) -> None:
        below = self._report(behind=radar.REBASE_DEBT_THRESHOLD)
        radar.score(below)
        assert "rebase_debt" not in below.risk_breakdown
        above = self._report(behind=radar.REBASE_DEBT_THRESHOLD + 1)
        radar.score(above)
        assert above.risk_breakdown["rebase_debt"] == radar.W_REBASE_DEBT

    def test_clean_merged_branch_is_zero_risk_and_prunable(self) -> None:
        r = self._report(merged_into_base=True, ahead=0, unpushed=0, dirty_lines=0)
        radar.score(r)
        assert r.risk_score == 0.0
        assert any("safe to prune" in f for f in r.flags)

    def test_unpushed_outranks_age(self) -> None:
        unpushed = self._report(has_remote=False, ahead=1)  # 10.0
        stale = self._report(last_commit_age_days=50)  # 5.0
        radar.score(unpushed)
        radar.score(stale)
        assert unpushed.risk_score > stale.risk_score

    def test_upstream_gone_scored_and_flagged(self) -> None:
        r = self._report(upstream_gone=True)
        radar.score(r)
        assert r.risk_breakdown["upstream_gone"] == radar.W_UPSTREAM_GONE
        assert any("[gone]" in f for f in r.flags)

    def test_gone_branch_not_marked_safe_to_prune(self) -> None:
        r = self._report(merged_into_base=True, upstream_gone=True)
        radar.score(r)
        assert all("safe to prune" not in f for f in r.flags)


class TestBranchMetadata:
    def test_single_pass_parses_date_upstream_track(self) -> None:
        sep = radar._FER_SEP
        out = "\n".join(
            [
                sep.join(["session/x", "2026-06-01T10:00:00+10:00", "origin/session/x", ""]),
                sep.join(["codex/y", "2026-05-20T09:00:00+10:00", "origin/codex/y", "[gone]"]),
                sep.join(["local-only", "2026-06-02T08:00:00+10:00", "", ""]),
            ]
        )
        with patch.object(radar, "_run_git", return_value=_cp(out)):
            meta = radar.branch_metadata()
        assert meta["session/x"]["upstream"] == "origin/session/x"
        assert radar.upstream_gone(meta["codex/y"]["track"]) is True
        assert radar.upstream_gone(meta["session/x"]["track"]) is False
        assert meta["local-only"]["upstream"] == ""

    def test_age_from_iso(self) -> None:
        age, date = radar.age_from_iso("2026-06-01T10:00:00+10:00")
        assert age >= 0
        assert date == "2026-06-01"
        assert radar.age_from_iso("")[1] == "unknown"
        assert radar.age_from_iso("garbage") == (0, "garbage")


class TestMainExitClean:
    def test_missing_base_exits_zero(self, capsys) -> None:
        with patch.object(radar, "base_exists", return_value=False):
            rc = radar.main(["--base", "origin/nonexistent"])
        assert rc == 0
        assert "not found" in capsys.readouterr().err

    def test_json_output_is_valid(self, capsys) -> None:
        rpt = radar.BranchReport(
            branch="b",
            ahead=1,
            behind=0,
            unpushed=0,
            has_remote=False,
            upstream_gone=False,
            last_commit_age_days=0,
            last_commit_date="2026-06-03",
            worktree_path=None,
            dirty_lines=0,
            merged_into_base=False,
        )
        radar.score(rpt)
        with (
            patch.object(radar, "base_exists", return_value=True),
            patch.object(radar, "build_reports", return_value=[rpt]),
        ):
            rc = radar.main(["--json"])
        assert rc == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed[0]["branch"] == "b"
        assert "risk_breakdown" in parsed[0]
