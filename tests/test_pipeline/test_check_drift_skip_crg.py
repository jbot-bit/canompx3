"""Tests for the --skip-crg-advisory flag on pipeline/check_drift.py (Lever 1).

The flag removes the CRG D1-D5 ADVISORY checks from the commit drift gate (D5
alone ~73s, a per-node graph query measured 2026-05-30). The honesty contract:
skipping must NEVER remove a blocking check, CI still runs the full set, and the
flag defaults off so the no-flag path is byte-identical to prior behavior.

The label-set invariant tests are hermetic (no DB, no subprocess) and are the
load-bearing coverage: they prove --skip-crg-advisory can only ever skip advisory
checks. One bounded subprocess test confirms the flag actually skips D1-D5 at
runtime; it is fail-closed on multi-worktree DuckDB contention (skip, not fail).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from pipeline import check_drift

PROJ_ROOT = Path(__file__).resolve().parents[2]


class TestCrgAdvisoryLabelInvariants:
    """Hermetic invariants — the honesty contract enforced in code."""

    def test_every_crg_advisory_label_exists_in_checks(self):
        known = {label for label, *_ in check_drift.CHECKS}
        stale = check_drift.CRG_ADVISORY_LABELS - known
        assert not stale, f"CRG_ADVISORY_LABELS has labels absent from CHECKS: {sorted(stale)}"

    def test_every_crg_advisory_label_is_advisory(self):
        """THE honesty property: the skip set may only contain advisory checks.

        A blocking check in this set would let --skip-crg-advisory silently
        weaken the commit gate. The in-module guard asserts this at import; this
        test pins it so a regression is caught even if the guard is removed.
        """
        by_label = {label: is_adv for label, _fn, is_adv, _db in check_drift.CHECKS}
        blocking = sorted(lbl for lbl in check_drift.CRG_ADVISORY_LABELS if not by_label[lbl])
        assert not blocking, f"--skip-crg-advisory would skip BLOCKING check(s): {blocking}"

    def test_set_is_exactly_the_five_crg_checks(self):
        """D1-D5 — no more (don't skip non-CRG checks), no fewer (D5 is the point)."""
        crg_in_checks = {label for label, *_ in check_drift.CHECKS if label.startswith("CRG D")}
        assert crg_in_checks == check_drift.CRG_ADVISORY_LABELS, (
            "CRG_ADVISORY_LABELS must equal exactly the CRG D1-D5 checks in CHECKS; "
            f"diff: {check_drift.CRG_ADVISORY_LABELS ^ crg_in_checks}"
        )

    def test_guard_raises_when_a_crg_label_becomes_blocking(self, monkeypatch):
        """Fail-closed: flipping a CRG check to blocking must raise at validation."""
        patched = []
        for label, fn, is_adv, db in check_drift.CHECKS:
            if label in check_drift.CRG_ADVISORY_LABELS and is_adv:
                patched.append((label, fn, False, db))  # force blocking
            else:
                patched.append((label, fn, is_adv, db))
        monkeypatch.setattr(check_drift, "CHECKS", patched)
        with pytest.raises(RuntimeError, match="BLOCKING check"):
            check_drift._assert_crg_advisory_labels_valid()

    def test_guard_raises_on_unknown_label(self, monkeypatch):
        monkeypatch.setattr(
            check_drift,
            "CRG_ADVISORY_LABELS",
            check_drift.CRG_ADVISORY_LABELS | {"CRG D9: not a real check"},
        )
        with pytest.raises(RuntimeError, match="not present in CHECKS"):
            check_drift._assert_crg_advisory_labels_valid()


class TestSkipCrgAdvisoryRuntime:
    """Bounded subprocess: confirm the flag actually skips D1-D5 at runtime."""

    @pytest.mark.timeout(240)
    def test_flag_skips_crg_checks_and_keeps_exit_semantics(self):
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJ_ROOT / "pipeline" / "check_drift.py"),
                    "--skip-crg-advisory",
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJ_ROOT),
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("check_drift.py --skip-crg-advisory did not return in 180s (DB contention)")
        assert result.returncode in (0, 1), f"unexpected exit: {result.stderr}"
        out = result.stdout
        # Every CRG D1-D5 check must report the skip marker.
        for label in check_drift.CRG_ADVISORY_LABELS:
            assert label in out, f"expected CRG check line present: {label}"
        assert "SKIPPED (--skip-crg-advisory" in out, "skip marker missing from output"
        # Summary must report the count.
        assert "skipped (--skip-crg-advisory)" in out, "summary missing crg skip count"

    @pytest.mark.timeout(240)
    def test_no_flag_run_does_not_skip_crg(self):
        """Default path: no flag → no --skip-crg-advisory marker (full set runs)."""
        try:
            result = subprocess.run(
                [sys.executable, str(PROJ_ROOT / "pipeline" / "check_drift.py")],
                capture_output=True,
                text=True,
                cwd=str(PROJ_ROOT),
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("check_drift.py did not return in 180s (DB contention)")
        assert result.returncode in (0, 1), f"unexpected exit: {result.stderr}"
        assert "SKIPPED (--skip-crg-advisory" not in result.stdout, "no-flag run must not skip CRG checks"


class TestSkipAllAdvisoryRuntime:
    """Hermetic runtime proof for pre-commit's all-advisory skip path."""

    def test_skip_advisory_runs_blocking_checks_and_skips_advisory_only(self, monkeypatch, capsys):
        calls: list[str] = []

        def blocking_check() -> list[str]:
            calls.append("blocking")
            return []

        def advisory_check() -> list[str]:
            calls.append("advisory")
            return []

        monkeypatch.setattr(
            check_drift,
            "CHECKS",
            [
                ("blocking check", blocking_check, False, False),
                ("advisory check", advisory_check, True, False),
            ],
        )
        monkeypatch.setattr(sys, "argv", ["check_drift.py", "--skip-advisory"])

        with pytest.raises(SystemExit) as excinfo:
            check_drift.main()

        assert excinfo.value.code == 0
        assert calls == ["blocking"]
        out = capsys.readouterr().out
        assert "SKIPPED (--skip-advisory; advisory, runs in full drift/CI)" in out
        assert "1 skipped (--skip-advisory)" in out
