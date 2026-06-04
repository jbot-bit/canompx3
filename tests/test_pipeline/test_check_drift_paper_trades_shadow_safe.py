"""Tests for ``check_paper_trades_reads_are_shadow_safe`` (R1 Stage-2 root fix).

`paper_trades` is a shared table discriminated by ``execution_source``. Live /
monitoring consumers must never see forward-monitoring ``'shadow'`` rows. The
Stage-1 adversarial review proved the "every reader remembers a predicate"
vigilance contract leaks (6 → 7 → 8 unguarded readers found across review
passes). This static AST check ends the bug class: an unguarded ``FROM
paper_trades`` SELECT fails CI.

These tests are mutation-proof per ``institutional-rigor.md`` § 11 /
``integrity-guardian.md`` § 7 — the suite injects a KNOWN violation and confirms
the check catches it, then confirms each compliant form passes. A check that only
ever passes is not evidence.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.check_drift import check_paper_trades_reads_are_shadow_safe


def _write(tmp_path: Path, name: str, body: str) -> Path:
    d = tmp_path / "trading_app"
    d.mkdir(exist_ok=True)
    f = d / name
    f.write_text(body, encoding="utf-8")
    return d


def test_unguarded_select_is_caught(tmp_path: Path) -> None:
    """KNOWN-VIOLATION INJECTION — an unguarded SELECT FROM paper_trades fails."""
    d = _write(
        tmp_path,
        "bad.py",
        'def stat(con):\n    return con.execute("SELECT COUNT(*) FROM paper_trades WHERE orb_label = ?", [x])\n',
    )
    viols = check_paper_trades_reads_are_shadow_safe(scan_dirs=[d])
    assert len(viols) == 1, f"unguarded read must be caught, got {viols}"
    assert "bad.py" in viols[0]


def test_execution_source_predicate_passes(tmp_path: Path) -> None:
    d = _write(
        tmp_path,
        "guarded.py",
        "def stat(con):\n    return con.execute(\n"
        '        "SELECT COUNT(*) FROM paper_trades WHERE orb_label = ? '
        "AND execution_source != 'shadow'\", [x])\n",
    )
    assert check_paper_trades_reads_are_shadow_safe(scan_dirs=[d]) == []


def test_view_read_passes(tmp_path: Path) -> None:
    d = _write(
        tmp_path,
        "viewreader.py",
        'def stat(con):\n    return con.execute("SELECT COUNT(*) FROM live_paper_trades").fetchone()\n',
    )
    assert check_paper_trades_reads_are_shadow_safe(scan_dirs=[d]) == []


def test_pnl_dollar_not_null_implicit_guard_passes(tmp_path: Path) -> None:
    """consistency_tracker's documented form: shadow rows have NULL pnl_dollar."""
    d = _write(
        tmp_path,
        "ct.py",
        "def stat(con):\n    return con.execute(\n"
        '        "SELECT SUM(pnl_dollar) FROM paper_trades WHERE pnl_dollar IS NOT NULL")\n',
    )
    assert check_paper_trades_reads_are_shadow_safe(scan_dirs=[d]) == []


def test_delete_is_not_flagged(tmp_path: Path) -> None:
    """Writes (DELETE/UPDATE) are intentionally exempt — only SELECT reads are
    guarded. paper_trade_logger's idempotent DELETE keyed on strategy_id must NOT
    be forced to add an execution_source predicate (it deletes CORE rows; shadow
    rows are tier-disjoint and must stay untouched)."""
    d = _write(
        tmp_path,
        "writer.py",
        'def clear(con):\n    con.execute("DELETE FROM paper_trades WHERE strategy_id = ?", [sid])\n',
    )
    assert check_paper_trades_reads_are_shadow_safe(scan_dirs=[d]) == []


def test_prose_mention_is_not_flagged(tmp_path: Path) -> None:
    """A docstring saying 'reads from paper_trades' is not SQL — no false positive."""
    d = _write(
        tmp_path,
        "doc.py",
        '"""This module reads from paper_trades table for monitoring."""\ndef noop():\n    return None\n',
    )
    assert check_paper_trades_reads_are_shadow_safe(scan_dirs=[d]) == []


def test_fstring_with_predicate_after_interpolation_passes(tmp_path: Path) -> None:
    """An f-string whose execution_source predicate sits AFTER a ``{placeholders}``
    interpolation must still be recognized as guarded (the project_pulse shape that
    initially false-positived during implementation)."""
    d = _write(
        tmp_path,
        "fstr.py",
        "def q(con, ph):\n"
        "    return con.execute(\n"
        '        f"SELECT strategy_id, COUNT(*) FROM paper_trades '
        "WHERE strategy_id IN ({ph}) AND execution_source != 'shadow' GROUP BY 1\")\n",
    )
    assert check_paper_trades_reads_are_shadow_safe(scan_dirs=[d]) == []


def test_real_repo_tree_is_clean() -> None:
    """The production tree passes after the Stage-2 reader migration."""
    assert check_paper_trades_reads_are_shadow_safe() == []
