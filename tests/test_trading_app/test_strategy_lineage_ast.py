"""Tests for trading_app.strategy_lineage_ast.scan_python_for_column_refs.

Per docs/plans/2026-04-30-crg-maximization-v2.md § PR-4a — 6 cases covering
ast_literal hits, regex fallback, banned-column flagging, empty/malformed inputs.
"""

from pathlib import Path

from trading_app.strategy_lineage_ast import ColumnRef, scan_python_for_column_refs


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def test_subscript_literal_df_bracket_string(tmp_path: Path) -> None:
    """`df["atr_20"]` → 1 ast_literal hit on column 'atr_20'."""
    src = 'import pandas as pd\ndef f(df):\n    return df["atr_20"].mean()\n'
    p = _write(tmp_path, "a.py", src)
    refs = scan_python_for_column_refs(p, column_allowlist={"atr_20", "rsi_14"})
    assert len(refs) == 1
    r = refs[0]
    assert r.column == "atr_20"
    assert r.evidence == "ast_literal"
    assert r.lineno == 3
    assert r.file == p


def test_fstring_constant_inside_join(tmp_path: Path) -> None:
    """f-string containing a string Constant for a column → ast_literal hit."""
    src = 'def q():\n    col = "atr_20"\n    sql = f"SELECT {col} FROM daily_features"\n    return sql\n'
    p = _write(tmp_path, "b.py", src)
    refs = scan_python_for_column_refs(p, column_allowlist={"atr_20"})
    cols = {r.column for r in refs}
    assert "atr_20" in cols
    assert all(r.evidence in ("ast_literal", "regex_fallback") for r in refs)
    assert any(r.evidence == "ast_literal" for r in refs)


def test_sql_fragment_string_constant(tmp_path: Path) -> None:
    """Plain SQL string containing a column name → ast_literal Constant hit."""
    src = 'QUERY = "SELECT atr_20, rsi_14 FROM daily_features WHERE atr_20 > 0"\n'
    p = _write(tmp_path, "c.py", src)
    refs = scan_python_for_column_refs(p, column_allowlist={"atr_20", "rsi_14"})
    cols = {r.column for r in refs}
    assert {"atr_20", "rsi_14"}.issubset(cols)
    assert all(r.evidence == "ast_literal" for r in refs)


def test_lookahead_banned_column_flagged(tmp_path: Path) -> None:
    """Banned column (e.g., break_dir on E2) shows up labeled, caller filters."""
    src = 'def features(df):\n    if df["break_dir"].iloc[-1] == 1:\n        return df["atr_20"]\n    return None\n'
    p = _write(tmp_path, "d.py", src)
    refs = scan_python_for_column_refs(p, column_allowlist={"atr_20", "break_dir"})
    cols_by_evidence = {(r.column, r.evidence) for r in refs}
    assert ("break_dir", "ast_literal") in cols_by_evidence
    assert ("atr_20", "ast_literal") in cols_by_evidence


def test_file_with_no_refs_returns_empty(tmp_path: Path) -> None:
    """Pure logic file with no allowlist matches → []."""
    src = "def add(a, b):\n    return a + b\n"
    p = _write(tmp_path, "e.py", src)
    refs = scan_python_for_column_refs(p, column_allowlist={"atr_20"})
    assert refs == []


def test_malformed_python_no_raise(tmp_path: Path) -> None:
    """Syntax error in source → returns [] without raising."""
    src = "def broken(:\n  pass\n"
    p = _write(tmp_path, "bad.py", src)
    refs = scan_python_for_column_refs(p, column_allowlist={"atr_20"})
    assert refs == []


def test_columnref_is_namedtuple_with_expected_fields() -> None:
    """ColumnRef public contract: file, lineno, column, evidence."""
    r = ColumnRef(file=Path("x.py"), lineno=1, column="atr_20", evidence="ast_literal")
    assert r.file == Path("x.py")
    assert r.lineno == 1
    assert r.column == "atr_20"
    assert r.evidence == "ast_literal"
