"""
WS2: Tests for behavioral anti-pattern scanner (scripts/tools/audit_behavioral.py).

Covers all 6 checks. Each test creates temp files with violations and verifies detection.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

# We need to import the module to monkeypatch its globals
from scripts.tools import audit_behavioral


def _mkfile(path: Path, content: str):
    """Create a file with content, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── Check 1: Hardcoded check counts ──────────────────────────────────


class TestHardcodedCheckCounts:
    """Check 1: 'all N checks' patterns should be computed dynamically."""

    def test_catches_hardcoded_count(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "pipeline" / "bad.py", 'print("all 17 checks passed")\n')
        monkeypatch.setattr(audit_behavioral, "PIPELINE_DIRS", [tmp_path / "pipeline"])
        monkeypatch.setattr(audit_behavioral, "CI_DIRS", [])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_hardcoded_check_counts()
        assert len(violations) > 0
        assert "17" in violations[0]

    def test_passes_dynamic_count(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "pipeline" / "good.py", 'print(f"all {len(CHECKS)} checks passed")\n')
        monkeypatch.setattr(audit_behavioral, "PIPELINE_DIRS", [tmp_path / "pipeline"])
        monkeypatch.setattr(audit_behavioral, "CI_DIRS", [])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_hardcoded_check_counts()
        assert len(violations) == 0


# ── Check 2: Hardcoded instrument lists ──────────────────────────────


class TestHardcodedInstrumentLists:
    """Check 2: 3+ instrument symbols in lists/SQL must import from config."""

    def test_catches_python_list(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "pipeline" / "bad.py", "instruments = ['MGC', 'MNQ', 'MES']\n")
        monkeypatch.setattr(audit_behavioral, "INSTRUMENT_SCAN_DIRS", [tmp_path / "pipeline"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_hardcoded_instrument_lists()
        assert len(violations) > 0
        assert "hardcoded instrument list" in violations[0]

    def test_catches_sql_in_clause(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "scripts" / "tools" / "bad.py", "query = \"IN ('MGC', 'MNQ', 'MES')\"\n")
        monkeypatch.setattr(audit_behavioral, "INSTRUMENT_SCAN_DIRS", [tmp_path / "scripts" / "tools"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_hardcoded_instrument_lists()
        assert len(violations) > 0

    def test_passes_import_from_config(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "pipeline" / "good.py", "from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS\n")
        monkeypatch.setattr(audit_behavioral, "INSTRUMENT_SCAN_DIRS", [tmp_path / "pipeline"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_hardcoded_instrument_lists()
        assert len(violations) == 0

    def test_allowlisted_file_passes(self, tmp_path, monkeypatch):
        # asset_configs.py is in the allowlist
        _mkfile(tmp_path / "pipeline" / "asset_configs.py", "INSTRUMENTS = ['MGC', 'MNQ', 'MES', 'M2K']\n")
        monkeypatch.setattr(audit_behavioral, "INSTRUMENT_SCAN_DIRS", [tmp_path / "pipeline"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_hardcoded_instrument_lists()
        assert len(violations) == 0


# ── Check 3: Broad except returning success ──────────────────────────


class TestBroadExceptSuccess:
    """Check 3: except Exception + return True/0 in health/audit code."""

    def test_catches_except_return_true(self, tmp_path, monkeypatch):
        _mkfile(
            tmp_path / "pipeline" / "health_check.py",
            """\
def check():
    try:
        do_something()
    except Exception:
        return True
""",
        )
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_broad_except_success()
        assert len(violations) > 0
        assert "broad except" in violations[0]

    def test_catches_except_return_0(self, tmp_path, monkeypatch):
        _mkfile(
            tmp_path / "pipeline" / "check_drift.py",
            """\
def run():
    try:
        validate()
    except Exception as e:
        return 0
""",
        )
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_broad_except_success()
        assert len(violations) > 0

    def test_passes_except_return_none(self, tmp_path, monkeypatch):
        _mkfile(
            tmp_path / "pipeline" / "health_check.py",
            """\
def check():
    try:
        do_something()
    except Exception:
        return None
""",
        )
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_broad_except_success()
        assert len(violations) == 0

    def test_allowlisted_file_passes(self, tmp_path, monkeypatch):
        # live_config.py is allowlisted (dollar gate fails open intentionally)
        _mkfile(
            tmp_path / "trading_app" / "live_config.py",
            """\
def dollar_gate():
    try:
        check_risk()
    except Exception:
        return True
""",
        )
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_broad_except_success()
        assert len(violations) == 0


# ── Check 4: CLI arg drift ───────────────────────────────────────────


class TestCliArgDrift:
    """Check 4: New CLI args without doc/test references (warning only)."""

    def test_returns_empty_on_clean_diff(self, tmp_path, monkeypatch):
        """No new add_argument in diff → no warnings."""
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        # Mock subprocess.run to return clean diff
        mock_result = type("Result", (), {"returncode": 0, "stdout": "diff --git a/x.py b/x.py\n"})()
        with patch("subprocess.run", return_value=mock_result):
            warnings = audit_behavioral.check_cli_arg_drift()
        assert len(warnings) == 0

    def test_warns_on_undocumented_arg(self, tmp_path, monkeypatch):
        """New add_argument with no doc reference → warning."""
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        diff = '+++ b/pipeline/run.py\n+    parser.add_argument("--new-flag", help="test")\n'
        mock_result = type("Result", (), {"returncode": 0, "stdout": diff})()
        with patch("subprocess.run", return_value=mock_result):
            warnings = audit_behavioral.check_cli_arg_drift()
        assert len(warnings) > 0
        assert "--new-flag" in warnings[0]

    def test_no_warning_when_arg_documented(self, tmp_path, monkeypatch):
        """New add_argument WITH matching doc reference → no warning."""
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        diff = (
            "+++ b/pipeline/run.py\n"
            '+    parser.add_argument("--new-flag", help="test")\n'
            "+++ b/docs/usage.md\n"
            "+Use --new-flag or new_flag to enable the feature.\n"
        )
        mock_result = type("Result", (), {"returncode": 0, "stdout": diff})()
        with patch("subprocess.run", return_value=mock_result):
            warnings = audit_behavioral.check_cli_arg_drift()
        assert len(warnings) == 0


# ── Check 5: Triple-join guard ───────────────────────────────────────


class TestTripleJoinGuard:
    """Check 5: JOIN daily_features must include orb_minutes."""

    def test_catches_sql_join_without_orb_minutes(self, tmp_path, monkeypatch):
        _mkfile(
            tmp_path / "research" / "bad.py",
            '''\
query = """
    SELECT * FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day
    AND o.symbol = d.symbol
"""
''',
        )
        monkeypatch.setattr(audit_behavioral, "TRIPLE_JOIN_SCAN_DIRS", [tmp_path / "research"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_triple_join_guard()
        assert len(violations) > 0
        assert "orb_minutes" in violations[0]

    def test_passes_sql_join_with_orb_minutes(self, tmp_path, monkeypatch):
        _mkfile(
            tmp_path / "research" / "good.py",
            '''\
query = """
    SELECT * FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day
    AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
"""
''',
        )
        monkeypatch.setattr(audit_behavioral, "TRIPLE_JOIN_SCAN_DIRS", [tmp_path / "research"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_triple_join_guard()
        assert len(violations) == 0

    def test_catches_dataframe_merge_without_orb_minutes(self, tmp_path, monkeypatch):
        _mkfile(
            tmp_path / "research" / "bad_merge.py",
            """\
import pandas as pd
daily_features = pd.read_sql("SELECT * FROM daily_features", con)
result = outcomes.merge(daily_features, on=['trading_day', 'symbol'])
""",
        )
        monkeypatch.setattr(audit_behavioral, "TRIPLE_JOIN_SCAN_DIRS", [tmp_path / "research"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_triple_join_guard()
        assert len(violations) > 0
        assert "merge" in violations[0].lower()

    def test_passes_dataframe_merge_with_orb_minutes(self, tmp_path, monkeypatch):
        _mkfile(
            tmp_path / "research" / "good_merge.py",
            """\
import pandas as pd
daily_features = pd.read_sql("SELECT * FROM daily_features", con)
result = outcomes.merge(daily_features,
    on=['trading_day', 'symbol', 'orb_minutes'])
""",
        )
        monkeypatch.setattr(audit_behavioral, "TRIPLE_JOIN_SCAN_DIRS", [tmp_path / "research"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_triple_join_guard()
        assert len(violations) == 0

    def test_allowlisted_file_passes(self, tmp_path, monkeypatch):
        # Files in archive/ directory are allowlisted (nested under research/)
        _mkfile(
            tmp_path / "research" / "archive" / "old.py",
            '''\
query = """
    SELECT * FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day = d.trading_day
"""
''',
        )
        monkeypatch.setattr(audit_behavioral, "TRIPLE_JOIN_SCAN_DIRS", [tmp_path / "research"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_triple_join_guard()
        assert len(violations) == 0

    def test_ignores_prose_docstring_mentioning_join_daily_features(self, tmp_path, monkeypatch):
        """Docstrings that mention 'JOIN daily_features' in English prose
        must not be flagged. SQL_KEYWORD_PATTERN previously matched bare
        'JOIN' which fired on prose like:

            \"\"\"Load orb_outcomes JOIN daily_features for one cell.\"\"\"

        The fix tightens the SQL block heuristic to require BOTH SELECT
        AND FROM tokens (real SQL has both; prose almost never does).
        """
        _mkfile(
            tmp_path / "research" / "prose.py",
            '''\
def load_lane(con, instrument, session):
    """Load orb_outcomes JOIN daily_features for one cell. Triple-join correct.

    This prose mentions JOIN daily_features for documentation purposes
    only. No SQL is in this docstring.
    """
    pass
''',
        )
        monkeypatch.setattr(audit_behavioral, "TRIPLE_JOIN_SCAN_DIRS", [tmp_path / "research"])
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        violations = audit_behavioral.check_triple_join_guard()
        assert len(violations) == 0, (
            f"Prose docstring incorrectly flagged. Violations: {violations}"
        )


# ── Check 6: Double-break look-ahead scanner ─────────────────────────


class TestDoubleBreakLookahead:
    """Check 6: double_break must not be used as filter/predictor."""

    def _patch_for_detection(self, monkeypatch, tmp_path):
        """Patch PROJECT_ROOT and strip 'test_' from allowlist.

        pytest tmp_path contains 'test_' in the directory name, which
        matches the allowlist substring check. Remove it for detection tests.
        """
        monkeypatch.setattr(audit_behavioral, "PROJECT_ROOT", tmp_path)
        clean_allowlist = audit_behavioral.DOUBLE_BREAK_ALLOWLIST - {"test_"}
        monkeypatch.setattr(audit_behavioral, "DOUBLE_BREAK_ALLOWLIST", clean_allowlist)

    def test_catches_where_double_break(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "trading_app" / "bad.py", 'query = "SELECT * FROM outcomes WHERE double_break = 1"\n')
        self._patch_for_detection(monkeypatch, tmp_path)
        violations = audit_behavioral.check_double_break_lookahead()
        assert len(violations) > 0
        assert "look-ahead" in violations[0].lower()

    def test_catches_if_double_break(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "pipeline" / "bad.py", "if double_break:\n    skip_trade()\n")
        self._patch_for_detection(monkeypatch, tmp_path)
        violations = audit_behavioral.check_double_break_lookahead()
        assert len(violations) > 0

    def test_catches_df_filter_double_break(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "scripts" / "tools" / "bad.py", "filtered = df[df.double_break == True]\n")
        self._patch_for_detection(monkeypatch, tmp_path)
        violations = audit_behavioral.check_double_break_lookahead()
        assert len(violations) > 0

    def test_passes_no_double_break(self, tmp_path, monkeypatch):
        _mkfile(tmp_path / "trading_app" / "good.py", "query = \"SELECT * FROM outcomes WHERE symbol = 'MGC'\"\n")
        self._patch_for_detection(monkeypatch, tmp_path)
        violations = audit_behavioral.check_double_break_lookahead()
        assert len(violations) == 0

    def test_allowlisted_file_passes(self, tmp_path, monkeypatch):
        # archive/ is in the allowlist (substring match on full path)
        _mkfile(tmp_path / "research" / "archive" / "old_script.py", 'query = "SELECT * WHERE double_break = 1"\n')
        self._patch_for_detection(monkeypatch, tmp_path)
        violations = audit_behavioral.check_double_break_lookahead()
        assert len(violations) == 0
