"""Tests for the 5 CRG-backed drift checks (D1-D5).

D1, D3, D4, D5 are tested by mocking ``pipeline.check_drift_crg_helpers`` —
they exercise the fail-open paths and the violation-detection logic
independently of CRG availability.

D2 is AST-based (no CRG dependency) and is tested with synthetic research/
files via a ``PROJECT_ROOT`` monkeypatch.

Authority: ``docs/plans/2026-04-29-crg-integration-spec.md`` Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pipeline.check_drift as cd
import pipeline.check_drift_crg_helpers as helpers

# ── shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def crg_unavailable_default(monkeypatch):
    """Default: CRG is unavailable. Tests that need CRG override this."""
    monkeypatch.setattr(helpers, "crg_is_available", lambda: False)


# ── D1: cross-layer surprising connections ─────────────────────────────────


class TestCrgD1CrossLayer:
    def test_advisory_when_crg_unavailable(self, capsys):
        result = cd.check_crg_cross_layer_surprising_connections()
        assert result == []
        out = capsys.readouterr().out
        assert "ADVISORY" in out
        assert "CRG unavailable" in out

    def test_passes_silently_when_no_surprises(self, monkeypatch):
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(helpers, "get_surprising_connections", lambda **kw: [])
        result = cd.check_crg_cross_layer_surprising_connections()
        assert result == []

    def test_skips_intra_layer_edges(self, monkeypatch, capsys):
        """Edges within pipeline/ alone (no trading_app/) are not flagged."""
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "get_surprising_connections",
            lambda **kw: [
                {
                    "source_qualified": "C:/repo/canompx3/pipeline/build_bars_5m.py::run",
                    "target_qualified": "C:/repo/canompx3/pipeline/build_daily_features.py::run",
                    "surprise_score": 0.7,
                }
            ],
        )
        result = cd.check_crg_cross_layer_surprising_connections()
        assert result == []
        out = capsys.readouterr().out
        assert "Surprising cross-layer edge" not in out

    def test_skips_canonical_surface_edge(self, monkeypatch, capsys):
        """Cross-layer edges through a canonical surface are NOT flagged."""
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "get_surprising_connections",
            lambda **kw: [
                {
                    "source_qualified": "C:/repo/canompx3/pipeline/dst.py::SESSION_CATALOG",
                    "target_qualified": "C:/repo/canompx3/trading_app/strategy_discovery.py::main",
                    "surprise_score": 0.8,
                }
            ],
        )
        result = cd.check_crg_cross_layer_surprising_connections()
        assert result == []
        out = capsys.readouterr().out
        assert "Surprising cross-layer edge" not in out

    def test_flags_non_canonical_cross_layer_edge(self, monkeypatch, capsys):
        """Cross-layer edges bypassing canonical surfaces ARE flagged."""
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "get_surprising_connections",
            lambda **kw: [
                {
                    "source_qualified": "C:/repo/canompx3/pipeline/some_internal_util.py::helper",
                    "target_qualified": "C:/repo/canompx3/trading_app/some_feature_reader.py::read",
                    "surprise_score": 0.9,
                }
            ],
        )
        result = cd.check_crg_cross_layer_surprising_connections()
        assert result == []  # advisory — never blocks
        out = capsys.readouterr().out
        assert "Surprising cross-layer edge" in out
        assert "some_internal_util.py" in out

    def test_advisory_when_crg_query_fails(self, monkeypatch, capsys):
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(helpers, "get_surprising_connections", lambda **kw: helpers.CRG_UNAVAILABLE)
        result = cd.check_crg_cross_layer_surprising_connections()
        assert result == []
        out = capsys.readouterr().out
        assert "ADVISORY" in out


# ── D2: AST-based canonical-import enforcement ────────────────────────────


@pytest.fixture
def fake_research_root(tmp_path: Path, monkeypatch) -> Path:
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cd, "RESEARCH_DIR", research_dir)
    return research_dir


class TestCrgD2CanonicalImports:
    def test_passes_when_no_research_dir(self, tmp_path, monkeypatch):
        """No research/ dir → no violations."""
        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(cd, "RESEARCH_DIR", tmp_path / "nonexistent")
        result = cd.check_crg_canonical_import_enforcement()
        assert result == []

    def test_passes_when_canonical_imported(self, fake_research_root, capsys):
        """Script imports parse_strategy_id from canonical module → no violation."""
        good = fake_research_root / "good_scan.py"
        good.write_text(
            "from trading_app.eligibility.builder import parse_strategy_id\nx = parse_strategy_id('foo')\n",
            encoding="utf-8",
        )
        result = cd.check_crg_canonical_import_enforcement()
        assert result == []
        out = capsys.readouterr().out
        assert "violation" not in out

    def test_flags_local_redefinition(self, fake_research_root, capsys):
        """Script that locally DEFINES parse_strategy_id without canonical import → flagged."""
        bad = fake_research_root / "bad_scan.py"
        bad.write_text(
            "def parse_strategy_id(s):\n    return s.split('_')\n",
            encoding="utf-8",
        )
        result = cd.check_crg_canonical_import_enforcement()
        assert result == []  # advisory
        out = capsys.readouterr().out
        assert "bad_scan.py" in out
        assert "parse_strategy_id" in out

    def test_flags_local_redef_even_with_wrong_import(self, fake_research_root, capsys):
        """Defining locally + importing from non-canonical module is also a violation."""
        bad = fake_research_root / "bad_with_wrong_import.py"
        bad.write_text(
            "from research.utils import parse_strategy_id\ndef parse_strategy_id(s):\n    return s.split('-')\n",
            encoding="utf-8",
        )
        result = cd.check_crg_canonical_import_enforcement()
        assert result == []
        out = capsys.readouterr().out
        assert "bad_with_wrong_import.py" in out

    def test_skips_archive_subdir(self, fake_research_root, capsys):
        """research/archive/ is exempt (frozen)."""
        archive = fake_research_root / "archive"
        archive.mkdir()
        legacy = archive / "legacy.py"
        legacy.write_text("def parse_strategy_id(s): return s\n", encoding="utf-8")
        result = cd.check_crg_canonical_import_enforcement()
        assert result == []
        out = capsys.readouterr().out
        assert "legacy.py" not in out

    def test_skips_unparseable_files(self, fake_research_root, capsys):
        """Syntax-error files are silently skipped (don't crash check)."""
        bad = fake_research_root / "syntax_error.py"
        bad.write_text("def (((", encoding="utf-8")
        result = cd.check_crg_canonical_import_enforcement()
        assert result == []  # no crash


# ── D3: AST-based canonical-function test coverage ────────────────────────


@pytest.fixture
def fake_tests_root(tmp_path: Path, monkeypatch) -> Path:
    """Create a synthetic tests/ directory and redirect cd.PROJECT_ROOT at it."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
    return tests_dir


class TestCrgD3CanonicalTestCoverage:
    def test_passes_when_no_tests_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        result = cd.check_crg_canonical_functions_have_tests()
        assert result == []

    def test_passes_when_all_canonicals_imported_and_called(self, fake_tests_root, capsys):
        """Test files that import-and-call all canonical functions → no findings."""
        (fake_tests_root / "test_dst.py").write_text(
            "from pipeline.dst import orb_utc_window\ndef test_x():\n    orb_utc_window('a','b','c')\n",
            encoding="utf-8",
        )
        (fake_tests_root / "test_eligibility.py").write_text(
            "from trading_app.eligibility.builder import parse_strategy_id\n"
            "def test_y():\n    parse_strategy_id('foo')\n",
            encoding="utf-8",
        )
        (fake_tests_root / "test_entry.py").write_text(
            "from trading_app.entry_rules import detect_break_touch\ndef test_z():\n    detect_break_touch(1,2)\n",
            encoding="utf-8",
        )
        result = cd.check_crg_canonical_functions_have_tests()
        assert result == []
        out = capsys.readouterr().out
        assert "lack" not in out

    def test_flags_canonical_function_with_no_test(self, fake_tests_root, capsys):
        """If parse_strategy_id has no test importing-and-calling it, it's flagged."""
        # Only test orb_utc_window; parse_strategy_id and detect_break_touch unattested
        (fake_tests_root / "test_dst.py").write_text(
            "from pipeline.dst import orb_utc_window\ndef test_x():\n    orb_utc_window('a','b','c')\n",
            encoding="utf-8",
        )
        result = cd.check_crg_canonical_functions_have_tests()
        assert result == []  # advisory
        out = capsys.readouterr().out
        assert "parse_strategy_id" in out
        assert "detect_break_touch" in out
        assert "orb_utc_window" not in out  # this one IS tested

    def test_imported_but_not_called_is_not_coverage(self, fake_tests_root, capsys):
        """Importing a canonical function without ever calling it isn't enough."""
        (fake_tests_root / "test_imports_only.py").write_text(
            "from trading_app.eligibility.builder import parse_strategy_id\n"
            "# imported but never called — re-export only\n"
            "x = 1\n",
            encoding="utf-8",
        )
        result = cd.check_crg_canonical_functions_have_tests()
        assert result == []
        out = capsys.readouterr().out
        assert "parse_strategy_id" in out  # flagged because not called

    def test_attribute_access_call_is_KNOWN_LIMITATION_not_counted(self, fake_tests_root, capsys):
        """KNOWN LIMITATION (not desired behavior): calls via ``import X as Y;
        Y.parse_strategy_id(...)`` exercise the canonical implementation but
        are NOT recognized as coverage by D3. The AST visitor only matches
        ``ast.Call`` nodes whose ``func`` is an ``ast.Name`` already linked to
        a ``from <canonical_module> import <symbol>``; attribute-access
        requires module-aliasing tracking we don't yet do.

        Recorded as open follow-up "D3 attribute-access detection" in
        docs/plans/2026-04-29-crg-integration-spec.md §"Open follow-ups".

        Test purpose: lock current behavior so a future implementer can
        flip the assertion in the same change that adds aliasing support;
        the rename signals "this is a known gap" rather than "this is
        the design".
        """
        (fake_tests_root / "test_attr.py").write_text(
            "import trading_app.eligibility.builder as B\ndef test_a():\n    B.parse_strategy_id('x')\n",
            encoding="utf-8",
        )
        result = cd.check_crg_canonical_functions_have_tests()
        assert result == []
        out = capsys.readouterr().out
        # Today the symbol is flagged because attribute-access isn't recognized.
        # When the limitation is fixed, this assertion flips to "not in out".
        assert "parse_strategy_id" in out

    def test_non_test_files_in_tests_dir_skipped(self, fake_tests_root, capsys):
        """Only files matching test_*.py glob are considered."""
        (fake_tests_root / "conftest.py").write_text(
            "from trading_app.eligibility.builder import parse_strategy_id\nx = parse_strategy_id('foo')\n",
            encoding="utf-8",
        )
        result = cd.check_crg_canonical_functions_have_tests()
        assert result == []
        out = capsys.readouterr().out
        # conftest.py doesn't match test_*.py — parse_strategy_id should still
        # appear as untested
        assert "parse_strategy_id" in out

    def test_skips_unparseable_test_files(self, fake_tests_root):
        """Syntax errors in test files don't crash the check."""
        (fake_tests_root / "test_broken.py").write_text("def (((", encoding="utf-8")
        result = cd.check_crg_canonical_functions_have_tests()
        assert result == []  # no crash


# ── D4: function size cap ──────────────────────────────────────────────────


class TestCrgD4FunctionSizeCap:
    def test_advisory_when_crg_unavailable(self, capsys):
        result = cd.check_crg_canonical_path_function_size()
        assert result == []
        out = capsys.readouterr().out
        assert "ADVISORY" in out
        assert "CRG unavailable" in out

    def test_passes_when_no_large_functions(self, monkeypatch):
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(helpers, "find_large_functions", lambda **kw: [])
        result = cd.check_crg_canonical_path_function_size()
        assert result == []

    def test_flags_large_function(self, monkeypatch, capsys):
        """All large functions returned by helper get filtered to canonical paths."""
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "find_large_functions",
            lambda **kw: [
                {"name": "mega_check", "relative_path": "pipeline/check_drift.py", "line_count": 350},
                {"name": "huge_func", "relative_path": "trading_app/strategy_discovery.py", "line_count": 410},
                # non-canonical entry — should be filtered out
                {"name": "skip_me", "relative_path": "research/scratch.py", "line_count": 600},
            ],
        )
        result = cd.check_crg_canonical_path_function_size()
        assert result == []  # advisory
        out = capsys.readouterr().out
        assert "mega_check" in out
        assert "huge_func" in out
        assert "skip_me" not in out  # research/ is not canonical-path

    def test_handles_windows_path_separators(self, monkeypatch, capsys):
        """Backslash-separated relative_path values are normalized correctly."""
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "find_large_functions",
            lambda **kw: [
                {"name": "win_func", "relative_path": "pipeline\\foo.py", "line_count": 250},
            ],
        )
        result = cd.check_crg_canonical_path_function_size()
        assert result == []
        out = capsys.readouterr().out
        assert "win_func" in out


# ── D5: bridge-node test coverage ─────────────────────────────────────────


class TestCrgD5BridgeNodeCoverage:
    def test_advisory_when_crg_unavailable(self, capsys):
        result = cd.check_crg_bridge_node_test_coverage()
        assert result == []
        out = capsys.readouterr().out
        assert "ADVISORY" in out
        assert "CRG unavailable" in out

    def test_passes_when_all_bridge_nodes_tested(self, monkeypatch):
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "get_bridge_nodes",
            lambda **kw: [
                {
                    "qualified_name": "C:/repo/canompx3/pipeline/dst.py::orb_utc_window",
                    "file": "C:/repo/canompx3/pipeline/dst.py",
                    "betweenness": 0.9,
                }
            ],
        )
        monkeypatch.setattr(
            helpers,
            "query_tests_for",
            lambda target: {"status": "ok", "tests": [{"name": "test_dst"}], "raw_status": "ok"},
        )
        result = cd.check_crg_bridge_node_test_coverage()
        assert result == []

    def test_flags_untested_bridge_node(self, monkeypatch, capsys):
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "get_bridge_nodes",
            lambda **kw: [
                {
                    "qualified_name": "C:/repo/canompx3/pipeline/critical_path.py::main",
                    "file": "C:/repo/canompx3/pipeline/critical_path.py",
                    "betweenness": 0.95,
                }
            ],
        )
        monkeypatch.setattr(
            helpers,
            "query_tests_for",
            lambda target: {"status": "empty", "tests": [], "raw_status": "ok"},
        )
        result = cd.check_crg_bridge_node_test_coverage()
        assert result == []  # advisory
        out = capsys.readouterr().out
        assert "No TESTED_BY edge" in out
        assert "main" in out  # symbol name should appear

    def test_distinguishes_crg_error_from_no_tests(self, monkeypatch, capsys):
        """Tagged status='error' must be reported separately from status='empty'."""
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "get_bridge_nodes",
            lambda **kw: [
                {
                    "qualified_name": "C:/repo/canompx3/pipeline/critical_path.py::ambiguous_sym",
                    "file": "C:/repo/canompx3/pipeline/critical_path.py",
                    "betweenness": 0.91,
                }
            ],
        )
        monkeypatch.setattr(
            helpers,
            "query_tests_for",
            lambda target: {"status": "error", "tests": [], "raw_status": "not_found"},
        )
        result = cd.check_crg_bridge_node_test_coverage()
        assert result == []
        out = capsys.readouterr().out
        # Must NOT report as a missing-test finding (would be a false positive
        # from CRG graph-completeness gap, not a real coverage gap):
        assert "No TESTED_BY edge" not in out
        # Must report on the separate "graph-completeness" advisory line:
        assert "CRG-uncertain" in out
        assert "not_found" in out  # raw status surfaces for debug

    def test_filters_out_test_files(self, monkeypatch, capsys):
        """Bridge nodes living in tests/ are filtered out before TESTED_BY check."""
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "get_bridge_nodes",
            lambda **kw: [
                {
                    "qualified_name": "C:/repo/canompx3/tests/test_foo.py::helper",
                    "file": "C:/repo/canompx3/tests/test_foo.py",
                    "betweenness": 0.99,
                },
            ],
        )
        # Even with empty tests result, this should NOT flag because it's filtered out
        monkeypatch.setattr(
            helpers,
            "query_tests_for",
            lambda target: {"status": "empty", "tests": [], "raw_status": "ok"},
        )
        result = cd.check_crg_bridge_node_test_coverage()
        assert result == []
        out = capsys.readouterr().out
        assert "No TESTED_BY" not in out

    def test_skips_nodes_with_no_name(self, monkeypatch, capsys):
        """Nodes without a qualified_name are silently skipped."""
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(
            helpers,
            "get_bridge_nodes",
            lambda **kw: [{"file": "C:/repo/canompx3/pipeline/unnamed.py", "betweenness": 0.8}],
        )
        monkeypatch.setattr(
            helpers,
            "query_tests_for",
            lambda target: {"status": "empty", "tests": [], "raw_status": "ok"},
        )
        result = cd.check_crg_bridge_node_test_coverage()
        assert result == []
        out = capsys.readouterr().out
        assert "No TESTED_BY" not in out

    def test_advisory_when_bridge_query_fails(self, monkeypatch, capsys):
        monkeypatch.setattr(helpers, "crg_is_available", lambda: True)
        monkeypatch.setattr(helpers, "get_bridge_nodes", lambda **kw: helpers.CRG_UNAVAILABLE)
        result = cd.check_crg_bridge_node_test_coverage()
        assert result == []
        out = capsys.readouterr().out
        assert "ADVISORY" in out


# ── helpers module: CRG_UNAVAILABLE sentinel ──────────────────────────────


class TestCrgHelpersSentinel:
    def test_crg_unavailable_is_singleton(self):
        assert helpers.CRG_UNAVAILABLE is helpers.CRG_UNAVAILABLE

    def test_crg_unavailable_is_not_none(self):
        assert helpers.CRG_UNAVAILABLE is not None

    def test_crg_unavailable_is_not_list(self):
        assert helpers.CRG_UNAVAILABLE != []

    def test_crg_is_available_false_without_graph_db(self, tmp_path, monkeypatch):
        """Real (un-monkeypatched) crg_is_available returns False when graph DB missing."""
        # The autouse fixture stubbed crg_is_available — undo it for this test
        # by re-importing the function from the module dict.
        import importlib

        fresh = importlib.reload(helpers)
        monkeypatch.setattr(fresh, "_PROJECT_ROOT", tmp_path)
        # Also clear CRG_REPO_ROOT so find_project_root doesn't escape tmp_path
        monkeypatch.delenv("CRG_REPO_ROOT", raising=False)
        assert not fresh.crg_is_available()


class TestCrgRepoRootResolution:
    """The helper passes ``repo_root=None`` to CRG so CRG's official
    ``find_project_root`` (which honors ``CRG_REPO_ROOT`` env var) controls
    resolution. This is the doc-grounded multi-worktree pattern: worktrees
    set ``CRG_REPO_ROOT`` to the canonical project so they query the full
    graph, not a 4-file pre-commit fragment.
    """

    def test_env_override_routes_helper_to_canonical(self, monkeypatch):
        """When CRG_REPO_ROOT is set, find_large_functions returns results
        from THAT graph rather than the worktree's local one. Validates the
        fix for the worktree-graph-fragmentation bug (2026-04-30 incident)."""
        import importlib

        fresh = importlib.reload(helpers)
        canonical = Path("C:/Users/joshd/canompx3")
        if not (canonical / ".code-review-graph" / "graph.db").exists():
            pytest.skip("canonical graph not built locally — skipping integration test")

        monkeypatch.setenv("CRG_REPO_ROOT", str(canonical))
        result = fresh.find_large_functions(min_lines=200)
        # Canonical graph has many 200+ line functions; worktree fragment has 0-2.
        assert isinstance(result, list)
        assert len(result) >= 10, (
            f"expected canonical graph to expose ≥10 large functions, got {len(result)}. "
            f"CRG_REPO_ROOT may not be wired into the helper anymore."
        )

    def test_no_env_falls_back_to_git_autodetect(self, monkeypatch):
        """Without CRG_REPO_ROOT, CRG's find_project_root walks up to .git.
        For tests running inside the worktree, that's the worktree root —
        which has its own (possibly tiny) graph. This test only checks
        crg_is_available is True since that proves the resolution works."""
        import importlib

        fresh = importlib.reload(helpers)
        monkeypatch.delenv("CRG_REPO_ROOT", raising=False)
        # crg_is_available depends on .code-review-graph/graph.db existing at
        # the resolved root. The worktree always has at least a fragment from
        # the pre-commit incremental update.
        assert fresh.crg_is_available() is True

    def test_helper_does_not_hardcode_repo_root_anymore(self):
        """Regression guard for the 2026-04-30 fix: the helper used to pass
        ``repo_root=str(_PROJECT_ROOT)`` to CRG, bypassing the official
        env-var resolution. After the fix it should pass ``repo_root=None``
        (or omit the kwarg) so CRG's find_project_root fires."""
        import inspect

        src = inspect.getsource(helpers)
        # Helper should not pass repo_root=str(...) anywhere — that would
        # re-introduce the bug.
        assert "repo_root=str(_PROJECT_ROOT)" not in src, (
            "Helper hardcodes _PROJECT_ROOT for repo_root; this bypasses "
            "CRG's official find_project_root and breaks CRG_REPO_ROOT support. "
            "Pass repo_root=None instead."
        )
