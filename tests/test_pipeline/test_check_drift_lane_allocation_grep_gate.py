"""Tests for check_no_direct_lane_allocation_json_literals.

Companion to ``pipeline.check_drift.check_no_direct_lane_allocation_json_literals``.
Pattern: monkeypatch PROJECT_ROOT to ``tmp_path`` and clone a minimal source-
tree shape under ``tmp_path/{trading_app,scripts/tools,scripts/research,research}/``
containing files that either DO or DO NOT carry the ``lane_allocation.json``
literal.

The test also covers:
  - Per-instance violation surfacing for non-allowlisted sites
  - Allowlist suppression (permanent + temporary)
  - Dead-allowlist-entry sub-check (entry exists but no longer contains the literal)
  - Mutation-probe: temporarily remove one allowlist entry, confirm the site violates

Stage: docs/runtime/stages/2026-05-21-multi-profile-lane-allocation-stage-1b-i.md.
Companion: docs/specs/lane_allocation_schema.md § 4 (resolver contract), § 5a
(grep-gate spec).
"""

from __future__ import annotations

from pathlib import Path


def _write_py(tmp_path: Path, rel: str, body: str) -> None:
    """Write ``body`` to ``tmp_path / rel`` (creating parent dirs)."""
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def _patch_root(monkeypatch, tmp_path: Path) -> None:
    from pipeline import check_drift

    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)


class TestNoDirectLaneAllocationJsonLiterals:
    """check_no_direct_lane_allocation_json_literals enumerates direct path
    literals outside the resolver allowlist."""

    def test_passes_when_no_files(self, tmp_path, monkeypatch):
        """Empty tmp source tree => no violations."""
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _patch_root(monkeypatch, tmp_path)
        assert check_no_direct_lane_allocation_json_literals() == []

    def test_passes_when_no_literal_anywhere(self, tmp_path, monkeypatch):
        """Source files exist but none contain the literal => no violations."""
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(tmp_path, "trading_app/foo.py", "x = 1\n")
        _write_py(tmp_path, "scripts/tools/bar.py", "y = 2\n")
        _patch_root(monkeypatch, tmp_path)
        assert check_no_direct_lane_allocation_json_literals() == []

    def test_allowlisted_permanent_site_passes(self, tmp_path, monkeypatch):
        """A permanent-allowlist file may contain the literal — no violation."""
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        # prop_profiles.py is permanently allowlisted.
        _write_py(
            tmp_path,
            "trading_app/prop_profiles.py",
            'PATH = "docs/runtime/lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)
        assert check_no_direct_lane_allocation_json_literals() == []

    def test_allowlisted_temporary_site_passes(self, tmp_path, monkeypatch):
        """A temporary-allowlist file with the literal — no violation.

        After Stage 1b-iii, the temporary allowlist is empty by canonical
        state, so this test injects a synthetic entry to verify the
        allowlist mechanism still suppresses violations on listed files.
        """
        from pipeline import check_drift
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(
            tmp_path,
            "scripts/tools/test_only_synthetic_reader.py",
            'PATH = "docs/runtime/lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)
        monkeypatch.setattr(
            check_drift,
            "_LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST",
            frozenset({Path("scripts/tools/test_only_synthetic_reader.py")}),
        )
        assert check_no_direct_lane_allocation_json_literals() == []

    def test_non_allowlisted_site_violates(self, tmp_path, monkeypatch):
        """Injection test: drop the literal into a non-allowlisted file =>
        violation surfaces with the path + line number + migration pointer.
        """
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(
            tmp_path,
            "trading_app/new_reader.py",
            '# heading\nimport json\nPATH = "docs/runtime/lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "trading_app/new_reader.py:3" in violations[0]
        assert "resolve_allocation_json" in violations[0]
        assert "Stage 1b" in violations[0]

    def test_pipeline_dir_not_scanned(self, tmp_path, monkeypatch):
        """pipeline/ is explicitly out of scope (this check itself mentions
        the literal). Files under pipeline/ never violate.
        """
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(
            tmp_path,
            "pipeline/some_module.py",
            'reference = "lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)
        assert check_no_direct_lane_allocation_json_literals() == []

    def test_scripts_research_dir_is_scanned(self, tmp_path, monkeypatch):
        """Stage 1c brings scripts/research/ into scope."""
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(
            tmp_path,
            "scripts/research/some_research.py",
            'PATH = "lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "scripts/research/some_research.py:1" in violations[0]

    def test_top_level_research_dir_is_scanned(self, tmp_path, monkeypatch):
        """Stage 1c also covers top-level research/ scripts."""
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(
            tmp_path,
            "research/current_allocation_reader.py",
            'PATH = "docs/runtime/lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "research/current_allocation_reader.py:1" in violations[0]

    def test_scripts_tools_violation(self, tmp_path, monkeypatch):
        """scripts/tools/ IS in scope. A non-allowlisted file there violates."""
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(
            tmp_path,
            "scripts/tools/something_new.py",
            'P = "lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "scripts/tools/something_new.py:1" in violations[0]

    def test_first_line_pointer(self, tmp_path, monkeypatch):
        """The violation cites the FIRST line where the literal appears."""
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        body = (
            "# top\n"
            "# middle\n"
            "# third\n"
            'PATH = "lane_allocation.json"  # first occurrence on line 4\n'
            'PATH2 = "lane_allocation.json"  # also on line 5\n'
        )
        _write_py(tmp_path, "trading_app/reader.py", body)
        _patch_root(monkeypatch, tmp_path)
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "trading_app/reader.py:4" in violations[0]

    def test_comment_only_mention_still_violates(self, tmp_path, monkeypatch):
        """Literal in a comment (not a string) STILL violates. This is
        intentional — comment mentions of the legacy path go stale after
        Stage 1d removes it; we want them flagged too.
        """
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(
            tmp_path,
            "trading_app/commented.py",
            "# this references lane_allocation.json in a comment\nx = 1\n",
        )
        _patch_root(monkeypatch, tmp_path)
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "trading_app/commented.py:1" in violations[0]

    def test_dead_temporary_allowlist_entry_violates(self, tmp_path, monkeypatch):
        """Sub-check: a temporary-allowlist file that exists but NO LONGER
        contains the literal must surface as a violation. This prevents the
        allowlist from going silently broad if a reader is migrated without
        removing the allowlist entry.
        """
        from pipeline import check_drift
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        # After Stage 1b-iii the live allowlist is empty by canonical state.
        # Inject a synthetic entry pointing at a file that exists but does
        # NOT contain the literal — simulates "reader migrated but allowlist
        # not updated", which the sub-check must surface so the allowlist
        # stays tight rather than going silently broad.
        _write_py(
            tmp_path,
            "scripts/tools/test_only_synthetic_reader.py",
            "# migrated to resolver\nfrom trading_app.prop_profiles import resolve_allocation_json\n",
        )
        _patch_root(monkeypatch, tmp_path)
        monkeypatch.setattr(
            check_drift,
            "_LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST",
            frozenset({Path("scripts/tools/test_only_synthetic_reader.py")}),
        )
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "TEMPORARY allowlist entry" in violations[0]
        assert "scripts/tools/test_only_synthetic_reader.py" in violations[0]
        assert "shrink monotonically" in violations[0]

    def test_dead_temporary_allowlist_entry_skipped_when_file_absent(self, tmp_path, monkeypatch):
        """If a temporary-allowlist file does not exist in the tree, do NOT
        flag it as dead. The deletion-of-the-actual-file commit will catch
        it via other surface (test removal, import errors, etc.).
        """
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _patch_root(monkeypatch, tmp_path)
        # No files written — all temporary-allowlist entries are absent.
        assert check_no_direct_lane_allocation_json_literals() == []

    def test_mutation_probe_removing_allowlist_entry_surfaces_site(self, tmp_path, monkeypatch):
        """Mutation-probe per feedback_injection_test_catches_float_repr_class_bug.md:
        temporarily shrink the allowlist and confirm a previously-allowed site
        now violates. This verifies the allowlist is actually consulted
        (not silently ignored).
        """
        from pipeline import check_drift
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        # After Stage 1b-iii the live allowlist is empty by canonical state,
        # so this mutation-probe injects + removes a synthetic entry. The
        # site with the literal should pass under the injected allowlist
        # and fail when we strip the entry — proving the allowlist is
        # actually consulted (not silently ignored).
        synthetic = Path("scripts/tools/test_only_synthetic_reader.py")
        _write_py(
            tmp_path,
            str(synthetic),
            'PATH = "docs/runtime/lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)

        monkeypatch.setattr(check_drift, "_LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST", frozenset({synthetic}))
        assert check_no_direct_lane_allocation_json_literals() == []

        monkeypatch.setattr(check_drift, "_LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST", frozenset())
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "scripts/tools/test_only_synthetic_reader.py:1" in violations[0]
        assert "resolve_allocation_json" in violations[0]

    def test_nested_path_under_trading_app(self, tmp_path, monkeypatch):
        """rglob picks up files in subdirectories — e.g., live/."""
        from pipeline.check_drift import check_no_direct_lane_allocation_json_literals

        _write_py(
            tmp_path,
            "trading_app/live/new_thing.py",
            'PATH = "lane_allocation.json"\n',
        )
        _patch_root(monkeypatch, tmp_path)
        violations = check_no_direct_lane_allocation_json_literals()
        assert len(violations) == 1
        assert "trading_app/live/new_thing.py:1" in violations[0]
