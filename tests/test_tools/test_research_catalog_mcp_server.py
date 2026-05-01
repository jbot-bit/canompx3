"""Tests for scripts.tools.research_catalog_mcp_server."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts.tools import research_catalog_mcp_server


@contextmanager
def _temporary_catalog():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        literature = root / "literature"
        hypotheses = root / "hypotheses"
        results = root / "results"
        literature.mkdir()
        hypotheses.mkdir()
        results.mkdir()
        with (
            patch.object(research_catalog_mcp_server, "LITERATURE_ROOT", literature),
            patch.object(research_catalog_mcp_server, "HYPOTHESES_ROOT", hypotheses),
            patch.object(research_catalog_mcp_server, "RESULTS_ROOT", results),
        ):
            research_catalog_mcp_server._artifact_index_cache_clear()
            yield literature, hypotheses, results
            research_catalog_mcp_server._artifact_index_cache_clear()


def test_list_literature_sources_includes_metadata() -> None:
    with _temporary_catalog() as (literature, _hypotheses, _results):
        (literature / "chordia.md").write_text(
            "\n".join(
                [
                    "# Chordia 2018",
                    "",
                    "**Source:** `resources/chordia.pdf`",
                    "**Authors:** Tarun Chordia",
                    "**Date:** 2018",
                    "**Criticality:** HIGH",
                    "",
                    "Multiple-testing discipline for strategy sweeps.",
                ]
            ),
            encoding="utf-8",
        )

        payload = research_catalog_mcp_server._list_literature_sources()

    assert payload["total_sources"] == 1
    assert payload["items"][0]["artifact_id"] == "chordia"
    assert payload["items"][0]["authors"] == "Tarun Chordia"


def test_get_literature_excerpt_honors_section_hint() -> None:
    with _temporary_catalog() as (literature, _hypotheses, _results):
        (literature / "harvey.md").write_text(
            "\n".join(
                [
                    "# Harvey Liu 2015",
                    "",
                    "**Date:** 2015",
                    "",
                    "Intro paragraph.",
                    "",
                    "## Thresholds",
                    "",
                    "Use t-statistic hurdles above naive 1.96.",
                    "",
                    "## Other",
                    "",
                    "Secondary section.",
                ]
            ),
            encoding="utf-8",
        )

        payload = research_catalog_mcp_server._get_literature_excerpt("harvey", section_hint="threshold")

    assert payload["artifact_id"] == "harvey"
    assert "## Thresholds" in payload["excerpt"]
    assert "naive 1.96" in payload["excerpt"]
    assert "Secondary section." not in payload["excerpt"]


def test_list_open_hypotheses_uses_exact_result_match_heuristic() -> None:
    with _temporary_catalog() as (_literature, hypotheses, results):
        (hypotheses / "closed-one.yaml").write_text(
            "\n".join(
                [
                    "metadata:",
                    '  name: "closed-one"',
                    '  purpose: "Has a matching result file."',
                    '  date_locked: "2026-04-16"',
                    '  holdout_date: "2026-01-01"',
                ]
            ),
            encoding="utf-8",
        )
        (hypotheses / "open-one.yaml").write_text(
            "\n".join(
                [
                    "metadata:",
                    '  name: "open-one"',
                    '  purpose: "No result yet."',
                    '  date_locked: "2026-04-17"',
                ]
            ),
            encoding="utf-8",
        )
        (results / "closed-one.md").write_text("# Closed One\n", encoding="utf-8")

        payload = research_catalog_mcp_server._list_open_hypotheses()

    assert payload["total_open_hypotheses"] == 1
    assert payload["items"][0]["artifact_id"] == "open-one"
    assert payload["status_basis"].startswith("Open means no exact filename stem match")


def test_get_hypothesis_artifact_reads_yaml_metadata() -> None:
    with _temporary_catalog() as (_literature, hypotheses, _results):
        (hypotheses / "garch-g0.yaml").write_text(
            "\n".join(
                [
                    "# comment",
                    "metadata:",
                    '  name: "garch-g0-preflight"',
                    '  purpose: "Verify the object before downstream work."',
                    '  date_locked: "2026-04-16T21:20:00+10:00"',
                    '  holdout_date: "2026-01-01"',
                    '  testing_mode: "family"',
                ]
            ),
            encoding="utf-8",
        )

        payload = research_catalog_mcp_server._get_hypothesis_artifact("garch-g0")

    assert payload["title"] == "garch-g0-preflight"
    assert payload["holdout_date"] == "2026-01-01"
    assert "Verify the object" in payload["excerpt"]


def test_get_audit_result_reads_prereg_reference() -> None:
    with _temporary_catalog() as (_literature, _hypotheses, results):
        (results / "garch-g0-preflight.md").write_text(
            "\n".join(
                [
                    "# Garch G0 Preflight",
                    "",
                    "**Date:** 2026-04-16 11:16 AEST",
                    "**Pre-registration:** `docs/audit/hypotheses/garch-g0.yaml`",
                    "",
                    "Result summary paragraph.",
                    "",
                    "## Verdict",
                    "",
                    "SURVIVED SCRUTINY: yes",
                ]
            ),
            encoding="utf-8",
        )

        payload = research_catalog_mcp_server._get_audit_result("garch-g0-preflight", section_hint="Verdict")

    assert payload["pre_registration"] == "`docs/audit/hypotheses/garch-g0.yaml`"
    assert "## Verdict" in payload["excerpt"]
    assert "SURVIVED SCRUTINY" in payload["excerpt"]


def test_artifact_index_cache_invalidates_when_file_added() -> None:
    """Adding a file mid-process must be reflected without manual cache_clear.

    Hardens against the project's "no stale shit" rule for read-only knowledge
    surfaces in long-running MCP processes.
    """
    with _temporary_catalog() as (literature, _hypotheses, _results):
        first = research_catalog_mcp_server._list_literature_sources()
        assert first["total_sources"] == 0

        (literature / "new_paper.md").write_text(
            "# New Paper\n\n**Date:** 2026-05-01\n\nFresh content.\n",
            encoding="utf-8",
        )

        second = research_catalog_mcp_server._list_literature_sources()

    assert second["total_sources"] == 1
    assert second["items"][0]["artifact_id"] == "new_paper"


def test_artifact_index_cache_invalidates_when_file_modified() -> None:
    """Editing an existing file mid-process must update the served excerpt."""
    with _temporary_catalog() as (literature, _hypotheses, _results):
        target = literature / "evolving.md"
        target.write_text("# Evolving\n\nFirst version.\n", encoding="utf-8")
        first = research_catalog_mcp_server._get_literature_excerpt("evolving")
        assert "First version" in first["excerpt"]

        # bump mtime explicitly so this still passes on filesystems with
        # coarse mtime granularity (FAT32, some networked FSes)
        import os
        import time

        new_mtime = target.stat().st_mtime + 5
        os.utime(target, (new_mtime, new_mtime))
        target.write_text("# Evolving\n\nSecond version.\n", encoding="utf-8")
        os.utime(target, (new_mtime + 5, new_mtime + 5))
        time.sleep(0)  # allow stat refresh on platforms that defer

        second = research_catalog_mcp_server._get_literature_excerpt("evolving")

    assert "Second version" in second["excerpt"]
    assert "First version" not in second["excerpt"]


def test_search_research_catalog_returns_ranked_hits() -> None:
    with _temporary_catalog() as (literature, hypotheses, results):
        (literature / "chordia.md").write_text(
            "# Chordia Thresholds\n\nChordia multiple-testing thresholds.\n",
            encoding="utf-8",
        )
        (hypotheses / "garch-g0.yaml").write_text(
            "\n".join(
                [
                    "metadata:",
                    '  name: "garch-g0-preflight"',
                    '  purpose: "Preflight the garch research object."',
                ]
            ),
            encoding="utf-8",
        )
        (results / "garch-g0-preflight.md").write_text(
            "# Garch G0 Preflight\n\nPreflight verdict for the garch program.\n",
            encoding="utf-8",
        )

        payload = research_catalog_mcp_server._search_research_catalog("garch preflight", limit=3)

    ids = {(item["kind"], item["artifact_id"]) for item in payload["items"]}
    assert ("hypothesis", "garch-g0") in ids
    assert ("result", "garch-g0-preflight") in ids
