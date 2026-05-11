"""Tests for scripts.tools.research_catalog_mcp_server."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

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
        # Point the BLUEPRINT path at a non-existent file by default so the
        # parser short-circuits; tests that need it override individually.
        blueprint_stub = root / "STRATEGY_BLUEPRINT.md"
        with (
            patch.object(research_catalog_mcp_server, "LITERATURE_ROOT", literature),
            patch.object(research_catalog_mcp_server, "HYPOTHESES_ROOT", hypotheses),
            patch.object(research_catalog_mcp_server, "RESULTS_ROOT", results),
            patch.object(research_catalog_mcp_server, "BLUEPRINT_PATH", blueprint_stub),
        ):
            research_catalog_mcp_server._artifact_index_cache_clear()
            yield literature, hypotheses, results, blueprint_stub
            research_catalog_mcp_server._artifact_index_cache_clear()


def test_list_literature_sources_includes_metadata() -> None:
    with _temporary_catalog() as (literature, _hypotheses, _results, _blueprint):
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
    with _temporary_catalog() as (literature, _hypotheses, _results, _blueprint):
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
    with _temporary_catalog() as (_literature, hypotheses, results, _blueprint):
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
    with _temporary_catalog() as (_literature, hypotheses, _results, _blueprint):
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
    with _temporary_catalog() as (_literature, _hypotheses, results, _blueprint):
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
    with _temporary_catalog() as (literature, _hypotheses, _results, _blueprint):
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
    with _temporary_catalog() as (literature, _hypotheses, _results, _blueprint):
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
    with _temporary_catalog() as (literature, hypotheses, results, _blueprint):
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


# ---------------------------------------------------------------------------
# Verdict-tag filter + STRATEGY_BLUEPRINT.md §5 NO-GO indexing (Rev 2)
# ---------------------------------------------------------------------------


def _write_verdict_fixtures(results: Path) -> None:
    """Three result-kind artifacts: NO-GO, PARK, no-verdict — covering the filter axes."""
    (results / "alpha-feature-dead.md").write_text(
        "\n".join(
            [
                "---",
                "verdict: NO-GO",
                "---",
                "# Alpha Feature Dead",
                "",
                "Result body for alpha feature kill verdict.",
            ]
        ),
        encoding="utf-8",
    )
    (results / "beta-feature-park.md").write_text(
        "\n".join(
            [
                "# Beta Feature Park",
                "",
                "verdict: PARK",
                "",
                "Result body for beta feature park verdict.",
            ]
        ),
        encoding="utf-8",
    )
    (results / "gamma-feature-neutral.md").write_text(
        "# Gamma Feature\n\nResult body without any verdict marker.\n",
        encoding="utf-8",
    )


def test_search_filters_by_verdict_tag() -> None:
    with _temporary_catalog() as (_literature, _hypotheses, results, _blueprint):
        _write_verdict_fixtures(results)
        payload = research_catalog_mcp_server._search_research_catalog(
            "feature",
            verdict_tags=["NO-GO"],
        )

    ids = [item["artifact_id"] for item in payload["items"]]
    assert ids == ["alpha-feature-dead"]
    assert payload["verdict_tags"] == ["NO-GO"]


def test_search_no_verdict_filter_returns_all() -> None:
    with _temporary_catalog() as (_literature, _hypotheses, results, _blueprint):
        _write_verdict_fixtures(results)
        payload = research_catalog_mcp_server._search_research_catalog("feature")

    ids = {item["artifact_id"] for item in payload["items"]}
    assert ids == {
        "alpha-feature-dead",
        "beta-feature-park",
        "gamma-feature-neutral",
    }
    assert payload["verdict_tags"] is None


def test_artifact_index_parses_blueprint_nogo_subsections() -> None:
    with _temporary_catalog() as (_literature, _hypotheses, _results, blueprint):
        blueprint.write_text(
            "\n".join(
                [
                    "# Blueprint",
                    "",
                    "## 5. NO-GO Registry",
                    "",
                    "Some intro paragraph.",
                    "",
                    "| Path | Verdict | Evidence | What Would Reopen |",
                    "|------|---------|----------|-------------------|",
                    "| Foo filter | DEAD | Killed in audit X. | New mechanism class. |",
                    "| Bar entry model | PURGED | Three biases. | Nothing. |",
                    "| Baz observation | PROMISING | Survived first scan. | Pre-register confirmation. |",
                    "",
                    "## 6. Next section",
                    "",
                    "Outside the NO-GO scope — must not be parsed.",
                ]
            ),
            encoding="utf-8",
        )
        idx = research_catalog_mcp_server._artifact_index()

    bp_artifacts = [a for a in idx["result"] if a.metadata.get("source") == "STRATEGY_BLUEPRINT.md"]
    by_id = {a.artifact_id: a for a in bp_artifacts}
    assert by_id["blueprint-nogo-foo-filter"].metadata["verdict"] == "NO-GO"
    assert by_id["blueprint-nogo-foo-filter"].metadata["raw_verdict"] == "DEAD"
    assert by_id["blueprint-nogo-bar-entry-model"].metadata["verdict"] == "NO-GO"
    assert by_id["blueprint-nogo-bar-entry-model"].metadata["raw_verdict"] == "PURGED"
    # PROMISING is recognized but classified as PROMISING (not NO-GO) so a
    # `verdict_tags=["NO-GO"]` filter still excludes it. Indexing it (rather
    # than silently dropping) prevents the gap class caught 2026-05-12 where
    # 14 BLUEPRINT rows with qualified verdicts (DEAD (qualifier), DEAD --
    # PERMANENT, INVESTIGATED -- NO-GO, GUILTY) were invisible to search.
    assert by_id["blueprint-nogo-baz-observation"].metadata["verdict"] == "PROMISING"


def test_verdict_detected_from_filename_stem() -> None:
    with _temporary_catalog() as (_literature, _hypotheses, results, _blueprint):
        (results / "2026-04-15-multi-timeframe-chain-full-scope-nogo.md").write_text(
            "# Multi-timeframe chain full scope\n\nNo body verdict marker; stem alone.\n",
            encoding="utf-8",
        )
        idx = research_catalog_mcp_server._artifact_index()

    matches = [a for a in idx["result"] if a.artifact_id == "2026-04-15-multi-timeframe-chain-full-scope-nogo"]
    assert len(matches) == 1
    assert matches[0].metadata["verdict"] == "NO-GO"


def test_verdict_detected_from_frontmatter() -> None:
    with _temporary_catalog() as (_literature, _hypotheses, results, _blueprint):
        # Disagreeing signals: front-matter says KILL, filename suffix says -park.
        # Spec contract is "front-matter wins" — the test must exercise the
        # disagreement, not a tautological agreement (regression guard for
        # the F-2 untested-precedence finding from the 2026-05-12 review).
        (results / "ambiguous-stem-park.md").write_text(
            "\n".join(
                [
                    "---",
                    "verdict: KILL",
                    "---",
                    "# Ambiguous Stem",
                    "",
                    "Filename says -park, front-matter says KILL.",
                ]
            ),
            encoding="utf-8",
        )
        idx = research_catalog_mcp_server._artifact_index()

    matches = [a for a in idx["result"] if a.artifact_id == "ambiguous-stem-park"]
    assert len(matches) == 1
    assert matches[0].metadata["verdict"] == "KILL", (
        "Front-matter declaration must override filename suffix convention."
    )


def test_verdict_detected_from_body_marker_production_format() -> None:
    """Detect ``**Verdict:** TOKEN`` body markers used in real audit-results.

    Regression guard for the 2026-05-12 review's F-1 silent-functional-gap:
    front-matter regex alone matched 0 of 240 production audit-results. The
    canonical production format is ``**Verdict:** KILL per K1.`` — bold
    prefix + token + free-text suffix on the same line. Verified against
    44 production files. Without this detector the headline verdict_tags
    filter ships non-functional on the dataset it was designed to filter.
    """
    fixtures = {
        "result-bold-kill.md": (
            "# Audit\n\n**Verdict:** KILL per K1. Zero of 4 cells pass.\n",
            "KILL",
        ),
        "result-allcaps-bold-kill.md": (
            "# Audit\n\n**VERDICT: KILL**\n\nBody.\n",
            "KILL",
        ),
        "result-bold-park.md": (
            "# Audit\n\n**Verdict:** PARK_PENDING_OOS_POWER (CB1 cell only).\n",
            "PARK",
        ),
        "result-prefixed-section.md": (
            "# Audit\n\n## Verdict\n\n**Verdict on the live cell:** KILL per pre-reg.\n",
            "KILL",
        ),
        "result-non-kill-validated.md": (
            "# Audit\n\n**Verdict:** VALIDATED on 2 of 3 sessions.\n",
            "VALIDATED",
        ),
    }
    with _temporary_catalog() as (_literature, _hypotheses, results, _blueprint):
        for name, (body, _expected) in fixtures.items():
            (results / name).write_text(body, encoding="utf-8")
        idx = research_catalog_mcp_server._artifact_index()
    by_id = {a.artifact_id: a for a in idx["result"]}
    for name, (_body, expected) in fixtures.items():
        stem = name[: -len(".md")]
        assert stem in by_id, f"missing {stem}"
        got = by_id[stem].metadata.get("verdict")
        assert got == expected, f"{stem}: expected {expected!r}, got {got!r}"


def test_compound_verdict_qualifiers_normalize_to_kill_tag() -> None:
    """Qualified verdict cells must not be silently dropped.

    Regression guard for the 2026-05-12 silent-gap class where 14 BLUEPRINT
    NO-GO Registry rows used qualifier patterns (`DEAD (impractical)`,
    `**DEAD -- PERMANENT**`, `INVESTIGATED -- NO-GO`, `GUILTY`) and were
    skipped by an exact-match-only normalizer.
    """
    cases = {
        "DEAD": "NO-GO",
        "DEAD (reversed)": "NO-GO",
        "DEAD (look-ahead artifact)": "NO-GO",
        "**DEAD -- PERMANENT**": "NO-GO",
        "INVESTIGATED -- NO-GO": "NO-GO",
        "PURGED": "NO-GO",
        "RETIRED": "NO-GO",
        "GUILTY": "NO-GO",
        "ARITHMETIC_ONLY": "NO-GO",
        "PARK": "PARK",
        "KILL": "KILL",
        "NULL (distinctness survives)": "NULL",
        "PROMISING (upgraded from observation)": "PROMISING",
        "totally unknown verdict": None,
        "": None,
    }
    for raw, expected in cases.items():
        got = research_catalog_mcp_server._normalize_verdict(raw)
        assert got == expected, f"verdict {raw!r} normalized to {got!r}, expected {expected!r}"


def test_search_rejects_unknown_verdict_tag() -> None:
    with _temporary_catalog() as (_literature, _hypotheses, results, _blueprint):
        _write_verdict_fixtures(results)
        with pytest.raises(ValueError, match="Unknown verdict_tag"):
            research_catalog_mcp_server._search_research_catalog(
                "feature",
                verdict_tags=["BOGUS"],
            )
