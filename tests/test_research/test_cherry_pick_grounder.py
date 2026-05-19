"""Unit tests for cherry_pick_grounder.

Covers:
- load_bridge_draft: malformed input failures
- build_search_query: mechanism-key composition
- parse_llm_grounding_output: refusal sentinel, missing keys, empty values,
  markdown fences, valid two-key YAML
- ground_bridge_draft happy path (mocked LLM + corpus)
- ground_bridge_draft NO_LOCAL_LIT path (zero search hits)
- ground_bridge_draft LLM_REFUSED path
- ground_bridge_draft INVALID_OUTPUT path (malformed LLM YAML)
- ground_bridge_draft CONTENT_MISMATCH path (citation exists but mechanism mismatch)
- Field-presence trap defense: theory_citation is NEVER written empty/null/whitespace
- grounded_output_path naming
- write_grounded refuses overwrite
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.research import cherry_pick_grounder as grounder
from scripts.research.lhp.llm_client import (
    LLMRefusalToGround,
    ProposerResult,
    set_mock_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_bridge_draft(
    tmp_path: Path,
    *,
    strategy_id: str = "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5_O5",
    session: str = "NYSE_OPEN",
    entry_model: str = "E2",
    filter_type: str = "ORB_G5",
    direction: str = "long",
) -> Path:
    """Write a bridge-style draft YAML to tmp_path."""
    drafts = tmp_path / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    path = drafts / f"2026-05-19-{strategy_id.lower().replace('_', '-')}-chordia-heavyweight-v1.draft.yaml"
    payload: dict[str, Any] = {
        "metadata": {
            "theory_grant": False,
            "name": "test_draft_v1",
            "is_triage_screen": False,
        },
        "scope": {
            "instrument": "MNQ",
            "strategy_id": strategy_id,
            "session": session,
            "orb_minutes": 5,
            "entry_model": entry_model,
            "confirm_bars": 1,
            "rr_target": 1.5,
            "direction": direction,
            "filter_type": filter_type,
        },
        "grounding": {
            "filter_grounding_status": {
                "verdict": "UNSUPPORTED",
                "basis": "bridge cannot author citations",
            },
        },
        "primary_schema": {
            "chordia_threshold_basis": (
                "Criterion 4 no-theory strict threshold (t >= 3.79)"
            ),
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_lit_corpus(tmp_path: Path) -> Path:
    """Write a tiny literature corpus with one mechanism-bearing extract."""
    lit_dir = tmp_path / "literature"
    lit_dir.mkdir(parents=True, exist_ok=True)
    (lit_dir / "harris_2002_microstructure.md").write_text(
        "# Harris 2002 Trading Exchanges Microstructure\n\n"
        "Stop cascade liquidity demand explains opening range breakout "
        "entries when momentum traders push price through E2 breakout levels "
        "during the NYSE_OPEN session. Adverse selection dominates spread "
        "widening on directional momentum. The stop-order ladder cascade "
        "triggers continuation patterns as liquidity providers withdraw and "
        "directional flow accelerates through ORB_G5 filtered breakout points.\n",
        encoding="utf-8",
    )
    (lit_dir / "unrelated_paper.md").write_text(
        "# Unrelated Paper\n\n"
        "This paper discusses something entirely different -- portfolio "
        "construction with quarterly rebalancing on equity indices.\n",
        encoding="utf-8",
    )
    return lit_dir


def _llm_response(text: str) -> ProposerResult:
    """Build a minimal ProposerResult around a YAML string."""
    return ProposerResult(
        yaml_text=text,
        model="mock-model",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        elapsed_s=0.01,
    )


@pytest.fixture(autouse=True)
def _clear_mock():
    """Ensure no mock leaks between tests."""
    set_mock_response(None)
    yield
    set_mock_response(None)


# ---------------------------------------------------------------------------
# load_bridge_draft
# ---------------------------------------------------------------------------


class TestLoadBridgeDraft:
    def test_loads_valid_draft(self, tmp_path):
        path = _write_bridge_draft(tmp_path)
        draft = grounder.load_bridge_draft(path)
        assert draft["scope"]["strategy_id"].startswith("MNQ_NYSE_OPEN_E2")

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            grounder.load_bridge_draft(tmp_path / "nope.yaml")

    def test_raises_on_non_mapping(self, tmp_path):
        path = tmp_path / "not_mapping.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a YAML mapping"):
            grounder.load_bridge_draft(path)

    def test_raises_on_missing_scope(self, tmp_path):
        path = tmp_path / "no_scope.yaml"
        path.write_text("metadata:\n  name: test\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing scope"):
            grounder.load_bridge_draft(path)


# ---------------------------------------------------------------------------
# build_search_query
# ---------------------------------------------------------------------------


class TestBuildSearchQuery:
    def test_includes_mechanism_keys(self):
        scope = {
            "instrument": "MNQ",
            "session": "NYSE_OPEN",
            "entry_model": "E2",
            "filter_type": "ORB_G5",
            "direction": "long",
            "orb_minutes": 5,
        }
        q = grounder.build_search_query(scope)
        assert "NYSE_OPEN" in q
        assert "E2" in q
        assert "ORB_G5" in q
        assert "long" in q

    def test_omits_non_mechanism_keys(self):
        scope = {"instrument": "MGC", "orb_minutes": 30, "entry_model": "E1"}
        q = grounder.build_search_query(scope)
        assert "MGC" not in q
        assert "30" not in q
        assert "E1" in q

    def test_handles_empty_scope(self):
        assert grounder.build_search_query({}) == ""


# ---------------------------------------------------------------------------
# parse_llm_grounding_output
# ---------------------------------------------------------------------------


class TestParseLLMGroundingOutput:
    def test_parses_valid_yaml(self):
        text = (
            'theory_citation: "harris_2002_microstructure"\n'
            'economic_basis: |\n  Stop cascade liquidity demand.\n'
        )
        parsed = grounder.parse_llm_grounding_output(text)
        assert parsed is not None
        assert parsed["theory_citation"] == "harris_2002_microstructure"
        assert "Stop cascade" in parsed["economic_basis"]

    def test_strips_markdown_fences(self):
        text = (
            "```yaml\n"
            'theory_citation: "x"\n'
            'economic_basis: "y mechanism here"\n'
            "```"
        )
        parsed = grounder.parse_llm_grounding_output(text)
        assert parsed is not None

    def test_returns_none_on_refusal(self):
        assert grounder.parse_llm_grounding_output(
            "REFUSE: no_literature_match"
        ) is None

    def test_returns_none_on_empty_citation(self):
        """FIELD-PRESENCE TRAP DEFENSE: empty citation must never parse OK."""
        text = 'theory_citation: ""\neconomic_basis: "x"\n'
        assert grounder.parse_llm_grounding_output(text) is None

    def test_returns_none_on_whitespace_citation(self):
        """FIELD-PRESENCE TRAP DEFENSE: whitespace-only citation rejected."""
        text = 'theory_citation: "   "\neconomic_basis: "x"\n'
        assert grounder.parse_llm_grounding_output(text) is None

    def test_returns_none_on_null_citation(self):
        text = "theory_citation: null\neconomic_basis: x\n"
        assert grounder.parse_llm_grounding_output(text) is None

    def test_returns_none_on_missing_basis(self):
        text = 'theory_citation: "x"\n'
        assert grounder.parse_llm_grounding_output(text) is None

    def test_returns_none_on_invalid_yaml(self):
        assert grounder.parse_llm_grounding_output(
            "not: : yaml: : at all"
        ) is None

    def test_returns_none_on_non_mapping(self):
        assert grounder.parse_llm_grounding_output("- list\n- form") is None


# ---------------------------------------------------------------------------
# grounded_output_path
# ---------------------------------------------------------------------------


class TestGroundedOutputPath:
    def test_draft_suffix_replaced(self, tmp_path):
        draft = tmp_path / "foo-chordia-heavyweight-v1.draft.yaml"
        out = grounder.grounded_output_path(draft)
        assert out.name == "foo-chordia-heavyweight-v1.grounded.yaml"

    def test_plain_yaml_suffix(self, tmp_path):
        draft = tmp_path / "foo.yaml"
        out = grounder.grounded_output_path(draft)
        assert out.name == "foo.grounded.yaml"


# ---------------------------------------------------------------------------
# write_grounded
# ---------------------------------------------------------------------------


class TestWriteGrounded:
    def test_writes_with_review_header(self, tmp_path):
        grounded = {"metadata": {"theory_grant": True}, "scope": {"x": 1}}
        out = tmp_path / "out.grounded.yaml"
        grounder.write_grounded(grounded, out)
        body = out.read_text(encoding="utf-8")
        assert "LITERATURE-GROUNDED DRAFT" in body
        assert "theory_grant: true" in body

    def test_refuses_overwrite(self, tmp_path):
        out = tmp_path / "out.grounded.yaml"
        out.write_text("existing\n", encoding="utf-8")
        with pytest.raises(FileExistsError):
            grounder.write_grounded({"x": 1}, out)


# ---------------------------------------------------------------------------
# ground_bridge_draft -- end-to-end with mocked LLM
# ---------------------------------------------------------------------------


class TestGroundBridgeDraftHappyPath:
    def test_emits_grounded_yaml_on_pass(self, tmp_path):
        draft = _write_bridge_draft(tmp_path)
        lit_dir = _write_lit_corpus(tmp_path)
        # LLM returns a valid two-key YAML pointing at harris_2002_microstructure
        # with overlapping mechanism terms ("stop", "cascade", "liquidity",
        # "breakout", "momentum") so verify_citation_content passes.
        set_mock_response(_llm_response(
            'theory_citation: "harris_2002_microstructure"\n'
            "economic_basis: |\n"
            "  Stop cascade liquidity demand mechanism: momentum traders "
            "push price through ORB breakout levels, triggering stop-order "
            "ladder cascade as liquidity providers withdraw on directional "
            "flow through ORB_G5 filtered breakouts.\n"
        ))

        result = grounder.ground_bridge_draft(
            draft, literature_dir=lit_dir, top_k=3
        )

        assert result.verdict == grounder.VERDICT_GROUNDED
        assert result.out_path is not None
        assert result.out_path.exists()
        assert result.out_path.name.endswith(".grounded.yaml")

        body = result.out_path.read_text(encoding="utf-8")
        # Strip the review header before yaml.safe_load
        yaml_start = body.find("metadata:")
        loaded = yaml.safe_load(body[yaml_start:])
        assert loaded["metadata"]["theory_grant"] is True
        assert loaded["metadata"]["theory_citation"] == "harris_2002_microstructure"
        assert loaded["grounding"]["filter_grounding_status"]["verdict"] == "LITERATURE_GROUNDED"
        # provenance block lets future auditors replay the decision
        prov = loaded["grounding"]["literature_grounding_provenance"]
        assert prov["verification_passes"] is True
        assert prov["source"] == "scripts/research/cherry_pick_grounder.py"

    def test_original_draft_untouched_on_pass(self, tmp_path):
        draft = _write_bridge_draft(tmp_path)
        lit_dir = _write_lit_corpus(tmp_path)
        original_bytes = draft.read_bytes()
        set_mock_response(_llm_response(
            'theory_citation: "harris_2002_microstructure"\n'
            "economic_basis: |\n"
            "  Stop cascade liquidity demand momentum breakout mechanism "
            "through ORB_G5 filtered directional flow cascade.\n"
        ))
        grounder.ground_bridge_draft(draft, literature_dir=lit_dir)
        assert draft.read_bytes() == original_bytes


class TestGroundBridgeDraftRefusals:
    def test_no_local_lit_when_corpus_irrelevant(self, tmp_path):
        """Search returns zero results -> NO_LOCAL_LIT verdict, no LLM call."""
        draft = _write_bridge_draft(
            tmp_path, filter_type="ZZZ_UNKNOWN_MECHANISM_TOKEN_ABCXYZ",
            session="NONEXISTENT_SESSION_FOOBAR"
        )
        lit_dir = _write_lit_corpus(tmp_path)
        # Don't set a mock -- if the grounder calls the LLM despite zero
        # search hits, the test will fail with a real network call attempt.

        result = grounder.ground_bridge_draft(draft, literature_dir=lit_dir)

        assert result.verdict == grounder.VERDICT_NO_LOCAL_LIT
        assert result.out_path is not None
        assert result.out_path.name.endswith(".rejected.txt")
        # The .grounded.yaml MUST NOT exist
        grounded_yaml = grounder.grounded_output_path(draft)
        assert not grounded_yaml.exists()

    def test_llm_refused_when_sentinel_raised(self, tmp_path):
        draft = _write_bridge_draft(tmp_path)
        lit_dir = _write_lit_corpus(tmp_path)
        set_mock_response(LLMRefusalToGround("REFUSE: no_literature_match"))

        result = grounder.ground_bridge_draft(draft, literature_dir=lit_dir)

        assert result.verdict == grounder.VERDICT_LLM_REFUSED
        assert result.out_path is not None
        assert result.out_path.name.endswith(".rejected.txt")

    def test_invalid_output_when_llm_returns_garbage(self, tmp_path):
        draft = _write_bridge_draft(tmp_path)
        lit_dir = _write_lit_corpus(tmp_path)
        set_mock_response(_llm_response("this is not yaml at all : : :"))

        result = grounder.ground_bridge_draft(draft, literature_dir=lit_dir)

        assert result.verdict == grounder.VERDICT_INVALID_OUTPUT

    def test_invalid_output_when_citation_is_empty(self, tmp_path):
        """FIELD-PRESENCE TRAP DEFENSE: empty theory_citation never written."""
        draft = _write_bridge_draft(tmp_path)
        lit_dir = _write_lit_corpus(tmp_path)
        set_mock_response(_llm_response(
            'theory_citation: ""\neconomic_basis: "anything"\n'
        ))

        result = grounder.ground_bridge_draft(draft, literature_dir=lit_dir)

        assert result.verdict == grounder.VERDICT_INVALID_OUTPUT
        # The .grounded.yaml MUST NOT exist with empty citation
        grounded_yaml = grounder.grounded_output_path(draft)
        assert not grounded_yaml.exists()

    def test_content_mismatch_when_citation_unrelated(self, tmp_path):
        """LLM cites a real file but economic_basis tokens don't overlap body."""
        draft = _write_bridge_draft(tmp_path)
        lit_dir = _write_lit_corpus(tmp_path)
        set_mock_response(_llm_response(
            'theory_citation: "unrelated_paper"\n'
            "economic_basis: |\n"
            "  Completely unrelated mechanism: superconductor lattice "
            "phonon coupling drives quantum entanglement breakouts.\n"
        ))

        result = grounder.ground_bridge_draft(draft, literature_dir=lit_dir)

        assert result.verdict == grounder.VERDICT_CONTENT_MISMATCH
        assert result.out_path is not None
        assert result.out_path.name.endswith(".rejected.txt")


class TestVerdictLabels:
    def test_all_verdicts_listed(self):
        """Verdict constants and ALL_VERDICTS tuple must stay in sync.

        If a new VERDICT_* constant is added, ALL_VERDICTS must include it.
        Documents the closed set the journal schema column accepts.
        """
        verdict_names = {
            n for n in dir(grounder)
            if n.startswith("VERDICT_") and isinstance(getattr(grounder, n), str)
        }
        verdict_values = {getattr(grounder, n) for n in verdict_names}
        assert verdict_values == set(grounder.ALL_VERDICTS)


class TestFormatOperatorMessage:
    def test_grounded_includes_citation(self, tmp_path):
        result = grounder.GroundingResult(
            verdict=grounder.VERDICT_GROUNDED,
            draft_path=tmp_path / "x.draft.yaml",
            out_path=tmp_path / "x.grounded.yaml",
            theory_citation="harris_2002_microstructure",
            economic_basis="stop cascade",
            cited_slugs=("harris_2002_microstructure",),
            reason="ok",
        )
        msg = grounder.format_operator_message(result)
        assert "GROUNDED" in msg
        assert "harris_2002_microstructure" in msg

    def test_refused_includes_reason(self, tmp_path):
        result = grounder.GroundingResult(
            verdict=grounder.VERDICT_NO_LOCAL_LIT,
            draft_path=tmp_path / "x.draft.yaml",
            out_path=tmp_path / "x.grounded.rejected.txt",
            theory_citation=None,
            economic_basis=None,
            cited_slugs=(),
            reason="no local match",
        )
        msg = grounder.format_operator_message(result)
        assert "REFUSED" in msg
        assert "NO_LOCAL_LIT" in msg
        assert "no local match" in msg
