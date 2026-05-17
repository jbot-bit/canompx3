"""Unit tests for the LLM Hypothesis Proposer (Track A).

All tests use mocked LLM calls — no live network. Golden fixtures live at
``tests/fixtures/lhp/``. The single canonical literature corpus on disk is
used directly because it is read-only.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.research.lhp import literature_index, llm_client, yaml_emitter  # noqa: E402
from scripts.research.lhp.literature_index import (  # noqa: E402
    citation_exists,
    load_corpus,
)
from scripts.research.lhp.static_checks import (  # noqa: E402
    check_banned_features,
    check_citations_exist,
    check_holdout_date,
    check_instruments_active,
    check_minbtl_budget,
    check_schema_load,
    check_sessions_valid,
    run_all,
)

_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "lhp"
_LITERATURE_DIR = _REPO_ROOT / "docs" / "institutional" / "literature"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(_LITERATURE_DIR)


def _read_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _parse_fixture(name: str) -> dict:
    import yaml

    return yaml.safe_load(_read_fixture(name))


# ---------- citation matching ----------


def test_citation_exists_matches_real_corpus_entry(corpus):
    assert citation_exists(corpus, "chordia_et_al_2018_two_million_strategies")


def test_citation_exists_matches_substring(corpus):
    assert citation_exists(corpus, "Fitschen 2013 Ch.3 — fitschen_2013_path_of_least_resistance")


def test_citation_exists_rejects_fabrication(corpus):
    assert not citation_exists(corpus, "Imaginary Author 2099 Definitely Not Real")


def test_citation_exists_rejects_empty(corpus):
    assert not citation_exists(corpus, "")
    assert not citation_exists(corpus, "   ")


# ---------- schema load ----------


def test_check_schema_load_accepts_known_good():
    text = _read_fixture("good_yaml_1.yaml")
    assert check_schema_load(text) == []


def test_check_schema_load_rejects_garbage():
    failures = check_schema_load("not: [valid yaml")
    assert any(f.code == "YAML_PARSE_ERROR" or f.fatal for f in failures)


def test_check_schema_load_rejects_missing_metadata():
    text = "hypotheses: []\n"
    failures = check_schema_load(text)
    assert any(f.fatal for f in failures)


# ---------- banned features (E2 look-ahead) ----------


def test_check_banned_features_catches_e2_break_bar_volume():
    parsed = _parse_fixture("bad_banned_feature.yaml")
    failures = check_banned_features(parsed)
    assert any(f.code == "BANNED_FEATURE" and f.fatal for f in failures)


def test_check_banned_features_passes_on_clean_e2():
    parsed = _parse_fixture("good_yaml_1.yaml")
    assert check_banned_features(parsed) == []


def test_check_banned_features_ignores_e1_break_bar_columns():
    parsed = _parse_fixture("bad_banned_feature.yaml")
    parsed["hypotheses"][0]["scope"]["entry_models"] = ["E1"]
    assert check_banned_features(parsed) == []


# ---------- holdout date ----------


def test_check_holdout_date_catches_wrong_date():
    parsed = _parse_fixture("bad_wrong_holdout.yaml")
    failures = check_holdout_date(parsed)
    assert any(f.code == "WRONG_HOLDOUT" and f.fatal for f in failures)


def test_check_holdout_date_accepts_sacred():
    parsed = _parse_fixture("good_yaml_1.yaml")
    assert check_holdout_date(parsed) == []


def test_check_holdout_date_rejects_post_sacred():
    parsed = _parse_fixture("good_yaml_1.yaml")
    parsed["metadata"]["holdout_date"] = "2026-06-01"
    failures = check_holdout_date(parsed)
    assert any(f.code == "WRONG_HOLDOUT" and f.fatal for f in failures)


# ---------- MinBTL budget ----------


def test_check_minbtl_budget_warns_above_strict_bound():
    parsed = _parse_fixture("good_yaml_2.yaml")  # total_expected_trials=50
    failures = check_minbtl_budget(parsed)
    # No fatal — 50 < 300 operational ceiling
    assert not any(f.fatal for f in failures)
    # But one non-fatal warning for exceeding strict Bailey E=1.0
    assert any(f.code == "MINBTL_LOOSE_OPERATIONAL" for f in failures)


def test_check_minbtl_budget_rejects_above_300_clean():
    parsed = _parse_fixture("bad_minbtl_exceeded.yaml")
    failures = check_minbtl_budget(parsed)
    assert any(f.code == "MINBTL_BUDGET_EXCEEDED" and f.fatal for f in failures)


def test_check_minbtl_budget_accepts_strict_bailey():
    parsed = _parse_fixture("good_yaml_1.yaml")  # total_expected_trials=2
    assert check_minbtl_budget(parsed) == []


# ---------- citations ----------


def test_check_citations_exist_passes_real_citations(corpus):
    parsed = _parse_fixture("good_yaml_1.yaml")
    assert check_citations_exist(parsed, corpus) == []


def test_check_citations_exist_catches_fabrication(corpus):
    parsed = _parse_fixture("bad_fabricated_citation.yaml")
    failures = check_citations_exist(parsed, corpus)
    assert any(f.code == "CITATION_NOT_FOUND" and f.fatal for f in failures)


def test_check_citations_exist_short_circuits_on_theory_grant_false(corpus):
    """Amendment 3.3: theory_grant=false short-circuits citation enforcement.

    The loader enforces the cross-rule (no prose-in-field) at load time; the
    static check has nothing to verify for honest no-theory preregs.
    """
    parsed = {
        "metadata": {"theory_grant": False},
        "hypotheses": [{"id": 1, "name": "h1"}],
    }
    assert check_citations_exist(parsed, corpus) == []


# ---------- instruments / sessions ----------


def test_check_instruments_active_rejects_dead():
    parsed = _parse_fixture("good_yaml_1.yaml")
    parsed["hypotheses"][0]["scope"]["instruments"] = ["M2K"]
    failures = check_instruments_active(parsed)
    assert any(f.code == "INSTRUMENT_INACTIVE" and f.fatal for f in failures)


def test_check_instruments_active_passes_real():
    parsed = _parse_fixture("good_yaml_1.yaml")
    assert check_instruments_active(parsed) == []


def test_check_sessions_valid_rejects_unknown():
    parsed = _parse_fixture("good_yaml_1.yaml")
    parsed["hypotheses"][0]["scope"]["sessions"] = ["NONEXISTENT_SESSION"]
    failures = check_sessions_valid(parsed)
    assert any(f.code == "SESSION_NOT_IN_CATALOG" and f.fatal for f in failures)


def test_check_sessions_valid_passes_real():
    parsed = _parse_fixture("good_yaml_1.yaml")
    assert check_sessions_valid(parsed) == []


# ---------- run_all integration ----------


def test_run_all_passes_clean_fixture(corpus):
    text = _read_fixture("good_yaml_1.yaml")
    parsed, failures = run_all(text, corpus)
    assert parsed is not None
    assert not any(f.fatal for f in failures), [(f.code, f.field, f.detail) for f in failures if f.fatal]


def test_run_all_flags_banned_feature(corpus):
    text = _read_fixture("bad_banned_feature.yaml")
    _parsed, failures = run_all(text, corpus)
    assert any(f.code == "BANNED_FEATURE" and f.fatal for f in failures)


def test_run_all_flags_wrong_holdout(corpus):
    text = _read_fixture("bad_wrong_holdout.yaml")
    _parsed, failures = run_all(text, corpus)
    assert any(f.code == "WRONG_HOLDOUT" and f.fatal for f in failures)


def test_run_all_flags_fabricated_citation(corpus):
    text = _read_fixture("bad_fabricated_citation.yaml")
    _parsed, failures = run_all(text, corpus)
    assert any(f.code == "CITATION_NOT_FOUND" and f.fatal for f in failures)


def test_run_all_flags_minbtl_over_300(corpus):
    text = _read_fixture("bad_minbtl_exceeded.yaml")
    _parsed, failures = run_all(text, corpus)
    assert any(f.code == "MINBTL_BUDGET_EXCEEDED" and f.fatal for f in failures)


# ---------- draft-subdir loader invisibility ----------


def test_drafts_subdir_not_picked_up_by_loader(tmp_path):
    """``hypothesis_loader.find_hypothesis_file_by_sha`` MUST ignore drafts/.

    The canonical glob is non-recursive: ``directory.glob("*.yaml")`` at
    ``trading_app/hypothesis_loader.py:139``. Files inside a ``drafts/``
    subdirectory are not matched.
    """
    from trading_app.hypothesis_loader import (
        compute_file_sha,
        find_hypothesis_file_by_sha,
    )

    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    draft = drafts_dir / "2099-01-01-llm-test.yaml"
    draft.write_text(_read_fixture("good_yaml_1.yaml"), encoding="utf-8")
    sha = compute_file_sha(draft)
    # Search the parent (canonical hypothesis dir behavior): drafts/ is invisible.
    assert find_hypothesis_file_by_sha(sha, search_dir=tmp_path) is None

    # Promoting (moving up one level) makes it discoverable.
    promoted = tmp_path / "2099-01-01-llm-test.yaml"
    draft.rename(promoted)
    sha_after = compute_file_sha(promoted)
    assert find_hypothesis_file_by_sha(sha_after, search_dir=tmp_path) == promoted


# ---------- emitter ----------


def test_default_draft_path_format(tmp_path):
    p = yaml_emitter.default_draft_path(tmp_path, "Foo Bar 123", today=date(2026, 5, 11))
    assert p.name == "2026-05-11-llm-foo-bar-123.yaml"
    assert p.parent.name == "drafts"
    assert p.parent.parent == tmp_path


def test_write_draft_prepends_header(tmp_path):
    out = tmp_path / "drafts" / "2026-05-11-llm-test.yaml"
    yaml_emitter.write_draft("metadata: {}\n", out)
    text = out.read_text(encoding="utf-8")
    assert "REVIEW CHECKLIST" in text
    assert "metadata: {}" in text


def test_write_draft_refuses_overwrite_without_flag(tmp_path):
    out = tmp_path / "drafts" / "2026-05-11-llm-test.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("preexisting\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        yaml_emitter.write_draft("metadata: {}\n", out)


def test_write_rejected_creates_rejected_txt(tmp_path):
    out = tmp_path / "drafts" / "2026-05-11-llm-test.yaml"
    rejected = yaml_emitter.write_rejected("body\n", out, "[X] field: detail")
    assert rejected.name == "2026-05-11-llm-test.rejected.txt"
    text = rejected.read_text(encoding="utf-8")
    assert "REJECTED LLM DRAFT" in text
    assert "[X] field: detail" in text


# ---------- LLM client ----------


def test_cost_ceiling_aborts_before_call():
    with pytest.raises(llm_client.CostCeilingExceeded):
        llm_client.propose(
            system_prompt="x" * 10,
            fewshot="y" * 10,
            corpus_summary="z" * 10,
            adjacency_context="a" * 10,
            user_instruction="b" * 10,
            cost_ceiling_usd=0.000_001,  # impossibly low
            model=llm_client.default_reasoning_model(),
        )


def test_estimate_cost_usd_finite():
    cost = llm_client.estimate_cost_usd(llm_client.default_reasoning_model(), 1000, 1000)
    assert cost > 0
    assert cost < 1.0


def test_propose_with_mock_support_passthrough():
    expected = llm_client.ProposerResult(
        yaml_text="metadata: {}\n",
        model="mock",
        input_tokens=10,
        output_tokens=10,
        cost_usd=0.0,
        elapsed_s=0.0,
    )
    llm_client.set_mock_response(expected)
    try:
        got = llm_client.propose_with_mock_support(
            system_prompt="",
            fewshot="",
            corpus_summary="",
            adjacency_context="",
            user_instruction="",
        )
        assert got is expected
    finally:
        llm_client.set_mock_response(None)


def test_propose_with_mock_refusal_raises():
    llm_client.set_mock_response(llm_client.LLMRefusalToGround("REFUSE: no_literature_match"))
    try:
        with pytest.raises(llm_client.LLMRefusalToGround):
            llm_client.propose_with_mock_support(
                system_prompt="",
                fewshot="",
                corpus_summary="",
                adjacency_context="",
                user_instruction="",
            )
    finally:
        llm_client.set_mock_response(None)


# ---------- CLI dry-run ----------


def test_cli_dry_run_writes_draft(tmp_path, capsys):
    from scripts.research import llm_hypothesis_proposer as cli

    rc = cli.main(
        [
            "--slug",
            "smoke",
            "--dry-run",
            "--fixture",
            str(_FIXTURES / "good_yaml_1.yaml"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "DRAFT_WRITTEN:" in captured.out
    drafts = list((tmp_path / "drafts").glob("*-llm-smoke.yaml"))
    assert len(drafts) == 1


def test_cli_dry_run_rejects_banned_feature(tmp_path):
    from scripts.research import llm_hypothesis_proposer as cli

    rc = cli.main(
        [
            "--slug",
            "smoke-bad",
            "--dry-run",
            "--fixture",
            str(_FIXTURES / "bad_banned_feature.yaml"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2  # fatal static-check failure
    rejected = list((tmp_path / "drafts").glob("*-llm-smoke-bad.rejected.txt"))
    assert len(rejected) == 1


def test_cli_dry_run_rejects_fabricated_citation(tmp_path):
    from scripts.research import llm_hypothesis_proposer as cli

    rc = cli.main(
        [
            "--slug",
            "smoke-cite",
            "--dry-run",
            "--fixture",
            str(_FIXTURES / "bad_fabricated_citation.yaml"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_cli_dry_run_rejects_minbtl_over_300(tmp_path):
    from scripts.research import llm_hypothesis_proposer as cli

    rc = cli.main(
        [
            "--slug",
            "smoke-minbtl",
            "--dry-run",
            "--fixture",
            str(_FIXTURES / "bad_minbtl_exceeded.yaml"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2
