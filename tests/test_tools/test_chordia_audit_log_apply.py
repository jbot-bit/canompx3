from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.tools.chordia_audit_log_apply import apply_reviewed_proposal


def _write_audit_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "default_has_theory: false",
                "audit_freshness_days: 90",
                "theory_grants: []",
                "audits:",
                "  - strategy_id: EXISTING",
                "    audit_date: 2026-05-01",
                "    verdict: PASS_CHORDIA",
                "    t_stat: 4.0",
                "    threshold: 3.79",
                "    sample_size: 100",
                "    note: |",
                "      existing",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_proposal(path: Path, *, verdict: str = "PASS_CHORDIA", proposal_only: bool = True) -> None:
    payload = {
        "proposal_only": proposal_only,
        "target": "docs/runtime/chordia_audit_log.yaml",
        "audits": [
            {
                "strategy_id": "NEW_PASS",
                "audit_date": "2026-05-31",
                "verdict": verdict,
                "has_theory": False,
                "t_stat": 4.085,
                "sample_size": 1539,
                "source_result": "docs/audit/results/new-pass.md",
                "note": "proposal only",
            },
            {
                "strategy_id": "EXISTING",
                "audit_date": "2026-05-31",
                "verdict": "FAIL_CHORDIA",
                "has_theory": False,
                "t_stat": 3.2,
                "sample_size": 1200,
                "source_result": "docs/audit/results/existing.md",
                "note": "proposal only",
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_duplicate_proposal(path: Path) -> None:
    payload = {
        "proposal_only": True,
        "target": "docs/runtime/chordia_audit_log.yaml",
        "audits": [
            {
                "strategy_id": "DUPLICATE",
                "audit_date": "2026-05-31",
                "verdict": "PASS_CHORDIA",
                "has_theory": False,
                "t_stat": 4.1,
                "sample_size": 1000,
                "source_result": "docs/audit/results/duplicate-a.md",
            },
            {
                "strategy_id": "DUPLICATE",
                "audit_date": "2026-05-31",
                "verdict": "FAIL_CHORDIA",
                "has_theory": False,
                "t_stat": 3.2,
                "sample_size": 1000,
                "source_result": "docs/audit/results/duplicate-b.md",
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_apply_reviewed_proposal_dry_run_does_not_write(tmp_path: Path) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    proposal = tmp_path / "proposal.yaml"
    _write_audit_log(audit_log)
    _write_proposal(proposal)

    before = audit_log.read_text(encoding="utf-8")
    result = apply_reviewed_proposal(proposal_path=proposal, audit_log_path=audit_log)

    assert result.applied_count == 1
    assert result.skipped_existing_count == 1
    assert result.live_mutation is False
    assert result.validated_setups_mutation is False
    assert audit_log.read_text(encoding="utf-8") == before


def test_apply_reviewed_proposal_write_requires_reviewed(tmp_path: Path) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    proposal = tmp_path / "proposal.yaml"
    _write_audit_log(audit_log)
    _write_proposal(proposal)

    with pytest.raises(ValueError, match="--write requires --reviewed"):
        apply_reviewed_proposal(proposal_path=proposal, audit_log_path=audit_log, write=True)


def test_apply_reviewed_proposal_inserts_only_new_rows(tmp_path: Path) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    proposal = tmp_path / "proposal.yaml"
    _write_audit_log(audit_log)
    _write_proposal(proposal)

    result = apply_reviewed_proposal(
        proposal_path=proposal,
        audit_log_path=audit_log,
        reviewed=True,
        reviewed_by="test-review",
        write=True,
    )

    text = audit_log.read_text(encoding="utf-8")
    assert result.applied_count == 1
    assert result.skipped_existing_count == 1
    assert text.count("strategy_id: NEW_PASS") == 1
    assert text.count("strategy_id: EXISTING") == 1
    assert "Reviewed by: test-review." in text
    assert yaml.safe_load(text)["audits"][0]["strategy_id"] == "NEW_PASS"


def test_apply_reviewed_proposal_rejects_non_proposal(tmp_path: Path) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    proposal = tmp_path / "proposal.yaml"
    _write_audit_log(audit_log)
    _write_proposal(proposal, proposal_only=False)

    with pytest.raises(ValueError, match="proposal_only"):
        apply_reviewed_proposal(proposal_path=proposal, audit_log_path=audit_log)


def test_apply_reviewed_proposal_rejects_invalid_verdict(tmp_path: Path) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    proposal = tmp_path / "proposal.yaml"
    _write_audit_log(audit_log)
    _write_proposal(proposal, verdict="FAIL_STRICT_CHORDIA")

    with pytest.raises(ValueError, match="invalid verdict"):
        apply_reviewed_proposal(proposal_path=proposal, audit_log_path=audit_log)


def test_apply_reviewed_proposal_rejects_duplicate_strategy_ids_in_proposal(tmp_path: Path) -> None:
    audit_log = tmp_path / "chordia_audit_log.yaml"
    proposal = tmp_path / "proposal.yaml"
    _write_audit_log(audit_log)
    _write_duplicate_proposal(proposal)

    with pytest.raises(ValueError, match="duplicate strategy_id"):
        apply_reviewed_proposal(proposal_path=proposal, audit_log_path=audit_log)
