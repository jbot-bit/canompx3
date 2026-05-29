"""Injection tests for check_dsr_universe_lock_declared (Criterion 5 Amendment 3.5).

Each test writes a prereg YAML into a tempdir and asserts the check fires (or
stays dormant) — proving the check actually tests each declared dimension
(integrity-guardian § 7: verify drift checks via known-violation injection).
"""

import textwrap

from pipeline.check_drift import check_dsr_universe_lock_declared


def _write(tmp_path, name, body, *, subdir=None):
    target_dir = tmp_path if subdir is None else (tmp_path / subdir)
    target_dir.mkdir(parents=True, exist_ok=True)
    p = target_dir / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


_VALID_BLOCK = """\
metadata:
  name: test-prereg
criterion_5:
  reference_family: "MNQ NYSE_PREOPEN O30 NFP-spillover v1"
  pre_2026_only: true
  failures_and_siblings_included: true
  effective_trials: 27
  effective_trials_derivation: declared_K_conservative
hypotheses:
  - id: H1
"""


def test_complete_valid_block_passes(tmp_path):
    _write(tmp_path, "2026-05-29-valid.yaml", _VALID_BLOCK)
    assert check_dsr_universe_lock_declared(hypotheses_dir=tmp_path) == []


def test_onc_clustered_derivation_passes(tmp_path):
    body = _VALID_BLOCK.replace("declared_K_conservative", "onc_clustered")
    _write(tmp_path, "2026-05-29-onc.yaml", body)
    assert check_dsr_universe_lock_declared(hypotheses_dir=tmp_path) == []


def test_no_block_is_dormant(tmp_path):
    """The corpus default — no criterion_5 block — must NOT fire."""
    _write(
        tmp_path,
        "2026-05-29-noblock.yaml",
        """\
        metadata:
          name: ordinary-prereg
        hypotheses:
          - id: H1
        """,
    )
    assert check_dsr_universe_lock_declared(hypotheses_dir=tmp_path) == []


def test_block_present_top_level_also_triggers(tmp_path):
    """criterion_5 may sit top-level, not only under metadata."""
    body = """\
    metadata:
      name: top-level-block
    criterion_5:
      reference_family: "fam"
      pre_2026_only: true
      failures_and_siblings_included: true
      effective_trials: 10
      effective_trials_derivation: onc_clustered
    hypotheses:
      - id: H1
    """
    _write(tmp_path, "2026-05-29-toplevel.yaml", body)
    assert check_dsr_universe_lock_declared(hypotheses_dir=tmp_path) == []


def test_block_not_a_mapping_blocks(tmp_path):
    _write(
        tmp_path,
        "2026-05-29-scalar.yaml",
        """\
        metadata:
          name: scalar-block
        criterion_5: "yes I pinned it, trust me"
        hypotheses:
          - id: H1
        """,
    )
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert len(v) == 1 and "must be a mapping" in v[0]


def test_missing_reference_family_blocks(tmp_path):
    body = _VALID_BLOCK.replace('  reference_family: "MNQ NYSE_PREOPEN O30 NFP-spillover v1"\n', "")
    _write(tmp_path, "2026-05-29-nofam.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("reference_family" in line for line in v)


def test_empty_reference_family_blocks(tmp_path):
    body = _VALID_BLOCK.replace(
        '  reference_family: "MNQ NYSE_PREOPEN O30 NFP-spillover v1"', '  reference_family: "   "'
    )
    _write(tmp_path, "2026-05-29-emptyfam.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("reference_family" in line for line in v)


def test_pre_2026_only_false_blocks(tmp_path):
    body = _VALID_BLOCK.replace("  pre_2026_only: true", "  pre_2026_only: false")
    _write(tmp_path, "2026-05-29-not2026.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("pre_2026_only" in line for line in v)


def test_pre_2026_only_missing_blocks(tmp_path):
    body = _VALID_BLOCK.replace("  pre_2026_only: true\n", "")
    _write(tmp_path, "2026-05-29-no2026.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("pre_2026_only" in line for line in v)


def test_failures_included_false_blocks(tmp_path):
    body = _VALID_BLOCK.replace("  failures_and_siblings_included: true", "  failures_and_siblings_included: false")
    _write(tmp_path, "2026-05-29-winnersonly.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("failures_and_siblings_included" in line for line in v)


def test_attestation_string_true_blocks(tmp_path):
    """An explicit bool is required — the string "true" must NOT satisfy it."""
    body = _VALID_BLOCK.replace("  pre_2026_only: true", '  pre_2026_only: "true"')
    _write(tmp_path, "2026-05-29-strtrue.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("pre_2026_only" in line for line in v)


def test_effective_trials_missing_blocks(tmp_path):
    body = _VALID_BLOCK.replace("  effective_trials: 27\n", "")
    _write(tmp_path, "2026-05-29-notrials.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("effective_trials" in line for line in v)


def test_effective_trials_zero_blocks(tmp_path):
    body = _VALID_BLOCK.replace("  effective_trials: 27", "  effective_trials: 0")
    _write(tmp_path, "2026-05-29-zerotrials.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("effective_trials" in line for line in v)


def test_effective_trials_bool_blocks(tmp_path):
    """YAML `true` is int-like in Python; the bool guard must reject it."""
    body = _VALID_BLOCK.replace("  effective_trials: 27", "  effective_trials: true")
    _write(tmp_path, "2026-05-29-booltrials.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("effective_trials" in line for line in v)


def test_bad_derivation_label_blocks(tmp_path):
    body = _VALID_BLOCK.replace(
        "  effective_trials_derivation: declared_K_conservative", "  effective_trials_derivation: vibes"
    )
    _write(tmp_path, "2026-05-29-baddriv.yaml", body)
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("effective_trials_derivation" in line for line in v)


def test_drafts_directory_excluded(tmp_path):
    """A malformed block inside drafts/ must NOT fire — drafts are author-owned."""
    _write(
        tmp_path,
        "2026-05-29-draft.yaml",
        """\
        metadata:
          name: draft-wip
        criterion_5:
          reference_family: ""
        hypotheses:
          - id: H1
        """,
        subdir="drafts",
    )
    assert check_dsr_universe_lock_declared(hypotheses_dir=tmp_path) == []


def test_unreadable_yaml_blocks(tmp_path):
    _write(
        tmp_path,
        "2026-05-29-broken.yaml",
        "criterion_5: [unterminated\n",
    )
    v = check_dsr_universe_lock_declared(hypotheses_dir=tmp_path)
    assert any("failed to read/parse" in line for line in v)
