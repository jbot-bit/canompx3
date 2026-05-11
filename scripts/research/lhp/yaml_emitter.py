"""Write a draft hypothesis YAML to ``docs/audit/hypotheses/drafts/<...>.yaml``.

The ``drafts/`` subdirectory is the publishing gate. ``trading_app.hypothesis_loader``
discovers files via ``directory.glob("*.yaml")`` — a non-recursive call on the
hypothesis registry directory only (verified at ``trading_app/hypothesis_loader.py:139``).
The subdirectory is invisible to that glob, so drafts cannot be promoted by
accident. To publish, the human moves the file up one level:

    mv docs/audit/hypotheses/drafts/2026-05-11-llm-foo.yaml \\
       docs/audit/hypotheses/2026-05-11-llm-foo.yaml

The earlier ``.draft.yaml`` suffix design was scrapped because
``Path.glob("*.yaml")`` matches ``a.draft.yaml`` (verified empirically in
tests — see test_draft_suffix history); a subdirectory is the only zero-loader-
change way to guarantee invisibility.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

_DRAFT_SUBDIR_NAME = "drafts"


REVIEW_HEADER = """# REVIEW CHECKLIST (LLM-generated draft — do NOT commit until verified)
# 1. [ ] Each theory_citation maps to a real file in docs/institutional/literature/
#         (script checked existence; YOU verify the citation actually says what
#         the economic_basis claims it says)
# 2. [ ] economic_basis is a mechanism, not a pattern description
# 3. [ ] kill_criteria are FALSIFIABLE
# 4. [ ] No proposed filter is in the banned-features list (auto-checked but verify)
# 5. [ ] expected_trial_count is justifiable (script defaults to Bailey-strict)
# 6. [ ] holdout_date == 2026-01-01 (Mode A)
# 7. [ ] If applicable: research_question_type, role{} block, data_source_mode set
#
# After review, move from drafts/ up one level to publish:
#   mv docs/audit/hypotheses/drafts/<this-file> docs/audit/hypotheses/<this-file>
# Then commit and run:
#   scripts/infra/prereg-loop.sh --hypothesis-file <published-path> --execute
#
"""


def default_draft_path(out_dir: Path, slug: str, today: date | None = None) -> Path:
    """Build the canonical draft path for ``slug``.

    Format: ``<out_dir>/drafts/YYYY-MM-DD-llm-<slug>.yaml``. The ``drafts``
    subdirectory is what makes the file invisible to the canonical
    hypothesis-discovery glob.
    """
    d = today or date.today()
    safe_slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in slug.lower())
    return out_dir / _DRAFT_SUBDIR_NAME / f"{d.isoformat()}-llm-{safe_slug}.yaml"


def write_draft(
    yaml_text: str,
    out_path: Path,
    *,
    overwrite: bool = False,
    warnings: list[str] | None = None,
) -> Path:
    """Write the draft with the review-checklist header prepended."""
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Draft already exists: {out_path}. Pass overwrite=True or pick a new slug.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [REVIEW_HEADER]
    if warnings:
        parts.append("# WARNINGS (non-fatal):\n")
        for w in warnings:
            parts.append(f"#   - {w}\n")
        parts.append("#\n")
    parts.append(yaml_text if yaml_text.endswith("\n") else yaml_text + "\n")
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


def write_rejected(yaml_text: str, out_path: Path, failures_summary: str) -> Path:
    """Write a ``<name>.rejected.txt`` file for audit when fatal checks fire.

    The ``.rejected.txt`` extension keeps the file out of every YAML-aware tool
    (no accidental rename to ``.yaml``, no editor schema validation noise) while
    still landing it next to the draft for the human to diff against.
    """
    rejected = out_path.with_suffix(".rejected.txt")
    rejected.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# REJECTED LLM DRAFT — fatal static checks failed.\n"
        "# This file is NOT a valid hypothesis. Do not rename to .yaml.\n"
        "# Failure summary:\n" + "\n".join(f"#   {line}" for line in failures_summary.splitlines()) + "\n#\n"
    )
    rejected.write_text(header + yaml_text + "\n", encoding="utf-8")
    return rejected


__all__ = ["REVIEW_HEADER", "default_draft_path", "write_draft", "write_rejected"]
