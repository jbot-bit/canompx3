"""Shared helper for scan-result-doc headers — reads from pre-reg YAML to
prevent the 2026-04-19 copy-paste-bug class (scan cloned from a prior scan
emits its parent's pre-reg name, commit SHA, and script path in the header).

Usage:
    from research.result_doc_header import build_header
    header_lines = build_header(
        prereg_path="docs/audit/hypotheses/2026-04-19-mes-comprehensive-mode-a-feature-v1.yaml",
        script_path=__file__,
        summary_line="40 cells | CONTINUE: 0 | KILL: 40",
    )

The helper reads `title`, `slug`, `reproducibility.commit_sha`, `status`
from the pre-reg YAML. All are mandatory for a LOCKED pre-reg; the helper
raises ValueError if any are missing. This forces scan scripts to point
at a LOCKED pre-reg by construction — cloned-from-sibling scripts will
fail fast if they forget to update `prereg_path`.

Authority: `.claude/rules/backtesting-methodology.md` § RULE 11 (audit
trail), `.claude/rules/research-truth-protocol.md` § Phase 0 Literature
Grounding (pre-reg integrity gate).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_prereg(prereg_path: str) -> dict:
    p = Path(prereg_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.exists():
        raise ValueError(f"pre-reg path does not exist: {prereg_path}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_script_path(script_path: str) -> str:
    """Return repo-relative path with forward slashes, or the input unchanged
    if it can't be made relative to _REPO_ROOT.
    """
    p = Path(script_path).resolve()
    try:
        rel = p.relative_to(_REPO_ROOT)
        return str(rel).replace("\\", "/")
    except ValueError:
        return script_path.replace("\\", "/")


def build_header(
    prereg_path: str,
    script_path: str,
    *,
    extra_lines: list[str] | None = None,
) -> list[str]:
    """Return a canonical result-doc header block for inclusion in a scan
    result MD.

    Arguments:
      prereg_path: repo-relative path to the LOCKED pre-reg YAML.
      script_path: canonical location of the scan script (pass `__file__`).
      extra_lines: optional additional header lines (e.g. IS window
        description). Appended AFTER the generated provenance block.

    Raises:
      ValueError on malformed or unlocked pre-reg.
    """
    cfg = _load_prereg(prereg_path)
    required = ["title", "slug", "status", "reproducibility"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(
            f"pre-reg {prereg_path} missing required top-level keys: {missing}"
        )
    if cfg["status"] != "LOCKED":
        raise ValueError(
            f"pre-reg {prereg_path} has status={cfg['status']!r}; result-doc "
            f"header build requires status=LOCKED."
        )
    repro = cfg.get("reproducibility") or {}
    commit_sha = repro.get("commit_sha")
    if not commit_sha or commit_sha == "TO_FILL_AFTER_COMMIT":
        raise ValueError(
            f"pre-reg {prereg_path} has no commit_sha stamped; cannot emit "
            f"canonical provenance header."
        )
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prereg_rel = prereg_path if not Path(prereg_path).is_absolute() else (
        str(Path(prereg_path).resolve().relative_to(_REPO_ROOT)).replace("\\", "/")
    )
    lines: list[str] = [
        f"# {cfg['title']}",
        "",
        f"**Generated:** {ts}",
        f"**Pre-reg:** `{prereg_rel}` (LOCKED, commit_sha={commit_sha})",
        f"**Script:** `{_normalize_script_path(script_path)}`",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("")
    return lines


def main() -> int:
    """Smoke-test when run directly: emits a header for the MES K=40 pre-reg."""
    lines = build_header(
        prereg_path="docs/audit/hypotheses/2026-04-19-mes-comprehensive-mode-a-feature-v1.yaml",
        script_path=__file__,
        extra_lines=["**IS window:** Mode A (trading_day < 2026-01-01)"],
    )
    for l in lines:
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
