#!/usr/bin/env python3
"""Stage-closed code-review nudge: PostToolUse(Edit|Write) — when a stage file
under ``docs/runtime/stages/`` is in ``mode: CLOSED`` and its ``scope_lock``
includes capital-class paths, emit a single-line additionalContext nudge
recommending Claude run ``/code-review`` on the staged diff before commit.

This is the soft companion to ``judgment-review-soft-block.py``:
  - Soft-block fires at PreToolUse(Bash) on ``git commit`` — blocks unreviewed
    ``[judgment]`` commits on capital-class paths.
  - This nudge fires at PostToolUse(Edit|Write) the moment a stage closes —
    BEFORE the bundling commit, so review can happen on the staged diff.

Mechanically a hook cannot dispatch a subagent. It writes to stdout, which the
Claude Code harness surfaces as ``additionalContext`` for the next turn. The
nudge is therefore a strong recommendation, not a forcing function. The
forcing-function side of the same gap lives in the soft-block.

Trigger predicates (ALL must be true to emit the nudge):

  1. The edited file path matches ``docs/runtime/stages/*.md``.
  2. After the edit, the file's YAML front-matter declares ``mode: CLOSED``.
  3. The stage's ``scope_lock`` contains at least one path that begins with a
     ``_CAPITAL_PATH_PREFIXES`` entry (sourced canonically from the sibling
     ``judgment-review-nudge.py`` via importlib shim — no inline copy).
  4. The stage body does NOT match any ``_REVIEW_MENTION_PATTERNS`` regex
     (i.e., the operator hasn't already noted a review pass).
  5. The per-slug marker file ``.claude/scratch/.stage-review-<slug>.ts`` was
     NOT touched within ``_SUPPRESS_SECONDS``.
  6. The global ``.claude/scratch/.code-review-ts`` marker is NOT fresh.

Canonical-source delegation (institutional-rigor.md § 4): the constants
``_CAPITAL_PATH_PREFIXES``, ``_REVIEW_MENTION_PATTERNS``, ``_SUPPRESS_SECONDS``
are imported at module load from the sibling ``judgment-review-nudge.py`` via
``importlib.util.spec_from_file_location`` (hyphenated filename forbids a
direct ``from ... import``). A drift parity check
(``check_stage_closed_review_nudge_capital_paths_parity``) guards against
accidental future inlining — same shape as Check 179.

Fail-open contract (branch-flip-protection.md § "Fail-safe guarantee"): every
read error, malformed event JSON, malformed YAML, missing scope_lock,
filesystem error → exit 0. The hook can never block an edit it cannot
reason about; PostToolUse hooks return guidance, not refusals.

Test seam: ``STAGE_REVIEW_SCRATCH_DIR`` env var overrides the marker-file
directory so the suppression path is testable in a tempdir. Production runs
ignore the env var.

Doctrine grounding:
  - ``.claude/rules/adversarial-audit-gate.md`` — the doctrine being mechanised.
  - ``.claude/rules/institutional-rigor.md`` § 4 (delegate to canonical sources),
    § 6 (no silent failures).
  - ``memory/project_review_enforcement_gaps_and_plan_2026_05_23.md`` § (B) —
    the plan that scoped this hook.
  - ``memory/feedback_n3_same_class_doctrine_threshold.md`` — n=3 forcing-function
    threshold for "doctrine present, mechanism missing" class.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_HOOK_DIR = Path(__file__).resolve().parent
_NUDGE_PATH = _HOOK_DIR / "judgment-review-nudge.py"
_PROJECT_ROOT = _HOOK_DIR.parent.parent
_SCRATCH_DIR_DEFAULT = _PROJECT_ROOT / ".claude" / "scratch"


def _load_nudge_constants():
    """Load canonical constants from the sibling nudge via importlib shim.

    Hyphenated filename forbids a plain ``from ... import``;
    ``spec_from_file_location`` is the canonical workaround (mirrors
    ``judgment-review-soft-block.py``). Fail-closed at import: if the nudge
    cannot load, the hook degrades to no-op rather than emitting nudges from
    silently-inlined fallback values.
    """
    spec = importlib.util.spec_from_file_location(
        "judgment_review_nudge", str(_NUDGE_PATH)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"stage-closed-code-review-nudge: could not build importlib spec "
            f"for sibling nudge at {_NUDGE_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _NUDGE = _load_nudge_constants()
    _CAPITAL_PATH_PREFIXES = _NUDGE._CAPITAL_PATH_PREFIXES
    _REVIEW_MENTION_PATTERNS = _NUDGE._REVIEW_MENTION_PATTERNS
    _SUPPRESS_SECONDS = _NUDGE._SUPPRESS_SECONDS
except Exception:
    print(
        "[stage-closed-code-review-nudge] WARN: could not load canonical "
        "constants from judgment-review-nudge.py; hook is a no-op.",
        file=sys.stderr,
    )
    sys.exit(0)


_GLOBAL_CODE_REVIEW_MARKER = "code-review-ts"


def _scratch_dir() -> Path:
    """Resolve scratch dir, honoring the test env override."""
    override = os.environ.get("STAGE_REVIEW_SCRATCH_DIR")
    if override:
        return Path(override)
    return _SCRATCH_DIR_DEFAULT


def _slug_marker_path(slug: str) -> Path:
    """Per-slug suppression marker — prevents re-firing on subsequent edits."""
    return _scratch_dir() / f".stage-review-{slug}.ts"


def _global_marker_path() -> Path:
    """Global suppression marker — operator says "review was just done"."""
    return _scratch_dir() / f".{_GLOBAL_CODE_REVIEW_MARKER}"


def _marker_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return (time.time() - path.stat().st_mtime) < _SUPPRESS_SECONDS
    except Exception:
        return False


def _touch_marker(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except Exception as exc:
        print(
            f"[stage-closed-code-review-nudge] WARN: could not touch marker "
            f"{path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _normalize(p: str) -> str:
    """Normalize to forward-slash relative-from-project-root path."""
    fwd = p.replace("\\", "/")
    root_fwd = str(_PROJECT_ROOT).replace("\\", "/")
    if fwd.startswith(root_fwd):
        return fwd[len(root_fwd):].lstrip("/")
    return fwd


def _is_stage_file(file_path: str) -> tuple[bool, str]:
    """Return (is_stage_file, slug). Slug is the filename stem without `.md`.

    Match by path components rather than project-root prefix so the hook works
    both for production (under ``_PROJECT_ROOT``) and for tests (under
    ``tmp_path``). The signal is "this file lives under a ``docs/runtime/stages``
    directory and ends in ``.md``".
    """
    if not file_path:
        return False, ""
    norm = _normalize(file_path)
    if not norm.endswith(".md"):
        return False, ""
    parts = Path(norm).parts
    for i in range(len(parts) - 3):
        if parts[i:i + 3] == ("docs", "runtime", "stages"):
            stem = Path(norm).stem
            return (True, stem) if stem else (False, "")
    return False, ""


def _parse_mode(content: str) -> str:
    """Extract `mode:` value from YAML front-matter. Returns lowercase string or ''."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("mode:"):
            value = stripped.split(":", 1)[1].strip()
            value = value.strip("'\"").strip()
            return value.upper()
    return ""


def _parse_scope_lock(content: str) -> list[str]:
    """Parse scope_lock paths from either YAML key or `## Scope Lock` section.

    Returns forward-slash relative paths. Tolerates both list and inline forms.
    Mirrors the parser shape of ``.claude/hooks/stage-gate-guard.py`` so a
    stage file accepted by stage-gate-guard is also parseable here.
    """
    paths: list[str] = []

    if "## Scope Lock" in content:
        section = content.split("## Scope Lock", 1)[1].split("\n##", 1)[0]
        for line in section.splitlines():
            cleaned = line.strip().lstrip("- ").strip("`").strip()
            if cleaned and not cleaned.startswith("#"):
                paths.append(cleaned.replace("\\", "/"))
        if paths:
            return paths

    in_scope = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("scope_lock:"):
            rest = stripped.split(":", 1)[1].strip()
            if rest.startswith("["):
                items = rest.strip("[]").split(",")
                for item in items:
                    cleaned = item.strip().strip("'\"").replace("\\", "/")
                    if cleaned:
                        paths.append(cleaned)
                return paths
            in_scope = True
            continue
        if in_scope:
            if stripped.startswith("- "):
                cleaned = stripped[2:].strip().strip("'\"").replace("\\", "/")
                if cleaned:
                    paths.append(cleaned)
            elif stripped and not stripped.startswith("#"):
                break
    return paths


def _touches_capital_class(scope_paths: list[str]) -> list[str]:
    """Return the subset of scope paths that match a capital-class prefix."""
    hits: list[str] = []
    for p in scope_paths:
        if any(p.startswith(pfx) for pfx in _CAPITAL_PATH_PREFIXES):
            hits.append(p)
    return hits


def _body_mentions_review(content: str) -> bool:
    """Match any canonical review-mention regex against the full file content."""
    for pat in _REVIEW_MENTION_PATTERNS:
        if pat.search(content):
            return True
    return False


def _read_event() -> dict:
    """Read the PostToolUse JSON event from stdin. Empty dict on any failure."""
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def main() -> int:
    event = _read_event()
    if not event:
        return 0
    file_path = event.get("tool_input", {}).get("file_path", "")
    is_stage, slug = _is_stage_file(file_path)
    if not is_stage:
        return 0

    abs_path = Path(file_path)
    if not abs_path.is_absolute():
        abs_path = _PROJECT_ROOT / file_path
    try:
        content = abs_path.read_text(encoding="utf-8")
    except Exception:
        return 0

    mode = _parse_mode(content)
    if mode != "CLOSED":
        return 0

    scope_paths = _parse_scope_lock(content)
    if not scope_paths:
        return 0
    capital_hits = _touches_capital_class(scope_paths)
    if not capital_hits:
        return 0

    if _body_mentions_review(content):
        return 0

    if _marker_fresh(_global_marker_path()):
        return 0
    slug_marker = _slug_marker_path(slug)
    if _marker_fresh(slug_marker):
        return 0

    _touch_marker(slug_marker)

    truth_layer = any(
        p.startswith("trading_app/live/")
        or p.startswith("trading_app/risk_manager.py")
        or p.startswith("trading_app/execution_engine.py")
        or p.startswith("trading_app/session_orchestrator.py")
        for p in capital_hits
    )
    review_skill = "/capital-review" if truth_layer else "/code-review"

    hit_list = ", ".join(capital_hits[:3])
    suffix = " ..." if len(capital_hits) > 3 else ""
    print(
        f"[stage-closed code-review nudge] stage {slug} closed on capital-class scope "
        f"({hit_list}{suffix}). Run {review_skill} on the staged diff vs origin/main "
        f"before commit. (Suppress for 60m by `touch {slug_marker}`.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
