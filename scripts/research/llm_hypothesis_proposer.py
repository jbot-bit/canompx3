#!/usr/bin/env python3
"""LLM Hypothesis Proposer CLI — Track A entry point.

Produces a ``.draft.yaml`` candidate pre-registration at
``docs/audit/hypotheses/YYYY-MM-DD-llm-<slug>.draft.yaml``.

Workflow
--------
1. Load literature corpus (read-only)
2. Connect read-only to gold.db, list active validated_setups
3. Build LLM context (system prompt + fewshot + corpus summary + adjacency)
4. Call LLM (or load fixture in --dry-run)
5. Run static checks (delegating to canonical sources)
6. Write .draft.yaml on pass, .draft.yaml.rejected on fatal-fail

The script never writes to gold.db, never modifies an existing hypothesis
file, and never touches pipeline/ or trading_app/ runtime state.

Exit codes
----------
0  draft written, all fatal checks passed
1  LLM refused to ground (no literature match)
2  one or more fatal static checks failed (.draft.yaml.rejected written)
3  cost ceiling exceeded — no LLM call made
4  YAML did not parse / schema-load failed
5  internal error (IO, bad fixture, etc.)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Repo root: parents[2] resolves <repo>/scripts/research/llm_hypothesis_proposer.py
# -> <repo>. Anchored so the script works whether invoked from worktree root or
# from any subdirectory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env so OPENROUTER_API_KEY / ANTHROPIC_API_KEY resolve in subprocess shells
# that don't auto-source .env (e.g., Claude Code's bash). _REPO_ROOT here is the
# *worktree* root; .env is canonically at the parent canompx3 root and is gitignored
# (so it's NOT copied into worktrees). Try worktree-local first (allows overrides)
# then fall back to the canonical root next to gold.db. Fail-open: missing dotenv
# or missing .env file is not an error — llm_client raises a clear LLMRequestError
# when no key is present. Parse warnings on shell-specific lines silenced.
try:
    import logging as _logging

    from dotenv import load_dotenv  # type: ignore[import-not-found]

    from pipeline.paths import GOLD_DB_PATH as _GOLD_DB_PATH  # type: ignore[import-not-found]

    _dotenv_log = _logging.getLogger("dotenv.main")
    _prev_level = _dotenv_log.level
    _dotenv_log.setLevel(_logging.ERROR)
    try:
        for _candidate in (_REPO_ROOT / ".env", Path(_GOLD_DB_PATH).parent / ".env"):
            if _candidate.is_file():
                load_dotenv(_candidate, override=False)
    finally:
        _dotenv_log.setLevel(_prev_level)
except ImportError:
    pass

from scripts.research.lhp.adjacency import (  # noqa: E402
    _connect_read_only,
    adjacency_summary_for_llm,
    list_active_strategies,
)
from scripts.research.lhp.literature_index import (  # noqa: E402
    corpus_summary_for_llm,
    load_corpus,
)
from scripts.research.lhp.llm_client import (  # noqa: E402
    CostCeilingExceeded,
    LLMRefusalToGround,
    LLMRequestError,
    propose_with_mock_support,
)
from scripts.research.lhp.static_checks import run_all  # noqa: E402
from scripts.research.lhp.yaml_emitter import (  # noqa: E402
    default_draft_path,
    write_draft,
    write_rejected,
)

_LITERATURE_DIR = _REPO_ROOT / "docs" / "institutional" / "literature"
_DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "audit" / "hypotheses"
_DEFAULT_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "lhp"
_SYSTEM_PROMPT_PATH = _REPO_ROOT / "docs" / "prompts" / "hypothesis-proposer-system.md"
_FEWSHOT_PATH = _REPO_ROOT / "docs" / "prompts" / "hypothesis-proposer-fewshot.md"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm_hypothesis_proposer",
        description="Propose a literature-grounded pre-reg YAML via LLM.",
    )
    p.add_argument("--slug", required=True, help="Short slug for output filename.")
    p.add_argument(
        "--max-hypotheses",
        type=int,
        default=3,
        help="Hint to the LLM (it is not enforced; the schema check enforces real limits).",
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=28,
        help="Default Bailey-strict ceiling. Hard ceiling is 300 (clean) or 2000 (proxy).",
    )
    p.add_argument("--cost-ceiling", type=float, default=0.50, help="USD per call.")
    # Model default is None so the LLM client falls back to the canonical
    # ``CLAUDE_REASONING_MODEL`` from ``trading_app.ai.claude_client``. Pass
    # an explicit id (including OpenRouter slug) to override.
    p.add_argument("--model", default=None, help="LLM model id. Defaults to canonical CLAUDE_REASONING_MODEL.")
    p.add_argument(
        "--instrument",
        choices=["MNQ", "MES", "MGC"],
        default=None,
        help="Optionally narrow LLM context to one instrument.",
    )
    p.add_argument(
        "--user-instruction",
        default="Propose a literature-grounded edge worth pre-registering.",
        help="The free-text instruction sent to the LLM.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the LLM call; load YAML from --fixture instead.",
    )
    p.add_argument(
        "--fixture",
        default=None,
        help="Path to a YAML fixture to use under --dry-run. Defaults to tests/fixtures/lhp/good_yaml_1.yaml.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Override output directory.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing draft with the same name.",
    )
    return p


def _emit_failures(failures: list, file: object | None = None) -> None:
    """Print failures. ``file`` defaults to the *current* sys.stderr, evaluated
    at call time (not at import time) so pytest's ``capsys`` stream replacement
    is honoured."""
    stream = file if file is not None else sys.stderr
    fatal = [f for f in failures if f.fatal]
    warns = [f for f in failures if not f.fatal]
    if fatal:
        print("Fatal static-check failures:", file=stream)
        for f in fatal:
            print(f"  [{f.code}] {f.field}: {f.detail}", file=stream)
    if warns:
        print("Non-fatal warnings:", file=stream)
        for f in warns:
            print(f"  [{f.code}] {f.field}: {f.detail}", file=stream)


def _format_failures_summary(failures: list) -> str:
    return "\n".join(f"[{f.code}] {f.field}: {f.detail}" for f in failures)


def _load_fixture(args: argparse.Namespace) -> str:
    fixture_path = Path(args.fixture) if args.fixture else _DEFAULT_FIXTURE_DIR / "good_yaml_1.yaml"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    return fixture_path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    try:
        corpus = load_corpus(_LITERATURE_DIR)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 5

    if args.dry_run:
        try:
            yaml_text = _load_fixture(args)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 5
    else:
        if not _SYSTEM_PROMPT_PATH.is_file() or not _FEWSHOT_PATH.is_file():
            print(
                f"ERROR: system prompt or fewshot file missing. Expected {_SYSTEM_PROMPT_PATH} and {_FEWSHOT_PATH}.",
                file=sys.stderr,
            )
            return 5
        system_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        fewshot = _FEWSHOT_PATH.read_text(encoding="utf-8")
        corpus_summary = corpus_summary_for_llm(corpus)

        try:
            from pipeline.paths import GOLD_DB_PATH  # imported late so --dry-run works in CI without DB

            con = _connect_read_only(str(GOLD_DB_PATH))
            try:
                active = list_active_strategies(con)
            finally:
                con.close()
        except Exception as exc:  # pragma: no cover - integration path
            print(f"ERROR: cannot read gold.db: {exc}", file=sys.stderr)
            return 5
        if args.instrument:
            active = [a for a in active if a.get("instrument") == args.instrument]
        adjacency_context = adjacency_summary_for_llm(active)

        try:
            result = propose_with_mock_support(
                system_prompt=system_prompt,
                fewshot=fewshot,
                corpus_summary=corpus_summary,
                adjacency_context=adjacency_context,
                user_instruction=args.user_instruction,
                cost_ceiling_usd=args.cost_ceiling,
                model=args.model,
            )
        except CostCeilingExceeded as exc:
            print(f"ERROR: cost ceiling exceeded: {exc}", file=sys.stderr)
            return 3
        except LLMRefusalToGround as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1
        except LLMRequestError as exc:
            print(f"ERROR: LLM request failed: {exc}", file=sys.stderr)
            return 5
        yaml_text = result.yaml_text

    parsed, failures = run_all(yaml_text, corpus)
    has_fatal = any(f.fatal for f in failures)

    out_path = default_draft_path(args.out_dir, args.slug, today=date.today())

    if has_fatal:
        rejected = write_rejected(yaml_text, out_path, _format_failures_summary(failures))
        _emit_failures(failures)
        print(f"REJECTED: {rejected}", file=sys.stderr)
        # Schema-load failure gets its own exit code so the operator can
        # distinguish "model emitted garbage" from "model emitted plausible
        # YAML that violated a rule".
        if any(f.code in ("YAML_PARSE_ERROR", "YAML_NOT_MAPPING", "SCHEMA_LOAD_FAILED") for f in failures):
            return 4
        return 2

    warnings = [f"[{f.code}] {f.field}: {f.detail}" for f in failures if not f.fatal]
    written = write_draft(yaml_text, out_path, overwrite=args.overwrite, warnings=warnings)
    if warnings:
        _emit_failures(failures)
    print(f"DRAFT_WRITTEN: {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
