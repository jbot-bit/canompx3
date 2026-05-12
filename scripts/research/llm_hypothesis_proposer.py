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
    p.add_argument(
        "--candidate-strategy-id",
        default=None,
        help=(
            "Optional strategy_id from validated_setups to pre-screen "
            "(Mode A + graveyard + neighbor scan). If provided, the LLM gets "
            "the screen results in its context AND the proposer refuses outright "
            "if the candidate fails Mode A or hits a blocking graveyard verdict."
        ),
    )
    p.add_argument(
        "--require-screen-pass",
        action="store_true",
        default=True,
        help="Refuse to call the LLM if the candidate screen fails. Default True.",
    )
    p.add_argument(
        "--no-require-screen-pass",
        action="store_false",
        dest="require_screen_pass",
        help="Allow LLM call even if candidate screen fails. Surfaces screen results in context but does not block.",
    )
    p.add_argument(
        "--auto-run",
        action="store_true",
        help=(
            "After a clean draft is written, promote .draft.yaml -> .yaml and "
            "invoke scripts/infra/prereg-loop.sh --execute. Requires "
            "--candidate-strategy-id (so the screen evidence is on file). "
            "Default OFF — operator must explicitly opt in."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Candidate pre-screen (Phase A helpers)
# ---------------------------------------------------------------------------


def _lookup_candidate(strategy_id: str) -> dict[str, object] | None:
    """Resolve a strategy_id to a row from validated_setups or experimental_strategies.

    Returns a dict shaped for the Mode A / graveyard / neighbor helpers, or
    None if no row matches. Searches ``validated_setups`` first
    (deployed/promoted strategies); falls back to ``experimental_strategies``
    so the proposer can re-screen rejected candidates without first having
    them promoted. The latter is where the failure modes (yesterday's 3
    REJECTED pre-regs) live.
    """
    from pipeline.paths import GOLD_DB_PATH

    cols = [
        "instrument",
        "orb_label",
        "orb_minutes",
        "entry_model",
        "confirm_bars",
        "rr_target",
        "filter_type",
        "sample_size",
        "expectancy_r",
        "sharpe_ratio",
        "strategy_id",
        "source_table",
    ]

    con = _connect_read_only(str(GOLD_DB_PATH))
    try:
        rows = con.execute(
            """
            SELECT instrument, orb_label, orb_minutes, entry_model, confirm_bars,
                   rr_target, filter_type, sample_size, expectancy_r, sharpe_ratio,
                   strategy_id,
                   'validated_setups' AS source_table
            FROM validated_setups
            WHERE strategy_id = ?
            UNION ALL
            SELECT instrument, orb_label, orb_minutes, entry_model, confirm_bars,
                   rr_target, filter_type, sample_size, expectancy_r, sharpe_ratio,
                   strategy_id,
                   'experimental_strategies' AS source_table
            FROM experimental_strategies
            WHERE strategy_id = ?
            LIMIT 1
            """,
            [strategy_id, strategy_id],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return None
    return dict(zip(cols, rows[0], strict=True))


def _run_candidate_screen(candidate: dict[str, object]) -> dict[str, object]:
    """Phase A pipeline: Mode A → graveyard → neighbor scan."""
    from scripts.research.lhp.adjacency import screen_candidate_mode_a
    from scripts.research.lhp.graveyard import check_graveyard
    from scripts.research.lhp.neighbor_scan import scan_neighbors

    mode_a = screen_candidate_mode_a(candidate, strict_oos_n=True)
    graveyard = check_graveyard(candidate)
    neighbors = scan_neighbors(candidate, include_aperture=True, include_session=False)
    return {
        "candidate": candidate,
        "mode_a": mode_a,
        "graveyard": graveyard,
        "neighbor_scan": neighbors,
    }


def _format_screen_for_llm(screen: dict[str, object]) -> str:
    """Compact serialization of screen results for the LLM context block."""
    candidate = screen.get("candidate") or {}
    mode_a = screen.get("mode_a") or {}
    graveyard = screen.get("graveyard") or {}
    neighbors = screen.get("neighbor_scan") or {}

    applying_hits = [h for h in graveyard.get("hits", []) if h.get("applies_to_candidate")]
    lines = [
        "CANDIDATE SCREEN RESULTS",
        "========================",
        f"strategy_id: {candidate.get('strategy_id', '<unknown>')}",
        f"  spec: instr={candidate.get('instrument')} session={candidate.get('orb_label')} "
        f"O{candidate.get('orb_minutes')} {candidate.get('entry_model')}/CB{candidate.get('confirm_bars')} "
        f"RR{candidate.get('rr_target')} {candidate.get('filter_type')}",
        "",
        "mode_a_screen:",
        f"  passes_criterion_8: {mode_a.get('passes_criterion_8')}",
        f"  oos_is_ratio: {mode_a.get('oos_is_ratio')}",
        f"  n_oos: {mode_a.get('n_oos')}",
        f"  reason: {mode_a.get('reason')}",
        "",
        "graveyard:",
        f"  has_blocking_verdict: {graveyard.get('has_blocking_verdict')}",
        f"  has_warning: {graveyard.get('has_warning')}",
        f"  backend: {graveyard.get('backend')}",
        f"  summary: {graveyard.get('summary')}",
    ]
    if applying_hits:
        lines.append("  applying_hits:")
        for h in applying_hits[:3]:
            lines.append(f"    - [{h.get('verdict')}] {h.get('title', '?')[:100]}")
            if h.get("reopen_criteria"):
                lines.append(f"      reopen: {h['reopen_criteria'][:160]}")
    lines.extend(
        [
            "",
            "neighbor_scan:",
            f"  family_health: {neighbors.get('family_health')}",
            f"  siblings_tested: {neighbors.get('siblings_tested')}",
            f"  siblings_evaluable: {neighbors.get('siblings_evaluable')}",
            f"  siblings_killed: {neighbors.get('siblings_killed')}",
            f"  siblings_blocked_by_graveyard: {neighbors.get('siblings_blocked_by_graveyard')}",
            f"  summary: {neighbors.get('summary')}",
        ]
    )
    if neighbors.get("siblings"):
        lines.append("  details:")
        for s in (neighbors.get("siblings") or [])[:6]:
            spec = s.get("spec", {})
            lines.append(
                f"    - filter={spec.get('filter_type')} O{spec.get('orb_minutes')} "
                f"pass={s.get('mode_a_passes')} gv_block={s.get('graveyard_blocking')} "
                f"n_oos={s.get('mode_a_n_oos')}"
            )
    return "\n".join(lines)


def _build_prior_art_block(screen: dict[str, object]) -> dict[str, object]:
    """Build the prior_art dict that gets injected into the draft yaml."""
    graveyard = screen.get("graveyard") or {}
    neighbors = screen.get("neighbor_scan") or {}
    applying_hits = [h for h in graveyard.get("hits", []) if h.get("applies_to_candidate")]
    notes = []
    for h in applying_hits[:5]:
        verdict = h.get("verdict", "?")
        title = (h.get("title") or "?")[:120]
        notes.append(f"{verdict}: {title}")
    return {
        "family_health": neighbors.get("family_health", "UNKNOWN"),
        "siblings_tested": neighbors.get("siblings_tested", 0),
        "siblings_evaluable": neighbors.get("siblings_evaluable", 0),
        "siblings_killed": neighbors.get("siblings_killed", 0),
        "siblings_blocked_by_graveyard": neighbors.get("siblings_blocked_by_graveyard", 0),
        "notes": notes,
        "graveyard_summary": graveyard.get("summary"),
        "neighbor_scan_summary": neighbors.get("summary"),
    }


def _inject_prior_art(yaml_text: str, prior_art: dict[str, object]) -> str:
    """Add prior_art to the YAML if not already present.

    Done as a textual prepend rather than yaml round-trip to preserve LLM
    output formatting (quoting, multiline strings) byte-for-byte.
    """
    import yaml as _yaml

    if "prior_art:" in yaml_text.splitlines()[:50][0:50]:
        # Heuristic — if the LLM already emitted prior_art, trust it. Future
        # work: parse + merge instead of trust.
        return yaml_text
    block_dict = {"prior_art": prior_art}
    block_text = _yaml.safe_dump(block_dict, sort_keys=False, allow_unicode=True)
    return block_text + "\n" + yaml_text


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

    # ------------------------------------------------------------------
    # Phase A pre-screen: if the operator supplied a candidate strategy_id,
    # run Mode A + graveyard + neighbor scan BEFORE spending an LLM call.
    # ------------------------------------------------------------------
    screen: dict[str, object] | None = None
    screen_context_block: str | None = None
    prior_art_block: dict[str, object] | None = None
    if args.candidate_strategy_id:
        candidate = _lookup_candidate(args.candidate_strategy_id)
        if candidate is None:
            print(
                f"ERROR: candidate strategy_id {args.candidate_strategy_id!r} not in "
                "validated_setups OR experimental_strategies. Cannot screen.",
                file=sys.stderr,
            )
            return 5
        print(
            f"CANDIDATE: {candidate.get('strategy_id')} (source={candidate.get('source_table')})",
            file=sys.stderr,
        )
        print(f"PRE-SCREEN: {args.candidate_strategy_id}", file=sys.stderr)
        screen = _run_candidate_screen(candidate)
        screen_context_block = _format_screen_for_llm(screen)
        prior_art_block = _build_prior_art_block(screen)
        print(screen_context_block, file=sys.stderr)
        print("", file=sys.stderr)
        mode_a_pass = (screen.get("mode_a") or {}).get("passes_criterion_8")
        graveyard_block = (screen.get("graveyard") or {}).get("has_blocking_verdict")
        applying = any(h.get("applies_to_candidate") for h in ((screen.get("graveyard") or {}).get("hits") or []))
        if args.require_screen_pass:
            if mode_a_pass is False:
                print(
                    "REFUSED: candidate fails Mode A pre-screen. Use --no-require-screen-pass to override.",
                    file=sys.stderr,
                )
                return 1
            if graveyard_block and applying:
                print(
                    "REFUSED: candidate hit applying blocking verdict in graveyard. "
                    "Use --no-require-screen-pass and cite reopen criteria to override.",
                    file=sys.stderr,
                )
                return 1

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
        # Append the Phase A candidate screen so the LLM sees Mode A status,
        # graveyard hits, and family health alongside the active-strategies
        # adjacency. Surfaced even when the screen failed (we may have run
        # with --no-require-screen-pass).
        if screen_context_block:
            adjacency_context = adjacency_context + "\n\n" + screen_context_block

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

    # Inject the prior_art block produced by the Phase A screen so the
    # static checker (check_prior_art_block) is satisfied and any future
    # auditor can see what evidence informed the draft.
    if prior_art_block:
        yaml_text = _inject_prior_art(yaml_text, prior_art_block)

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

    if args.auto_run:
        if not args.candidate_strategy_id:
            print(
                "AUTO_RUN_REFUSED: --auto-run requires --candidate-strategy-id "
                "so the screen evidence is on file before execution.",
                file=sys.stderr,
            )
            return 7
        import shutil
        import subprocess

        # Promote .draft.yaml -> .yaml so the hypothesis loader picks it up.
        # write_draft outputs <name>.draft.yaml — strip the .draft suffix.
        draft_name = Path(written).name
        if draft_name.endswith(".draft.yaml"):
            final_name = draft_name[: -len(".draft.yaml")] + ".yaml"
        elif draft_name.endswith(".yaml"):
            final_name = draft_name
        else:
            final_name = draft_name + ".yaml"
        final_path = Path(written).parent.parent / final_name  # drafts/ -> hypotheses/
        try:
            shutil.copy2(written, final_path)
        except OSError as exc:
            print(f"AUTO_RUN_REFUSED: cannot promote draft -> hypothesis: {exc}", file=sys.stderr)
            return 7
        print(f"PROMOTED: {final_path}")

        runner = _REPO_ROOT / "scripts" / "infra" / "prereg-loop.sh"
        if not runner.is_file():
            print(f"AUTO_RUN_REFUSED: runner not found at {runner}", file=sys.stderr)
            return 7
        cmd = ["bash", str(runner), "--hypothesis-file", str(final_path), "--execute"]
        print(f"AUTO_RUN: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"AUTO_RUN_FAILED: runner exit {result.returncode}", file=sys.stderr)
            return result.returncode
        print("AUTO_RUN_COMPLETE")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
