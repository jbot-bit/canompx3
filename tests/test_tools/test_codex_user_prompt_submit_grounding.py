from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / ".codex" / "hooks" / "user_prompt_submit_grounding.py"
    spec = importlib.util.spec_from_file_location("codex_user_prompt_submit_grounding", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules.pop("codex_user_prompt_submit_grounding", None)
    spec.loader.exec_module(module)
    return module


def test_research_prompt_injects_methodologist_and_local_lit() -> None:
    module = _load_module()

    hints = module._build_context("is this OOS edge significant after FDR?")

    joined = "\n".join(hints)
    assert "research-methodologist" in joined
    assert "docs/institutional/literature/" in joined


def test_live_prompt_injects_live_risk_auditor() -> None:
    module = _load_module()

    hints = module._build_context("review broker execution and kill flatten risk")

    assert any("live-risk-auditor" in hint for hint in hints)


def test_test_gap_prompt_injects_coverage_scout() -> None:
    module = _load_module()

    hints = module._build_context("what pytest targets cover this missing tests gap?")

    assert any("test-coverage-scout" in hint for hint in hints)


def test_superpowers_prompt_injects_canompx3_authority_guardrail() -> None:
    module = _load_module()

    hints = module._build_context("are the superpowers plugins good for this project?")

    joined = "\n".join(hints)
    assert "Superpowers" in joined
    assert "canompx3 authority wins" in joined
    assert "unmanaged worktrees" in joined


def test_debugging_prompt_injects_systematic_debugging() -> None:
    module = _load_module()

    hints = module._build_context("debug this flaky failing regression")

    assert any("superpowers:systematic-debugging" in hint for hint in hints)


def test_parallel_agent_prompt_injects_parallel_guardrail() -> None:
    module = _load_module()

    hints = module._build_context("spawn parallel subagents for independent tasks")

    joined = "\n".join(hints)
    assert "superpowers:dispatching-parallel-agents" in joined
    assert "never allow same-file" in joined


def test_completion_prompt_injects_verification_before_completion() -> None:
    module = _load_module()

    hints = module._build_context("commit and make a PR when complete")

    assert any("superpowers:verification-before-completion" in hint for hint in hints)
