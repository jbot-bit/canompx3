"""Tests for scripts/tools/drain_worktrees.ps1 classification + capital-prefix parity.

The PowerShell script's branch classifier (DRAIN/CAPITAL/DIVERGED/MERGED) and its
DRY-RUN-by-default safety are the load-bearing behaviors. We test:
  1. Capital-prefix parity: the PS mirror list == the canonical Python source.
  2. The script is dry-run by default (no `push origin` reachable without -Execute).
  3. The classification ORDER (DIVERGED before CAPITAL) is present — a stale branch
     must not be mislabeled CAPITAL.

We assert against the script TEXT (not a live PS run) so the test is hermetic,
fast, and OS-portable. A full behavioral run is covered by the live dry-run smoke
in the stage file's verification.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "tools" / "drain_worktrees.ps1"
_CANONICAL_HOOK = _ROOT / ".claude" / "hooks" / "judgment-review-nudge.py"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _canonical_capital_prefixes() -> list[str]:
    """Extract _CAPITAL_PATH_PREFIXES tuple literal from the canonical Python hook."""
    txt = _CANONICAL_HOOK.read_text(encoding="utf-8")
    m = re.search(r"_CAPITAL_PATH_PREFIXES\s*=\s*\((.*?)\)", txt, re.DOTALL)
    assert m, "canonical _CAPITAL_PATH_PREFIXES not found in judgment-review-nudge.py"
    return re.findall(r'"([^"]+)"', m.group(1))


def _ps_capital_prefixes() -> list[str]:
    """Extract the $CapitalPathPrefixes @(...) mirror from the PowerShell script."""
    txt = _script_text()
    m = re.search(r"\$CapitalPathPrefixes\s*=\s*@\((.*?)\)", txt, re.DOTALL)
    assert m, "$CapitalPathPrefixes mirror not found in drain_worktrees.ps1"
    return re.findall(r"'([^']+)'", m.group(1))


def test_script_exists():
    assert _SCRIPT.exists(), "drain_worktrees.ps1 missing"


def test_capital_prefix_parity_with_canonical_source():
    """The PS mirror MUST match the canonical Python list exactly (order-insensitive).

    This is the institutional-rigor §4 guard: the capital classifier must not drift
    from the canonical _CAPITAL_PATH_PREFIXES. If this fails, edit the Python source
    first, then re-mirror.
    """
    canonical = set(_canonical_capital_prefixes())
    mirror = set(_ps_capital_prefixes())
    assert mirror == canonical, (
        f"capital-prefix drift: PS mirror {mirror} != canonical {canonical}. "
        "Edit judgment-review-nudge.py first, then re-mirror in drain_worktrees.ps1."
    )


def test_dry_run_is_default_no_unattended_push():
    """Operator decision 2026-06-10: DRY-RUN ONLY, no unattended push.

    The push call MUST be guarded behind -Execute. We assert: (a) param has -Execute
    switch, (b) the early `if (-not $Execute) { ...; exit 0 }` guard exists BEFORE any
    `push origin`, so no push is reachable without the flag.
    """
    txt = _script_text()
    assert "[switch]$Execute" in txt, "-Execute switch param missing"
    # The dry-run guard must appear before the first push.
    guard_idx = txt.find("if (-not $Execute)")
    push_idx = txt.find("push origin")
    assert guard_idx != -1, "dry-run guard (if (-not $Execute)) missing"
    assert push_idx != -1, "no push call found (script should be able to push under -Execute)"
    assert guard_idx < push_idx, "dry-run guard must precede any push (no push reachable in dry-run)"


def test_diverged_checked_before_capital():
    """DIVERGED must be classified BEFORE CAPITAL, else stale branches mislabel.

    A branch behind main can't FF-push; its capital-ness is moot. The classifier
    must `continue` on DIVERGED before computing capital hits.
    """
    txt = _script_text()
    diverged_idx = txt.find('$plan.DIVERGED +=')
    cap_idx = txt.find('$capHits = Get-CapitalHits $b')
    assert diverged_idx != -1 and cap_idx != -1
    assert diverged_idx < cap_idx, (
        "DIVERGED classification must precede CAPITAL hit computation "
        "(else stale branches flood CAPITAL with main's own changes)"
    )


def test_never_force_pushes():
    """No push COMMAND may carry --force (capital-safety invariant).

    We scan only lines that actually invoke `git ... push`, ignoring the docstring
    (which legitimately says "never --force"). A push line with --force/-f fails.
    """
    txt = _script_text()
    push_lines = [
        ln for ln in txt.splitlines()
        if "push" in ln and "git" in ln and not ln.strip().startswith("#")
    ]
    for ln in push_lines:
        assert "--force" not in ln and " -f" not in ln, f"force-push found: {ln.strip()}"
    assert "push origin --delete" not in txt, "drain_worktrees must not delete remote branches"
