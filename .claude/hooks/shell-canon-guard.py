#!/usr/bin/env python3
"""Shell-canon guard: PreToolUse(PowerShell) — steer bash-equivalent work to bash.

canompx3 is bash-canonical (see .claude/rules/shell-canon.md): 81% of permission
grants, every infra script, the pre-commit hook, venv routing, and all WSL paths
are bash-native. PowerShell is reserved for genuinely Windows-only operations.

This guard fires BEFORE a PowerShell command runs. If the command is something
bash does fine (git, python, file ops, grep-equivalents), it emits a soft steer
asking Claude to use the Bash tool instead. If the command is a genuinely
Windows-only operation (allowlist below), it passes silently.

Soft-block semantics: exit 2 + stderr is a *steer*, not a hard wall — Claude
reads the message and reissues via Bash. The operator can still force PowerShell
by re-running; this only removes the reflexive ambiguity that produced repeated
shell errors.

Fail-open design: any parse error, missing field, or unexpected shape exits 0
(pass). The guard must never block a session it cannot read. Mirrors the
fail-open contract of branch-flip-guard.py / data-first-guard.py.
"""

from __future__ import annotations

import json
import re
import sys

# ---------------------------------------------------------------------------
# Allowlist: genuinely Windows-only operations PowerShell SHOULD handle.
# A match here passes silently (no steer). Kept deliberately broad so the guard
# never fights a legitimate Windows-only need — fail toward allowing.
# ---------------------------------------------------------------------------
ALLOW_PATTERNS = re.compile(
    r"""
    \bwsl\b                       # WSL bridge — inherently PowerShell/Windows
  | \bGet-Process\b               # Windows process table
  | \bStop-Process\b
  | \bGet-Item\b | \bGet-ChildItem\b   # Windows filesystem cmdlets (incl. registry)
  | \bGet-Content\b | \bSet-Content\b | \bOut-File\b
  | \bSelect-String\b             # PS-native search (operator uses on Windows paths)
  | \bMeasure-Object\b | \bForEach-Object\b | \bWhere-Object\b
  | \bSelect-Object\b | \bFormat-List\b | \bFormat-Table\b
  | \bTest-Path\b | \bJoin-Path\b | \bResolve-Path\b
  | \bNew-Item\b | \bRemove-Item\b | \bCopy-Item\b | \bMove-Item\b
  | \bGet-Service\b | \bStart-Service\b | \bStop-Service\b
  | \bGet-ScheduledTask\b | \bRegister-ScheduledTask\b | \bschtasks\b
  | \bStart-Process\b             # GUI launchers
  | \bGet-CimInstance\b | \bGet-WmiObject\b
  | \bpowercfg\b | \bPredatorSense\b
  | \.ps1\b                       # PS infra scripts
  | START_BOT | START_REMOTE | stop_live | STOP_       # project launchers
  | code-review-graph\s+update    # operator runs CRG update via PS on Windows paths
  | HKLM: | HKCU: | HKEY_         # Windows registry
  | \$env: | \$Host\b             # PS-specific environment access
  | \bcode-review-graph\b         # CRG sidecar invoked on Windows paths
  """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Steer triggers: operations bash does cleanly. A match here (and NO allowlist
# match) produces the soft steer. We look for the leading verb/tool of the
# command so we don't misfire on these names appearing as substrings of paths.
# ---------------------------------------------------------------------------
BASH_EQUIVALENT = re.compile(
    r"""
    ^\s*git\b                     # git — bash-canonical, in every infra script
  | ^\s*gh\b                      # GitHub CLI
  | ^\s*python\b | ^\s*python3\b | ^\s*pytest\b | ^\s*ruff\b | ^\s*uv\b | ^\s*pip\b
  | ^\s*node\b | ^\s*npm\b | ^\s*npx\b
  | ^\s*jq\b
  | ^\s*cat\b | ^\s*head\b | ^\s*tail\b | ^\s*ls\b | ^\s*grep\b | ^\s*rg\b
  | ^\s*sed\b | ^\s*awk\b | ^\s*find\b | ^\s*wc\b | ^\s*sort\b | ^\s*uniq\b
  | ^\s*echo\b | ^\s*mkdir\b | ^\s*mv\b | ^\s*cp\b | ^\s*rm\b | ^\s*touch\b
  | ^\s*diff\b | ^\s*curl\b | ^\s*wget\b | ^\s*make\b | ^\s*tar\b
  | ^\s*bash\b
  """,
    re.IGNORECASE | re.VERBOSE,
)

STEER_MESSAGE = (
    "SHELL-CANON STEER: this PowerShell command is a bash-equivalent operation. "
    "canompx3 is bash-canonical (.claude/rules/shell-canon.md) — git/python/file "
    "ops/search all run via the Bash tool, which avoids the PowerShell-syntax "
    "error class (here-strings, 2>&1 ErrorRecord wrapping, native-exe exit codes). "
    "Re-issue this via the Bash tool. If it is genuinely Windows-only (wsl, "
    "Get-Process, scheduled tasks, *.ps1, GUI launcher), it should match the "
    "allowlist — if it does not, the operator can re-run as-is."
)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail-open: malformed event -> pass

    tool_name = event.get("tool_name", "")
    if tool_name != "PowerShell":
        sys.exit(0)  # fail-open: matcher misfire -> pass

    command = event.get("tool_input", {}).get("command", "")
    if not isinstance(command, str) or not command.strip():
        sys.exit(0)  # fail-open: nothing to inspect

    # Genuinely Windows-only -> allow silently.
    if ALLOW_PATTERNS.search(command):
        sys.exit(0)

    # Bash-equivalent leading verb -> steer.
    if BASH_EQUIVALENT.search(command):
        print(STEER_MESSAGE, file=sys.stderr)
        sys.exit(2)

    # Unknown shape -> fail-open (pass). We only steer the clear cases.
    sys.exit(0)


if __name__ == "__main__":
    main()
