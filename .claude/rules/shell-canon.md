# Shell Canon — bash is the default shell

**Load-policy:** referenced from CLAUDE.md § Shell. Read on demand when editing
`shell-canon-guard.py` or deciding which shell tool to use.

**Authority:** ratifies measured reality. bash already won this repo — 81% of
permission grants (116 vs 21), 30 of 41 infra scripts, the entire
`.githooks/pre-commit`, venv routing, and every WSL path are bash-native.
Canonicalizing on bash removes the repeated PowerShell-syntax error class
(`@'...'@` here-string leaks, `2>&1` ErrorRecord wrapping, native-exe exit-code
inversion, ternary/`??` parser errors) — it is not a coin-flip.

---

## The rule

**Default to the Bash tool.** Use it for:

- git / gh
- python / python3 / uv / uvx / pytest / ruff / pip
- node / npm / npx
- file ops (cat, head, tail, ls, mkdir, mv, cp, rm, touch)
- search (grep, rg, sed, awk, find), jq, diff, curl, wget, make, tar

**Use PowerShell ONLY for genuinely Windows-only operations:**

- `wsl` bridge
- Windows process/service table: `Get-Process`, `Stop-Process`, `Get-Service`
- Windows filesystem/registry cmdlets: `Get-Item`, `Get-ChildItem`, `Test-Path`,
  `Join-Path`, `HKLM:`/`HKCU:` registry access
- PS-native search the operator runs on Windows paths: `Select-String`,
  `Measure-Object`, `ForEach-Object`, `Where-Object`
- scheduled tasks: `Register-ScheduledTask`, `schtasks`, `Get-ScheduledTask`
- GUI launchers / project batch files: `Start-Process`, `START_BOT`,
  `START_REMOTE`, `*.bat`, `stop_live`
- `*.ps1` infra scripts
- `powercfg` / PredatorSense (CPU-instability mitigation)
- `code-review-graph update` invoked on Windows paths

---

## Enforcement

`.claude/hooks/shell-canon-guard.py` (PreToolUse / **PowerShell** matcher):

- The guard targets the *PowerShell* tool, not Bash — the error class is Claude
  *choosing* PowerShell when bash would do. Intercept at that choice.
- **Allowlist match** (the Windows-only list above) → pass silently.
- **Bash-equivalent leading verb** (git/python/file/search) → soft steer
  (exit 2 + stderr) asking Claude to re-issue via the Bash tool.
- **Unknown shape** → fail-open (pass). The guard only steers the clear cases.

Fail-open by design: any parse error, missing field, empty command, or wrong
matcher exits 0. The guard never blocks a session it cannot read — mirrors
`branch-flip-guard.py` / `data-first-guard.py`.

Soft-block, not hard wall: the steer removes the reflexive ambiguity; the
operator can still force PowerShell by re-running.

---

## Settings

`.claude/settings.json` carries `"defaultShell": "bash"`. The `!`-prefixed
in-session command path and the `statusLine` both run bash.

---

## Related

- `.claude/hooks/shell-canon-guard.py` — this rule's enforcement
- `memory/feedback_powershell_heredoc_at_leak_git_commit_n1_2026_05_25.md` — a
  concrete instance of the error class this canon prevents
- `CLAUDE.md` § Shell — the one-line policy pointer
