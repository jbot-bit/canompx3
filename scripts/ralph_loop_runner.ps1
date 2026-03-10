# Ralph Loop Runner — PowerShell version
#
# Usage:
#   .\scripts\ralph_loop_runner.ps1                    # Single iteration, Sonnet
#   .\scripts\ralph_loop_runner.ps1 -Mode loop         # Continuous loop
#   .\scripts\ralph_loop_runner.ps1 -Mode audit-only   # Audit only, no fixes
#   .\scripts\ralph_loop_runner.ps1 -Model opus        # Use Opus instead
#
# Stop a running loop: New-Item ralph_loop.stop

param(
    [ValidateSet("loop", "once", "audit-only")]
    [string]$Mode = "once",

    [ValidateSet("sonnet", "opus")]
    [string]$Model = "sonnet"
)

$ErrorActionPreference = "Stop"

# Repo root = parent of scripts/
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$StopFile    = Join-Path $RepoRoot "ralph_loop.stop"
$HistoryFile = Join-Path $RepoRoot "docs\ralph-loop\ralph-loop-history.md"
$LogDir      = Join-Path $RepoRoot "docs\ralph-loop\logs"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

function Get-Iteration {
    if (Test-Path $HistoryFile) {
        # Use regex (not -SimpleMatch) so ^ anchor works
        $matches = Select-String -Path $HistoryFile -Pattern "^## Iteration \d+"
        $count = ($matches | Measure-Object).Count
        return $count + 1
    }
    return 1
}

function Invoke-Claude($phase, $iter, $promptText) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $log = Join-Path $LogDir "iteration-${iter}-${phase}.log"

    Write-Host "[$ts] Ralph Loop - Iteration $iter - $($phase.ToUpper()) PHASE" -ForegroundColor Cyan
    Write-Host "  Spawning claude --print --model $Model ... (this takes 1-3 min, output streams below)" -ForegroundColor DarkGray

    # Write prompt to temp file to avoid PowerShell argument quoting issues
    $tmpPrompt = Join-Path $env:TEMP "ralph_prompt_${phase}.txt"
    $promptText | Out-File -FilePath $tmpPrompt -Encoding utf8

    # Clear nesting detection env vars so claude --print can spawn
    $savedClaudeCode = $env:CLAUDECODE
    $savedEntrypoint = $env:CLAUDE_CODE_ENTRYPOINT
    $env:CLAUDECODE = $null
    $env:CLAUDE_CODE_ENTRYPOINT = $null

    # Pipe prompt via stdin to avoid multiline arg issues
    Get-Content $tmpPrompt | claude --print --model $Model 2>&1 | Tee-Object -FilePath $log

    # Restore env vars
    $env:CLAUDECODE = $savedClaudeCode
    $env:CLAUDE_CODE_ENTRYPOINT = $savedEntrypoint

    Remove-Item $tmpPrompt -Force -ErrorAction SilentlyContinue

    $ts2 = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts2] $phase phase complete. Log: $log" -ForegroundColor Green
}

function Run-Audit($iter) {
    $prompt = @"
You are the Ralph Loop Auditor (read .claude/agents/ralph-auditor.md for your full instructions).

This is iteration $iter of the Ralph Loop.

Read the previous audit state from docs/ralph-loop/ralph-loop-audit.md
Read the history from docs/ralph-loop/ralph-loop-history.md

Then run a full audit:
1. Run infrastructure gates: drift check, behavioral audit, test suite, lint
2. Scan trading_app/live/ for the Seven Sins (silent failures, fail-open, phantom state, etc.)
3. Check canonical integrity (hardcoded lists, magic numbers, dependency direction)
4. Check test coverage for recently changed files

Write your structured findings to docs/ralph-loop/ralph-loop-audit.md using the format from your agent prompt.

IMPORTANT: Do NOT write any production code. Only update the audit file.
"@
    Invoke-Claude "audit" $iter $prompt
}

function Run-Implement($iter) {
    $prompt = @"
You are the Ralph Loop Architect (read .claude/agents/ralph-architect.md for your full instructions).

This is iteration $iter of the Ralph Loop.

1. Read the current audit from docs/ralph-loop/ralph-loop-audit.md
2. Select the highest-priority finding that is safe to fix autonomously
3. Write the plan to docs/ralph-loop/ralph-loop-plan.md
4. Then become the Implementer (read .claude/agents/ralph-implementer.md):
   - Follow the 2-pass method: Discovery first, then Implementation
   - Apply the minimal fix
   - Run tests and drift check
5. Report what was done

SAFETY: If the top finding requires schema changes, entry model changes,
or touches 5+ files - SKIP implementation and flag for human review.
Update the plan file with your decision.

Do NOT commit. The Verifier handles that gate.
"@
    Invoke-Claude "implement" $iter $prompt
}

function Run-Verify($iter) {
    $prompt = @"
You are the Ralph Loop Verifier (read .claude/agents/ralph-verifier.md for your full instructions).

This is iteration $iter of the Ralph Loop.

1. Read the plan from docs/ralph-loop/ralph-loop-plan.md
2. Read the audit from docs/ralph-loop/ralph-loop-audit.md
3. Run ALL 6 verification gates:
   - Gate 1: python pipeline/check_drift.py
   - Gate 2: python scripts/tools/audit_behavioral.py
   - Gate 3: python -m pytest tests/ -x -q
   - Gate 4: ruff check pipeline/ trading_app/ scripts/
   - Gate 5: Blast radius verification (read callers of changed functions)
   - Gate 6: Regression scan (verify the specific fix)
4. Write your verdict to the plan file
5. If ACCEPT: append the full iteration record to docs/ralph-loop/ralph-loop-history.md
6. If REJECT: document why in the plan file and flag for next iteration

Use the structured output format from your agent prompt.
"@
    Invoke-Claude "verify" $iter $prompt
}

# === Main ===
Write-Host ""
Write-Host "=== Ralph Loop Starting ===" -ForegroundColor White
Write-Host "  Mode:  $Mode" -ForegroundColor White
Write-Host "  Model: $Model" -ForegroundColor White
Write-Host "  Stop:  New-Item ralph_loop.stop" -ForegroundColor DarkGray
Write-Host ""

while ($true) {
    if (Test-Path $StopFile) {
        Write-Host "Stop file detected - shutting down." -ForegroundColor Red
        Remove-Item $StopFile -Force
        exit 0
    }

    $iter = Get-Iteration
    Write-Host "==========================================" -ForegroundColor White
    Write-Host "  Ralph Loop - Iteration $iter" -ForegroundColor White
    Write-Host "==========================================" -ForegroundColor White

    Run-Audit $iter

    if (Test-Path $StopFile) { Remove-Item $StopFile -Force; exit 0 }

    if ($Mode -ne "audit-only") {
        Run-Implement $iter

        if (Test-Path $StopFile) { Remove-Item $StopFile -Force; exit 0 }

        Run-Verify $iter
    }

    Write-Host "`nIteration $iter complete.`n" -ForegroundColor Green

    if ($Mode -eq "once" -or $Mode -eq "audit-only") {
        Write-Host "Done." -ForegroundColor White
        exit 0
    }

    Write-Host "Sleeping 10s... (create ralph_loop.stop to exit)" -ForegroundColor Gray
    Start-Sleep -Seconds 10
}
