<#
.SYNOPSIS
  Report (and optionally drain) local branches that are ahead of origin/main but
  never pushed home. Generalizes auto_merge_reaper_prefix_guard.ps1 (which was
  hardcoded to ONE branch) to EVERY local branch -- but DRY-RUN BY DEFAULT and
  NEVER unattended (operator decision 2026-06-10: "dry-run only, never auto-push").

.DESCRIPTION
  The recurrence fix for stranded-worktree drift: finished commits get marooned on
  feature branches because nothing forces them home. This scans all local branches,
  classifies each, and shows what WOULD drain. It pushes nothing unless you pass
  -Execute, and even then only CLEAN, NON-CAPITAL fast-forwards. It also catalogues
  stashes and flags stale worktrees each run (the blind spot that bit the 2026-06-10
  drain: uncatalogued capital stashes).

  Per-branch classification (vs a FRESH origin/main):
    DRAIN     clean fast-forward (0 behind), touches NO capital path  -> pushable
    CAPITAL   touches a capital path (trading_app/live, pipeline/, etc) -> SKIP, manual
    DIVERGED  behind > 0 (origin/main moved past it)                   -> SKIP, manual
    MERGED    0 ahead (already on main)                                -> nothing to do

  Safety properties inherited from the proven reaper:
    - FAIL-SAFE / IDEMPOTENT: any error or guard miss -> log + continue/skip, never crash.
    - PEER-LEASE AWARE: if a live peer holds main, skip the push (no concurrent writer).
    - NEVER force-pushes, resets, rebases, or resolves conflicts.
    - FETCH IMMEDIATELY BEFORE PUSH; a stale-ref reject -> skip that branch, never --force.
    - CAPITAL paths are NEVER auto-merged -- always surfaced for human review.

.PARAMETER Execute
  Actually push the DRAIN-classified (clean, non-capital) branches. Without this
  flag the script is read-only (dry-run): it prints the plan and pushes nothing.

.PARAMETER Repo
  Main repo path. Default C:\Users\joshd\canompx3.

.EXAMPLE
  pwsh scripts/tools/drain_worktrees.ps1            # dry-run report (default)
  pwsh scripts/tools/drain_worktrees.ps1 -Execute   # push clean non-capital FFs
#>

[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Repo = 'C:\Users\joshd\canompx3'
)

# Native git is driven by exit code / output, NOT by exceptions. Under 'Stop',
# PowerShell 5.1 wraps benign git stderr (e.g. "multiple merge bases") as a
# terminating ErrorRecord even when git exits 0 (shell-canon.md footgun). 'Continue'
# is the correct posture for native-exe orchestration; we check $LASTEXITCODE and
# the try/catch still backstops real PS errors.
$ErrorActionPreference = 'Continue'

# @canonical-source: .claude/hooks/judgment-review-nudge.py _CAPITAL_PATH_PREFIXES
# Kept in parity by check_drift (see check_drain_worktrees_capital_prefix_parity).
# If you edit this list, edit the Python source first -- that is canonical; this is
# the mirror. A drift check fails if the two diverge.
$CapitalPathPrefixes = @(
    'trading_app/live/',
    'trading_app/risk_manager.py',
    'trading_app/execution_engine.py',
    'trading_app/session_orchestrator.py',
    'pipeline/'
)

$LogFile = Join-Path $Repo 'docs\runtime\drain_worktrees.log'

function Write-Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Host $msg
    try { Add-Content -Path $LogFile -Value $line -Encoding utf8 } catch {}
}

function Test-PeerLeaseLive {
    # Reuse the canonical worktree_guard status. Returns $true if a live peer holds
    # main (conservative: any uncertainty -> treat as live so we never race a writer).
    try {
        $guard = & python "$Repo\scripts\tools\worktree_guard.py" --status --json 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $guard) { return $true }
        $g = $guard | ConvertFrom-Json
        return ($g.peer_live -eq $true -or $g.fresh_peer_heartbeat -eq $true)
    } catch {
        return $true  # fail-closed on liveness
    }
}

function Get-CapitalHits([string]$branch) {
    # Files this branch changes vs origin/main that fall under a capital prefix.
    # 2-dot (not 3-dot): 3-dot warns "multiple merge bases" on octopus histories.
    # We want "files this branch's tip differs from origin/main" -> 2-dot is correct
    # and warning-free.
    $files = (& git -C $Repo diff --name-only "origin/main..$branch" 2>$null)
    if (-not $files) { return @() }
    $hits = @()
    foreach ($f in $files) {
        $fn = $f.Replace('\', '/')
        foreach ($p in $CapitalPathPrefixes) {
            if ($fn.StartsWith($p)) { $hits += $fn; break }
        }
    }
    return $hits
}

try {
    Write-Log "=== drain_worktrees run start (Execute=$Execute) ==="
    if (-not (Test-Path $Repo)) { Write-Log "[abort] repo missing: $Repo"; exit 0 }

    & git -C $Repo fetch origin 2>$null | Out-Null

    # Enumerate local branches (exclude detached HEADs).
    $branches = (& git -C $Repo for-each-ref --format='%(refname:short)' refs/heads 2>$null)
    if (-not $branches) { Write-Log "[done] no local branches"; exit 0 }

    $plan = @{ DRAIN = @(); CAPITAL = @(); DIVERGED = @(); MERGED = @() }

    foreach ($b in $branches) {
        if ($b -eq 'main') { continue }
        $ahead  = [int](& git -C $Repo rev-list --count "origin/main..$b" 2>$null)
        if ($ahead -eq 0) { $plan.MERGED += $b; continue }
        $behind = [int](& git -C $Repo rev-list --count "$b..origin/main" 2>$null)
        # DIVERGED is checked FIRST: a branch behind main is skipped regardless of
        # what it touches (its capital-ness is moot — it can't FF-push). Only among
        # clean fast-forwards do we split CAPITAL (needs human review) vs DRAIN
        # (pushable). Otherwise a stale branch's diff-vs-main (which includes files
        # MAIN changed) floods CAPITAL with un-actionable noise.
        if ($behind -ne 0) {
            $plan.DIVERGED += ("{0} (+{1}/-{2})" -f $b, $ahead, $behind)
            continue
        }
        $capHits = Get-CapitalHits $b
        if ($capHits.Count -gt 0) {
            $plan.CAPITAL += ("{0} (+{1}, capital: {2})" -f $b, $ahead, ($capHits -join ', '))
        } else {
            $plan.DRAIN += ("{0} (+{1}, clean FF)" -f $b, $ahead)
        }
    }

    Write-Log "--- PLAN ---"
    Write-Log ("DRAIN    (clean, non-capital, pushable): {0}" -f $plan.DRAIN.Count)
    foreach ($x in $plan.DRAIN)    { Write-Log "  [DRAIN]    $x" }
    Write-Log ("CAPITAL  (SKIP, manual review):          {0}" -f $plan.CAPITAL.Count)
    foreach ($x in $plan.CAPITAL)  { Write-Log "  [CAPITAL]  $x" }
    Write-Log ("DIVERGED (SKIP, behind main):            {0}" -f $plan.DIVERGED.Count)
    foreach ($x in $plan.DIVERGED) { Write-Log "  [DIVERGED] $x" }
    Write-Log ("MERGED   (already on main):              {0}" -f $plan.MERGED.Count)

    # Stash + stale-worktree awareness (the auto-catalogue the 2026-06-10 drain lacked).
    $stashN = (& git -C $Repo stash list 2>$null | Measure-Object).Count
    $capStash = (& git -C $Repo stash list 2>$null | Select-String -Pattern '2305|2194|live|dashboard|start_bot|account|repoint' | Measure-Object).Count
    Write-Log "--- STASHES: $stashN total, ~$capStash capital/live-suspect (inspect before dropping) ---"
    $wtN = (& git -C $Repo worktree list 2>$null | Measure-Object).Count
    Write-Log "--- WORKTREES: $wtN total ---"

    if (-not $Execute) {
        Write-Log "[dry-run] -Execute not set; pushed nothing. Re-run with -Execute to drain the DRAIN set."
        Write-Log "=== run end (dry-run) ==="
        exit 0
    }

    # --- Execute: push ONLY the clean non-capital fast-forwards ---
    if ($plan.DRAIN.Count -eq 0) { Write-Log "[done] nothing to drain"; exit 0 }
    if (Test-PeerLeaseLive) {
        Write-Log "[skip-all] a live peer holds main (or liveness unknown) -- not pushing this run."
        exit 0
    }

    foreach ($entry in $plan.DRAIN) {
        $b = ($entry -split ' ')[0]
        # Fetch-immediately-before-push; re-confirm still a clean FF (peer may have moved main).
        & git -C $Repo fetch origin 2>$null | Out-Null
        $behind2 = [int](& git -C $Repo rev-list --count "$b..origin/main" 2>$null)
        if ($behind2 -ne 0) { Write-Log "[skip] $b diverged between plan and push (now -$behind2) -- manual."; continue }
        # Re-confirm capital classification didn't change under the new main.
        if ((Get-CapitalHits $b).Count -gt 0) { Write-Log "[skip] $b now touches a capital path under fresh main -- manual."; continue }
        $head = (& git -C $Repo rev-parse $b 2>$null)
        $push = & git -C $Repo push origin "${b}:main" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "[skip] $b push rejected (peer raced us): $push -- never --force, retry later."
        } else {
            Write-Log "[ok] drained $b -> origin/main ($head)"
        }
    }

    Write-Log "=== run end (execute) ==="
    exit 0
}
catch {
    Write-Log "[error] $($_.Exception.Message) -- exiting 0 (fail-safe)."
    exit 0
}
