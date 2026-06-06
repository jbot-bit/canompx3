<#
.SYNOPSIS
  Daily local git hygiene for canompx3 — keep the working repo from accumulating
  stale branches and dead worktrees while merged work lands via GitHub auto-merge.

.DESCRIPTION
  Built 2026-06-03. The project's "stale old shit" problem has two halves:
    - REMOTE pile-up: fixed structurally by repo setting delete_branch_on_merge=true
      (GitHub deletes the branch the moment a PR merges). This script does NOT touch
      that — it is already handled server-side.
    - LOCAL pile-up: branches whose upstream is [gone] (merged + deleted on remote)
      linger locally; abandoned worktrees stay registered. This script prunes those.

  This is the same primitive as the `commit-commands:clean_gone` plugin and the
  standard `git fetch --prune` + delete-gone-branches idiom — not a reinvention,
  just scheduled and made fail-safe for unattended runs.

  SAFETY (fail-closed, never destructive on shared state):
    - SKIPS entirely if a live peer Claude session holds the main worktree lease
      (heartbeat authoritative). No git mutation while a peer may be writing.
    - NEVER deletes the current branch, main, or any branch that is NOT [gone]
      (i.e. only branches whose tracked upstream was deleted on origin).
    - NEVER force-deletes unmerged work that still has a live upstream.
    - Worktree prune only removes worktrees git itself reports as prunable
      (missing directory) via `git worktree prune` — it does not remove live trees.
  All actions are logged. Any error -> log + exit 0 (retry next day).
#>

$ErrorActionPreference = 'Stop'

$Repo    = 'C:\Users\joshd\canompx3'
$LogFile = Join-Path $Repo 'docs\runtime\git_hygiene.log'

function Write-Log([string]$m) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    try { Add-Content -Path $LogFile -Value $line -Encoding utf8 } catch {}
}

try {
    Write-Log "=== git hygiene run start ==="
    if (-not (Test-Path $Repo)) { Write-Log "[skip] repo missing"; exit 0 }

    # --- Guard: never run while a live peer holds the lease ---
    $guard = & python "$Repo\scripts\tools\worktree_guard.py" --status --json 2>$null
    if ($LASTEXITCODE -eq 0 -and $guard) {
        try {
            $g = $guard | ConvertFrom-Json
            if ($g.peer_live -eq $true -or $g.fresh_peer_heartbeat -eq $true) {
                Write-Log ("[skip] peer lease LIVE (hb_age={0}s) -- retry next run" -f $g.heartbeat_age_seconds)
                exit 0
            }
        } catch { Write-Log "[skip] guard parse failed -- conservative skip"; exit 0 }
    } else {
        Write-Log "[skip] guard unavailable -- conservative skip"; exit 0
    }

    # --- 1. Prune remote-tracking refs that no longer exist on origin ---
    & git -C $Repo fetch --prune origin 2>$null | Out-Null
    Write-Log "[ok] fetch --prune origin"

    # --- 2. Prune dead worktrees (git only removes trees whose dir is gone) ---
    $wtBefore = (& git -C $Repo worktree list).Count
    & git -C $Repo worktree prune -v 2>&1 | ForEach-Object { Write-Log "[worktree] $_" }
    $wtAfter = (& git -C $Repo worktree list).Count
    Write-Log "[ok] worktree prune ($wtBefore -> $wtAfter)"

    # --- 3. Delete local branches whose upstream is [gone] (merged + remote-deleted) ---
    # Identify via `git branch -vv` lines containing ': gone]'. Never touch the
    # current branch (marked '*') or main/master.
    $current = (& git -C $Repo rev-parse --abbrev-ref HEAD).Trim()
    $gone = @()
    foreach ($line in (& git -C $Repo branch -vv)) {
        if ($line -match '^\*') { continue }                       # current branch
        if ($line -notmatch ':\s*gone\]') { continue }             # upstream still exists
        # branch name is the first token after optional leading spaces
        $name = ($line.Trim() -split '\s+')[0]
        if ($name -in @('main','master',$current)) { continue }
        $gone += $name
    }

    if ($gone.Count -eq 0) {
        Write-Log "[ok] no [gone] local branches to delete"
    } else {
        foreach ($b in $gone) {
            # -d (safe delete) refuses if not merged; fall back to -D ONLY for
            # branches whose upstream was [gone] (work already landed on origin).
            $out = & git -C $Repo branch -d $b 2>&1
            if ($LASTEXITCODE -ne 0) {
                $out = & git -C $Repo branch -D $b 2>&1
            }
            Write-Log "[deleted] local branch $b"
        }
        Write-Log "[ok] deleted $($gone.Count) [gone] local branch(es)"
    }

    Write-Log "=== git hygiene run done ==="
    exit 0
}
catch {
    Write-Log "[error] $($_.Exception.Message) -- exit 0, retry next run"
    exit 0
}
