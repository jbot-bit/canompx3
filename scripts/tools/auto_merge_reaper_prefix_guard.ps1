<#
.SYNOPSIS
  Guarded, unattended fast-forward merge of session/joshd-reaper-prefix-guard
  into origin/main. Built 2026-06-03 so the operator can reset their PC while a
  live peer Claude session finishes its own work on main.

.DESCRIPTION
  Fires on a 30-min schedule (see register_auto_merge_reaper_task.ps1). Each run
  is FAIL-SAFE and IDEMPOTENT: it acts ONLY when every guard passes, otherwise it
  logs the reason and exits 0 (so the schedule simply retries next tick).

  Guards (ALL must hold to push):
    1. Branch still exists on origin and locally (else: already merged -> self-clean).
    2. Peer worktree lease on main is NOT live (no concurrent index writer).
    3. After a fresh fetch, the branch is a CLEAN FAST-FORWARD of origin/main
       (0 commits behind). DIVERGED -> SKIP and leave the branch for manual merge
       (operator chose "skip, never auto-rebase unattended").

  On success: pushes the FF to origin/main, unregisters its own scheduled task,
  removes the isolated worktree, deletes the now-merged branch. Then it is done.

  Nothing here force-pushes, resets, or resolves conflicts. Worst case it does
  nothing and retries. All output is appended to the log file.
#>

$ErrorActionPreference = 'Stop'

$MainRepo   = 'C:\Users\joshd\canompx3'
$Worktree   = 'C:\Users\joshd\canompx3-reaper-prefix-guard'
$Branch     = 'session/joshd-reaper-prefix-guard'
$TaskName   = 'CanonMPX_AutoMerge_ReaperPrefixGuard'
$LogFile    = Join-Path $MainRepo 'docs\runtime\auto_merge_reaper_prefix_guard.log'

function Write-Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    try { Add-Content -Path $LogFile -Value $line -Encoding utf8 } catch {}
}

function Stop-AndDisableSelf {
    # Unregister the scheduled task so it never fires again after success.
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Log "[done] unregistered scheduled task $TaskName"
    } catch {
        Write-Log "[warn] could not unregister task: $($_.Exception.Message)"
    }
}

try {
    Write-Log "=== run start ==="

    # Git must be available; main repo must exist.
    if (-not (Test-Path $MainRepo)) { Write-Log "[skip] main repo missing: $MainRepo"; exit 0 }

    # --- Guard 1: does the branch still exist on origin? (already-merged self-clean) ---
    $remoteRef = (& git -C $MainRepo ls-remote --heads origin $Branch) 2>$null
    if (-not $remoteRef) {
        Write-Log "[done] branch $Branch absent on origin (already merged/deleted). Cleaning up."
        if (Test-Path $Worktree) {
            try { & git -C $MainRepo worktree remove $Worktree --force 2>$null; Write-Log "[done] removed worktree $Worktree" } catch {}
        }
        Stop-AndDisableSelf
        exit 0
    }

    # --- Guard 2: peer lease on main must NOT be live ---
    # Reuse the canonical worktree_guard status (heartbeat is authoritative, not PID).
    $statusJson = & git -C $MainRepo rev-parse --show-toplevel *> $null  # cheap git sanity
    $guard = & python "$MainRepo\scripts\tools\worktree_guard.py" --status --json 2>$null
    if ($LASTEXITCODE -eq 0 -and $guard) {
        try {
            $g = $guard | ConvertFrom-Json
            if ($g.peer_live -eq $true -or $g.fresh_peer_heartbeat -eq $true) {
                Write-Log ("[skip] peer lease LIVE on main (peer_live={0} fresh_hb={1} hb_age={2}s) -- retry next tick" -f $g.peer_live, $g.fresh_peer_heartbeat, $g.heartbeat_age_seconds)
                exit 0
            }
        } catch {
            Write-Log "[skip] could not parse worktree_guard status -- conservative skip"; exit 0
        }
    } else {
        Write-Log "[skip] worktree_guard status unavailable -- conservative skip (fail-closed on liveness)"; exit 0
    }

    # --- Guard 3: fresh fetch, must be a clean fast-forward (0 behind) ---
    & git -C $Worktree fetch origin 2>$null | Out-Null
    $behind = (& git -C $Worktree rev-list --count "HEAD..origin/main").Trim()
    $ahead  = (& git -C $Worktree rev-list --count "origin/main..HEAD").Trim()
    $mb = (& git -C $Worktree merge-base HEAD origin/main).Trim()
    $om = (& git -C $Worktree rev-parse origin/main).Trim()

    if ($ahead -eq '0') {
        Write-Log "[done] branch has 0 commits ahead of origin/main (already merged). Cleaning up."
        Stop-AndDisableSelf
        exit 0
    }
    if ($behind -ne '0' -or $mb -ne $om) {
        Write-Log "[skip] DIVERGED: behind=$behind ahead=$ahead (origin/main moved). Operator chose manual merge -- leaving branch intact."
        exit 0
    }

    # --- All guards passed: push the fast-forward to origin/main ---
    $myHead = (& git -C $Worktree rev-parse HEAD).Trim()
    Write-Log "[go] clean FF: pushing $myHead -> origin/main (ahead=$ahead behind=0)"
    $push = & git -C $Worktree push origin "HEAD:main" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Log "[skip] push rejected (peer raced us between fetch and push): $push -- retry next tick"
        exit 0
    }
    Write-Log "[ok] pushed. origin/main is now $myHead"

    # --- Success cleanup: remove worktree, delete merged branch local+remote ---
    try { & git -C $MainRepo worktree remove $Worktree --force 2>$null; Write-Log "[done] removed worktree" } catch { Write-Log "[warn] worktree remove failed: $($_.Exception.Message)" }
    try { & git -C $MainRepo branch -D $Branch 2>$null } catch {}
    try { & git -C $MainRepo push origin --delete $Branch 2>$null; Write-Log "[done] deleted merged branch $Branch" } catch {}

    Stop-AndDisableSelf
    Write-Log "=== MERGE COMPLETE ==="
    exit 0
}
catch {
    Write-Log "[error] $($_.Exception.Message) -- exiting 0, will retry next tick"
    exit 0
}
