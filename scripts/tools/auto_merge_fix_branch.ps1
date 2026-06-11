<#
.SYNOPSIS
  Guarded unattended merge of a verified fix branch into origin/main.

.DESCRIPTION
  Intended for automation-created bug-fix branches. The script is idempotent:
  it logs and exits 0 when guards are not satisfied so a scheduled task can
  retry later. It never rebases, force-pushes, resolves conflicts, or edits the
  dirty main worktree.

  Required guards:
    1. Worktree exists, is clean, and is on the requested branch.
    2. Fresh fetch succeeds.
    3. HEAD is a clean fast-forward of origin/main.
    4. Push to origin/main succeeds without force.

  Optional cleanup:
    - unregister the scheduled task that invoked it
    - delete the remote/local fix branch
    - remove the fix worktree
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Branch,

    [Parameter(Mandatory = $true)]
    [string]$Worktree,

    [string]$MainRepo = 'C:\Users\joshd\canompx3',

    [string]$TaskName = '',

    [string]$LogFile = '',

    [switch]$PushBranch,

    [switch]$DeleteBranchOnSuccess,

    [switch]$RemoveWorktreeOnSuccess,

    [string[]]$AllowedDirtyPath = @('HANDOFF.md'),

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'Never'

if (-not $LogFile) {
    $LogFile = Join-Path $MainRepo 'docs\runtime\auto_merge_fix_branch.log'
}

function Write-Log([string]$Message) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    try {
        $dir = Split-Path -Parent $LogFile
        if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Add-Content -Path $LogFile -Value $line -Encoding utf8
    } catch {}
    Write-Output $line
}

function Invoke-Git([string]$Repo, [string[]]$GitArgs, [switch]$AllowFailure) {
    $output = & git -C $Repo @GitArgs 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0 -and -not $AllowFailure) {
        throw "git -C $Repo $($GitArgs -join ' ') failed ($code): $output"
    }
    return [pscustomobject]@{ Code = $code; Output = (($output | Out-String).Trim()) }
}

function Unregister-Self {
    if (-not $TaskName) { return }
    try {
        if (-not $DryRun) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        }
        Write-Log "[done] unregistered scheduled task $TaskName"
    } catch {
        Write-Log "[warn] could not unregister task ${TaskName}: $($_.Exception.Message)"
    }
}

try {
    Write-Log "=== run start branch=$Branch worktree=$Worktree dry_run=$DryRun ==="

    if (-not (Test-Path $MainRepo)) { Write-Log "[skip] main repo missing: $MainRepo"; exit 0 }
    if (-not (Test-Path $Worktree)) { Write-Log "[skip] fix worktree missing: $Worktree"; exit 0 }

    $currentBranch = (Invoke-Git $Worktree @('branch', '--show-current')).Output
    if ($currentBranch -ne $Branch) {
        Write-Log "[skip] worktree is on '$currentBranch', expected '$Branch'"
        exit 0
    }

    $dirtyLines = @()
    $dirtyText = (Invoke-Git $Worktree @('status', '--porcelain')).Output
    if ($dirtyText) {
        $dirtyLines = $dirtyText -split "`r?`n" | Where-Object { $_ }
    }
    $blockingDirty = @()
    foreach ($line in $dirtyLines) {
        if ($line.Length -ge 4 -and $line[2] -eq ' ') {
            $path = ($line.Substring(3)).Trim()
        } elseif ($line.Length -ge 3) {
            $path = ($line.Substring(2)).Trim()
        } else {
            $path = $line.Trim()
        }
        if ($path -like '* -> *') { $path = ($path -split ' -> ')[-1].Trim() }
        $normalized = $path -replace '\\', '/'
        if ($AllowedDirtyPath -notcontains $normalized) {
            $blockingDirty += $line
        }
    }
    if ($blockingDirty.Count -gt 0) {
        Write-Log "[skip] fix worktree has dirty non-allowed paths; commit the verified fix first: $($blockingDirty -join '; ')"
        exit 0
    }
    if ($dirtyLines.Count -gt 0) {
        Write-Log "[info] ignoring allowed dirty paths while merging committed fix: $($dirtyLines -join '; ')"
    }

    Invoke-Git $Worktree @('fetch', 'origin') | Out-Null

    $head = (Invoke-Git $Worktree @('rev-parse', 'HEAD')).Output
    $originMain = (Invoke-Git $Worktree @('rev-parse', 'origin/main')).Output
    $mergeBase = (Invoke-Git $Worktree @('merge-base', 'HEAD', 'origin/main')).Output
    $behind = (Invoke-Git $Worktree @('rev-list', '--count', 'HEAD..origin/main')).Output
    $ahead = (Invoke-Git $Worktree @('rev-list', '--count', 'origin/main..HEAD')).Output

    if ($ahead -eq '0') {
        Write-Log "[done] branch has 0 commits ahead of origin/main; treating as already merged"
        Unregister-Self
        exit 0
    }

    if ($behind -ne '0' -or $mergeBase -ne $originMain) {
        Write-Log "[skip] branch is not a clean fast-forward of origin/main (ahead=$ahead behind=$behind); manual merge required"
        exit 0
    }

    if ($PushBranch) {
        Write-Log "[go] pushing backup branch origin/$Branch"
        if (-not $DryRun) {
            $pushBranchResult = Invoke-Git $Worktree @('push', '-u', 'origin', ('HEAD:' + $Branch)) -AllowFailure
            if ($pushBranchResult.Code -ne 0) {
                Write-Log "[skip] branch push failed: $($pushBranchResult.Output)"
                exit 0
            }
        }
    }

    Write-Log "[go] clean fast-forward: pushing $head to origin/main (ahead=$ahead behind=0)"
    if (-not $DryRun) {
        $pushMainResult = Invoke-Git $Worktree @('push', 'origin', 'HEAD:main') -AllowFailure
        if ($pushMainResult.Code -ne 0) {
            Write-Log "[skip] main push rejected or failed: $($pushMainResult.Output)"
            exit 0
        }
    }
    Write-Log "[ok] origin/main is now $head"

    if ($DeleteBranchOnSuccess) {
        if (-not $DryRun) {
            Invoke-Git $Worktree @('push', 'origin', '--delete', $Branch) -AllowFailure | Out-Null
            Invoke-Git $MainRepo @('branch', '-D', $Branch) -AllowFailure | Out-Null
        }
        Write-Log "[done] requested branch cleanup for $Branch"
    }

    if ($RemoveWorktreeOnSuccess) {
        if (-not $DryRun) {
            Invoke-Git $MainRepo @('worktree', 'remove', $Worktree, '--force') -AllowFailure | Out-Null
        }
        Write-Log "[done] requested worktree cleanup for $Worktree"
    }

    Unregister-Self
    Write-Log "=== MERGE COMPLETE ==="
    exit 0
} catch {
    Write-Log "[error] $($_.Exception.Message)"
    exit 0
}
