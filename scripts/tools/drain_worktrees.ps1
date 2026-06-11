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
    DIVERGED  behind > 0 (origin/main moved past it). SUB-SPLIT for the audit/reap surface:
                REDUNDANT  git cherry sees 0 unique '+' commits (all unique work
                           reached main via squash/rebase, by patch-id)  -> delete CANDIDATE
                STRANDED   git cherry sees >0 unique '+' commits (carries genuinely
                           absent work)                                   -> keep/toss decision
    MERGED    0 ahead (already on main)                                -> nothing to do

  IMPORTANT -- cherry is DISPLAY-ONLY, never a delete authority. git cherry compares by
  patch-id; merge commits HAVE NO patch-id, so cherry mis-labels a branch carrying a merge
  commit (e.g. wt-codex-pr363-merge) as REDUNDANT when its content is genuinely NOT on main.
  The actual delete (-ReapRedundant) is gated by `git merge-base --is-ancestor <branch>
  origin/main` -- the branch tip must be a genuine ancestor of origin/main -- THEN `git branch
  -d` as a second belt (never `git branch -D`/force). NOTE the trap this guards: `git branch
  -d` ALONE is NOT a "fully on origin/main" gate -- its merged-check is merged-to-UPSTREAM-or-
  HEAD, so a branch pushed to its own origin/<name> ref (or in a squash-merge repo, whose patch
  reached main but whose tip is not an ancestor) passes `-d` and gets deleted even though it is
  nowhere near origin/main. The explicit is-ancestor gate is what makes the delete safe. In a
  squash/rebase-merge repo this typically deletes ZERO (cherry-redundant != ancestor-of-main) --
  the correct conservative outcome. Deletion is reflog-recoverable regardless.

  Safety properties inherited from the proven reaper:
    - FAIL-SAFE / IDEMPOTENT: any error or guard miss -> log + continue/skip, never crash.
    - PEER-LEASE AWARE: if a live peer holds main, skip the push (no concurrent writer).
    - NEVER force-pushes, resets, rebases, or resolves conflicts.
    - FETCH IMMEDIATELY BEFORE PUSH; a stale-ref reject -> skip that branch, never --force.
    - CAPITAL paths are NEVER auto-merged -- always surfaced for human review.

.PARAMETER Execute
  Actually push the DRAIN-classified (clean, non-capital) branches. Without this
  flag the script is read-only (dry-run): it prints the plan and pushes nothing.

.PARAMETER Audit
  Read-only decision surface for the stranded-branch pile. Prints, per STRANDED
  branch, ONE line `[<uniq> uniq, <nfiles> files, <age>] <branch> :: <subject>`
  (sorted stalest-last) so the keep/toss call is token-efficient, plus a
  "SAFE TO DELETE (candidates)" header listing the REDUNDANT branches. Mutates
  nothing and does NOT fetch -- works even while a live peer holds main's lease.

.PARAMETER ReapRedundant
  Opt-in MUTATE. After a fresh fetch, delete each REDUNDANT-display branch ONLY if
  (a) it is not checked out in a worktree AND (b) `git merge-base --is-ancestor
  <branch> origin/main` certifies its tip is genuinely on origin/main; then
  `git branch -d` as a second belt (never `-D`/force). Branches failing either gate
  (incl. the squash/rebase case where the patch reached main but the tip is not an
  ancestor) are logged, not deleted. NEVER pushes. Reflog-recoverable. Default OFF.

.PARAMETER Repo
  Main repo path. Default C:\Users\joshd\canompx3.

.EXAMPLE
  pwsh scripts/tools/drain_worktrees.ps1                 # dry-run report (default; now splits REDUNDANT/STRANDED)
  pwsh scripts/tools/drain_worktrees.ps1 -Audit          # per-branch keep/toss decision surface (read-only)
  pwsh scripts/tools/drain_worktrees.ps1 -ReapRedundant  # git branch -d the git-certified-merged branches
  pwsh scripts/tools/drain_worktrees.ps1 -Execute        # push clean non-capital FFs
#>

[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$Audit,
    [switch]$ReapRedundant,
    [string]$Repo = 'C:\Users\joshd\canompx3'
)

# Native git is driven by exit code / output, NOT by exceptions. Under 'Stop',
# PowerShell 5.1 wraps benign git stderr (e.g. "multiple merge bases") as a
# terminating ErrorRecord even when git exits 0 (shell-canon.md footgun). 'Continue'
# is the correct posture for native-exe orchestration; we check $LASTEXITCODE and
# the try/catch still backstops real PS errors.
$ErrorActionPreference = 'Continue'

# Mirrors .claude/hooks/judgment-review-nudge.py _CAPITAL_PATH_PREFIXES (the capital-path
# canonical list). NOTE (2026-06-11): there is NO `check_drain_worktrees_capital_prefix_parity`
# drift check -- verified against pipeline/check_drift.py. The real parity checks
# (check_drift.py:17354/17360) guard the PYTHON hooks (nudge<->soft-block/close-nudge) only,
# NOT this .ps1 mirror. So this list is NOT auto-parity-enforced; if you edit it, keep it in
# sync with the Python source by hand. (Adding a real parity check for this mirror is out of
# scope here -- flagged to operator.)
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

function Get-StrandedPlusCount([string]$branch) {
    # DISPLAY-ONLY classifier (never a delete authority). `git cherry origin/main <branch>`
    # marks each unique commit '+' (patch NOT on main) or '-' (patch IS on main, via
    # squash/rebase, compared by patch-id). Returns the count of '+' lines:
    #   0   -> REDUNDANT display (every unique commit's patch reached main)
    #   >0  -> STRANDED display  (carries genuinely-absent work)
    # CAVEAT: merge commits have NO patch-id, so cherry CANNOT see them -> a branch whose
    # only unique work is a merge commit counts 0 here yet is genuinely unmerged. The reap
    # path does NOT trust this count; `git merge-base --is-ancestor <branch> origin/main` is
    # the real safety gate (NOT `git branch -d` alone -- see the -ReapRedundant block for why).
    $cherry = (& git -C $Repo cherry origin/main $branch 2>$null)
    if (-not $cherry) { return 0 }
    return (@($cherry | Where-Object { $_ -like '+*' }).Count)
}

function Get-UniqueFileCount([string]$branch) {
    # Files this branch changed SINCE IT FORKED from main -- merge-base..branch, NOT the
    # 2-dot origin/main..branch (which on a diverged branch is a TIP-to-TIP diff that floods
    # in every file MAIN changed since the fork: e.g. radar-review reports 13773 vs the true
    # 10). rev-list --count uses 2-dot correctly (set-difference semantics) for COMMITS, but
    # `diff` 2-dot is tip-to-tip -- the asymmetry trap. Use merge-base for the file count.
    $mb = (& git -C $Repo merge-base origin/main $branch 2>$null)
    if (-not $mb) { return 0 }
    $files = (& git -C $Repo diff --name-only "$mb..$branch" 2>$null)
    if (-not $files) { return 0 }
    return (@($files).Count)
}

try {
    Write-Log "=== drain_worktrees run start (Execute=$Execute Audit=$Audit ReapRedundant=$ReapRedundant) ==="
    if (-not (Test-Path $Repo)) { Write-Log "[abort] repo missing: $Repo"; exit 0 }

    # -Audit is strictly read-only: it must work even while a live peer holds main's lease,
    # so it does NOT fetch (fetch updates refs/remotes/origin and the worktree-guard blocks it
    # as an index-mutating op). Every other mode fetches so it classifies vs a FRESH origin/main.
    if ($Audit) {
        Write-Log "[audit] read-only mode: skipping fetch (classifying vs the origin/main already on disk)."
    } else {
        & git -C $Repo fetch origin 2>$null | Out-Null
    }

    # Enumerate local branches (exclude detached HEADs).
    $branches = (& git -C $Repo for-each-ref --format='%(refname:short)' refs/heads 2>$null)
    if (-not $branches) { Write-Log "[done] no local branches"; exit 0 }

    # DIVERGED is now sub-split (display only) into REDUNDANT (cherry sees 0 unique '+')
    # and STRANDED (cherry sees >0). Each STRANDED/REDUNDANT entry is a PSCustomObject so
    # the -Audit emitter and -ReapRedundant loop can read the branch name + metrics without
    # re-parsing a formatted string.
    $plan = @{ DRAIN = @(); CAPITAL = @(); REDUNDANT = @(); STRANDED = @(); MERGED = @() }

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
            $plus = Get-StrandedPlusCount $b
            $rec  = [PSCustomObject]@{
                Branch  = $b
                Ahead   = $ahead
                Behind  = $behind
                Plus    = $plus
                Files   = (Get-UniqueFileCount $b)
                AgeRel  = (& git -C $Repo log -1 --format='%cr' $b 2>$null)
                AgeSecs = [int64]((& git -C $Repo log -1 --format='%ct' $b 2>$null) -as [int64])
                Subject = (& git -C $Repo log -1 --format='%s' $b 2>$null)
            }
            if ($plus -eq 0) { $plan.REDUNDANT += $rec } else { $plan.STRANDED += $rec }
            continue
        }
        $capHits = Get-CapitalHits $b
        if ($capHits.Count -gt 0) {
            $plan.CAPITAL += ("{0} (+{1}, capital: {2})" -f $b, $ahead, ($capHits -join ', '))
        } else {
            $plan.DRAIN += ("{0} (+{1}, clean FF)" -f $b, $ahead)
        }
    }

    # STRANDED sorted stalest-LAST (ascending age => most-recent first, oldest at the bottom
    # where it's the last thing the operator reads). REDUNDANT sorted the same way.
    $strandedSorted  = @($plan.STRANDED  | Sort-Object -Property AgeSecs -Descending)
    $redundantSorted = @($plan.REDUNDANT | Sort-Object -Property AgeSecs -Descending)

    Write-Log "--- PLAN ---"
    Write-Log ("DRAIN     (clean, non-capital, pushable): {0}" -f $plan.DRAIN.Count)
    foreach ($x in $plan.DRAIN)    { Write-Log "  [DRAIN]     $x" }
    Write-Log ("CAPITAL   (SKIP, manual review):          {0}" -f $plan.CAPITAL.Count)
    foreach ($x in $plan.CAPITAL)  { Write-Log "  [CAPITAL]   $x" }
    Write-Log ("REDUNDANT (cherry sees 0 unique; -ReapRedundant delete CANDIDATE): {0}" -f $redundantSorted.Count)
    foreach ($r in $redundantSorted) {
        Write-Log ("  [REDUNDANT] {0} (+{1}/-{2}, {3})" -f $r.Branch, $r.Ahead, $r.Behind, $r.AgeRel)
    }
    Write-Log ("STRANDED  (carries unique work; keep/toss -- run -Audit): {0}" -f $strandedSorted.Count)
    foreach ($s in $strandedSorted) {
        Write-Log ("  [STRANDED]  {0} (+{1} uniq, {2} files, {3})" -f $s.Branch, $s.Plus, $s.Files, $s.AgeRel)
    }
    Write-Log ("MERGED    (already on main):              {0}" -f $plan.MERGED.Count)

    # Stash + stale-worktree awareness (the auto-catalogue the 2026-06-10 drain lacked).
    $stashN = (& git -C $Repo stash list 2>$null | Measure-Object).Count
    $capStash = (& git -C $Repo stash list 2>$null | Select-String -Pattern '2305|2194|live|dashboard|start_bot|account|repoint' | Measure-Object).Count
    Write-Log "--- STASHES: $stashN total, ~$capStash capital/live-suspect (inspect before dropping) ---"
    $wtN = (& git -C $Repo worktree list 2>$null | Measure-Object).Count
    Write-Log "--- WORKTREES: $wtN total ---"

    # --- Audit: read-only keep/toss decision surface for the stranded pile ---
    # Mutates nothing, does NOT fetch -> safe under a live-peer lease. Mutually exclusive with
    # the push path: -Audit always exits read-only regardless of -Execute.
    if ($Audit) {
        Write-Log "--- AUDIT: SAFE TO DELETE (candidates -- cherry sees 0 unique '+'; real delete gated by is-ancestor-of-origin/main) ---"
        if ($redundantSorted.Count -eq 0) { Write-Log "  (none)" }
        foreach ($r in $redundantSorted) {
            # Patch-equivalent SHA list = the unique commits cherry found already on main.
            $eqShas = (& git -C $Repo rev-list "origin/main..$($r.Branch)" 2>$null) -join ' '
            Write-Log ("  [{0}] {1} :: {2}" -f $r.AgeRel, $r.Branch, $r.Subject)
            if ($eqShas) { Write-Log ("      patch-equiv unique SHAs: {0}" -f $eqShas) }
        }
        Write-Log "--- AUDIT: STRANDED (keep the idea OR throw away -- one line each, stalest last) ---"
        if ($strandedSorted.Count -eq 0) { Write-Log "  (none)" }
        foreach ($s in $strandedSorted) {
            Write-Log ("  [{0} uniq, {1} files, {2}] {3} :: {4}" -f $s.Plus, $s.Files, $s.AgeRel, $s.Branch, $s.Subject)
        }
        Write-Log "[audit] read-only; nothing fetched, pushed, or deleted."
        Write-Log "=== run end (audit) ==="
        exit 0
    }

    # --- ReapRedundant: delete ONLY branches git-certified fully on origin/main ---
    # SAFETY MODEL (corrected 2026-06-11 after a live verification deleted a branch that
    # `git branch -d` wrongly cleared): the gate is `git merge-base --is-ancestor <branch>
    # origin/main`, NOT `git branch -d` alone. `-d`'s "merged" check is merged-to-UPSTREAM-or-
    # HEAD -- a branch pushed to its own origin/<name> tracking ref passes `-d` even though it
    # is nowhere near origin/main (this repo squash/rebase-merges, so a squashed branch's tip
    # is NEVER an ancestor of origin/main yet IS merged to its own upstream). So we:
    #   1. SKIP any branch checked out in a worktree (report which one).
    #   2. REQUIRE `git merge-base --is-ancestor <branch> origin/main` == 0 (genuinely on main).
    #      Any branch failing this is left for manual review -- this is what catches the
    #      merge-commit/squash false-positives cherry can't see.
    #   3. Only then `git branch -d` (second belt; never -D/force). Reflog-recoverable.
    # In a squash-merge repo this typically deletes ZERO this run -- the correct conservative
    # outcome (provably zero work lost), since cherry-redundant != ancestor-of-main.
    if ($ReapRedundant) {
        if ($redundantSorted.Count -eq 0) { Write-Log "[reap] no REDUNDANT-display branches"; Write-Log "=== run end (reap) ==="; exit 0 }
        # Fresh fetch so origin/main is current before the ancestry gate evaluates.
        & git -C $Repo fetch origin 2>$null | Out-Null
        # Branch names checked out in a worktree (git refuses to delete these; we report which).
        $wtBranchMap = @{}
        $wtLines = (& git -C $Repo worktree list --porcelain 2>$null)
        $curWt = $null
        foreach ($ln in $wtLines) {
            if ($ln -like 'worktree *')   { $curWt = $ln.Substring(9) }
            elseif ($ln -like 'branch *') {
                $bn = $ln.Substring(7) -replace '^refs/heads/', ''
                $wtBranchMap[$bn] = $curWt
            }
        }
        $deleted = 0
        foreach ($r in $redundantSorted) {
            $b = $r.Branch
            if ($wtBranchMap.ContainsKey($b)) {
                Write-Log ("[skip] {0} checked out in {1} -- reap after worktree removed." -f $b, $wtBranchMap[$b])
                continue
            }
            # The REAL gate: is this branch's tip genuinely an ancestor of origin/main?
            & git -C $Repo merge-base --is-ancestor $b 'origin/main' 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Log ("[skip] {0} NOT an ancestor of origin/main (cherry-redundant via squash/rebase, but tip not on main) -- left for manual review." -f $b)
                continue
            }
            $head = (& git -C $Repo rev-parse $b 2>$null)
            $out  = & git -C $Repo branch -d $b 2>&1   # second belt; never -D
            if ($LASTEXITCODE -eq 0) {
                $deleted++
                Write-Log ("[ok] deleted {0} ({1}) -- recover: git branch {0} {1}" -f $b, $head)
            } else {
                Write-Log ("[skip] {0} git branch -d refused despite ancestry pass -- left for manual review. ({1})" -f $b, ($out -join ' '))
            }
        }
        Write-Log ("[reap] deleted {0} branch(es); the rest are line-itemed above with their reason." -f $deleted)
        Write-Log "=== run end (reap) ==="
        exit 0
    }

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
