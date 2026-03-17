param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("claude", "codex", "codex-search", "list", "close", "close-pick", "resume", "menu", "prune")]
    [string]$Mode,

    [string]$Task,
    [string]$Tool
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$managerPy = Join-Path $repoRoot "scripts\tools\worktree_manager.py"

function Convert-ToGitBashPath([string]$PathValue) {
    $full = (Resolve-Path $PathValue).Path
    $drive = $full.Substring(0, 1).ToLowerInvariant()
    $rest = $full.Substring(2) -replace "\\", "/"
    return "/$drive$rest"
}

function Convert-ToWslPath([string]$PathValue) {
    $full = (Resolve-Path $PathValue).Path
    $drive = $full.Substring(0, 1).ToLowerInvariant()
    $rest = $full.Substring(2) -replace "\\", "/"
    return "/mnt/$drive$rest"
}

function Escape-BashSingleQuoted([string]$Value) {
    return $Value -replace "'", "'\"'\"'"
}

function Get-GitBashPath() {
    $candidates = @(
        (Join-Path $env:ProgramFiles "Git\bin\bash.exe"),
        (Join-Path $env:ProgramW6432 "Git\bin\bash.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Git\bin\bash.exe")
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Git Bash not found. Install Git for Windows or adjust the launcher."
}

function Run-GitBash([string]$CommandText) {
    $bash = Get-GitBashPath
    & $bash -lc $CommandText
    exit $LASTEXITCODE
}

function Run-WSL([string]$CommandText) {
    & wsl.exe bash -lc $CommandText
    exit $LASTEXITCODE
}

function Run-Manager([string[]]$Arguments) {
    $result = Invoke-Manager $Arguments
    if ($result.Success) {
        if ($result.Output) {
            $result.Output | Write-Output
        }
        exit 0
    }
    throw $result.Error
}

function Invoke-Manager([string[]]$Arguments) {
    $candidates = New-Object System.Collections.Generic.List[object]
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $candidates.Add(@{ Command = $venvPython; Args = @($managerPy) + $Arguments; Label = ".venv" })
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates.Add(@{ Command = "py"; Args = @("-3", $managerPy) + $Arguments; Label = "py -3" })
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates.Add(@{ Command = "python"; Args = @($managerPy) + $Arguments; Label = "python" })
    }

    $lastError = $null
    foreach ($candidate in $candidates) {
        try {
            $output = & $candidate.Command @($candidate.Args) 2>&1
            if ($LASTEXITCODE -eq 0) {
                return @{ Success = $true; Output = $output }
            }
            $lastError = "Manager failed via $($candidate.Label): $($output -join [Environment]::NewLine)"
        } catch {
            $lastError = "Manager failed via $($candidate.Label): $($_.Exception.Message)"
        }
    }

    return @{ Success = $false; Error = ($lastError ?? "No usable Python launcher found for workstream manager.") }
}

function Get-ManagedWorkstreams() {
    $result = Invoke-Manager @("list", "--managed-only", "--json")
    if (-not $result.Success -or -not $result.Output) {
        Write-Host ""
        Write-Host ($result.Error ?? "Unable to load managed workstreams.") -ForegroundColor Yellow
        return @()
    }
    $json = ($result.Output -join [Environment]::NewLine).Trim()
    if (-not $json) {
        return @()
    }
    return @($json | ConvertFrom-Json)
}

function Get-ExistingPurpose([string]$ToolName, [string]$WorkstreamName) {
    $result = Invoke-Manager @("show", "--tool", $ToolName, "--name", $WorkstreamName)
    if (-not $result.Success -or -not $result.Output) {
        return $null
    }
    $json = ($result.Output -join [Environment]::NewLine).Trim()
    if (-not $json -or $json -eq "{}") {
        return $null
    }
    try {
        $meta = $json | ConvertFrom-Json
        return $meta.purpose
    } catch {
        return $null
    }
}

function Show-ManagedWorkstreams([object[]]$Workstreams) {
    if (-not $Workstreams -or $Workstreams.Count -eq 0) {
        Write-Host ""
        Write-Host "No active workstreams found." -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "Active workstreams" -ForegroundColor Cyan
    Write-Host "------------------"
    for ($i = 0; $i -lt $Workstreams.Count; $i++) {
        $workstream = $Workstreams[$i]
        $status = if ($workstream.dirty) { "dirty" } else { "clean" }
        $purpose = if ($workstream.purpose) { $workstream.purpose } else { "No purpose saved" }
        $opened = if ($workstream.last_opened_at) { $workstream.last_opened_at } else { $workstream.created_at }
        Write-Host ("[{0}] {1} | {2} | {3}" -f ($i + 1), $workstream.name, $workstream.tool, $status)
        Write-Host ("     Purpose: {0}" -f $purpose) -ForegroundColor DarkGray
        Write-Host ("     Last used: {0}" -f $opened) -ForegroundColor DarkGray
        Write-Host ("     Branch: {0}" -f $workstream.branch) -ForegroundColor DarkGray
    }
}

function Select-ManagedWorkstream() {
    $workstreams = Get-ManagedWorkstreams
    Show-ManagedWorkstreams $workstreams
    if (-not $workstreams -or $workstreams.Count -eq 0) {
        return $null
    }

    $choice = Read-Host "Pick a workstream number"
    if (-not $choice) {
        return $null
    }
    $index = 0
    if (-not [int]::TryParse($choice, [ref]$index)) {
        throw "Invalid workstream selection."
    }
    if ($index -lt 1 -or $index -gt $workstreams.Count) {
        throw "Workstream selection out of range."
    }
    return $workstreams[$index - 1]
}

function Open-ClaudeWorkstream([string]$WorkstreamName, [string]$Purpose) {
    $repo = Convert-ToGitBashPath $repoRoot
    $nameEscaped = Escape-BashSingleQuoted $WorkstreamName
    $savedPurpose = Get-ExistingPurpose "claude" $WorkstreamName
    if ($savedPurpose) {
        $Purpose = $savedPurpose
    }
    $purposeArg = ""
    if ($Purpose) {
        $purposeEscaped = Escape-BashSingleQuoted $Purpose
        $purposeArg = " CANOMPX3_WORKSTREAM_PURPOSE='$purposeEscaped'"
    }
    Run-GitBash "cd '$repo' &&${purposeArg} exec ./scripts/infra/claude-worktree.sh open '$nameEscaped'"
}

function Open-CodexWorkstream([string]$WorkstreamName, [string]$Purpose, [switch]$SearchMode) {
    $repo = Convert-ToWslPath $repoRoot
    $nameEscaped = Escape-BashSingleQuoted $WorkstreamName
    $savedPurpose = Get-ExistingPurpose "codex" $WorkstreamName
    if ($savedPurpose) {
        $Purpose = $savedPurpose
        if ($savedPurpose -eq "Investigate / search") {
            $SearchMode = $true
        }
    }
    $purposePrefix = ""
    if ($Purpose) {
        $purposeEscaped = Escape-BashSingleQuoted $Purpose
        $purposePrefix = "CANOMPX3_WORKSTREAM_PURPOSE='$purposeEscaped' "
    }
    if ($SearchMode) {
        Run-WSL "cd '$repo' && ${purposePrefix}exec ./scripts/infra/codex-worktree.sh search '$nameEscaped'"
    }
    Run-WSL "cd '$repo' && ${purposePrefix}exec ./scripts/infra/codex-worktree.sh open '$nameEscaped'"
}

function Prompt-WorkstreamName() {
    $value = Read-Host "Workstream name"
    if (-not $value) {
        throw "Workstream name required."
    }
    return $value
}

function Select-WorkstreamPurpose() {
    Write-Host ""
    Write-Host "Pick the purpose" -ForegroundColor Cyan
    Write-Host "[1] Build / edit"
    Write-Host "[2] Investigate / search"
    Write-Host "[3] Review / verify"
    Write-Host ""
    $choice = Read-Host ">>>"
    switch ($choice) {
        "1" { return @{ Label = "Build / edit"; RecommendedTool = "codex"; SearchMode = $false } }
        "2" { return @{ Label = "Investigate / search"; RecommendedTool = "codex"; SearchMode = $true } }
        "3" { return @{ Label = "Review / verify"; RecommendedTool = "claude"; SearchMode = $false } }
        default { throw "Invalid purpose selection." }
    }
}

function Select-AgentForPurpose([hashtable]$PurposeInfo) {
    Write-Host ""
    Write-Host ("Recommended agent: {0}" -f $PurposeInfo.RecommendedTool) -ForegroundColor Cyan
    Write-Host "[1] Use recommended"
    Write-Host "[2] Claude"
    Write-Host "[3] Codex"
    Write-Host ""
    $choice = Read-Host ">>>"
    switch ($choice) {
        "" { return $PurposeInfo.RecommendedTool }
        "1" { return $PurposeInfo.RecommendedTool }
        "2" { return "claude" }
        "3" { return "codex" }
        default { throw "Invalid agent selection." }
    }
}

function Run-Menu() {
    while ($true) {
        Clear-Host
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host " AI WORKSTREAMS" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Purpose: run one problem per isolated workstream so Claude and Codex do not stomp on each other." -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "[1] Start new workstream"
        Write-Host "[2] Continue workstream"
        Write-Host "[3] Finish workstream"
        Write-Host "[4] Show active workstreams"
        Write-Host "[5] Clean stale workstream records"
        Write-Host "[Q] Quit"
        Write-Host ""

        $choice = Read-Host ">>>"
        switch ($choice.ToLowerInvariant()) {
            "1" {
                $name = Prompt-WorkstreamName
                $purpose = Select-WorkstreamPurpose
                $agent = Select-AgentForPurpose $purpose
                if ($agent -eq "claude") {
                    Open-ClaudeWorkstream $name $purpose.Label
                } else {
                    Open-CodexWorkstream $name $purpose.Label -SearchMode:$purpose.SearchMode
                }
                return
            }
            "2" {
                $workstream = Select-ManagedWorkstream
                if ($null -ne $workstream) {
                    if ($workstream.tool -eq "claude") {
                        Open-ClaudeWorkstream $workstream.name $workstream.purpose
                    } elseif ($workstream.tool -eq "codex") {
                        $searchMode = $workstream.purpose -eq "Investigate / search"
                        Open-CodexWorkstream $workstream.name $workstream.purpose -SearchMode:$searchMode
                    } else {
                        throw "Unsupported workstream tool: $($workstream.tool)"
                    }
                    return
                }
                Read-Host "Press Enter to continue"
            }
            "3" {
                $workstream = Select-ManagedWorkstream
                if ($null -ne $workstream) {
                    Run-Manager @("close", "--tool", $workstream.tool, "--name", $workstream.name, "--force", "--drop-branch")
                    return
                }
                Read-Host "Press Enter to continue"
            }
            "4" {
                Show-ManagedWorkstreams (Get-ManagedWorkstreams)
                Read-Host "Press Enter to continue"
            }
            "5" {
                Run-Manager @("prune")
                return
            }
            "q" {
                return
            }
            default {
                Write-Host "Invalid choice." -ForegroundColor Yellow
                Start-Sleep -Seconds 1
            }
        }
    }
}

switch ($Mode) {
    "claude" {
        if (-not $Task) {
            $Task = Prompt-WorkstreamName
        }
        Open-ClaudeWorkstream $Task "Build / edit"
    }

    "codex" {
        if (-not $Task) {
            $Task = Prompt-WorkstreamName
        }
        Open-CodexWorkstream $Task "Build / edit"
    }

    "codex-search" {
        if (-not $Task) {
            $Task = Prompt-WorkstreamName
        }
        Open-CodexWorkstream $Task "Investigate / search" -SearchMode
    }

    "list" {
        Show-ManagedWorkstreams (Get-ManagedWorkstreams)
    }

    "resume" {
        $workstream = Select-ManagedWorkstream
        if ($null -eq $workstream) {
            exit 0
        }
        if ($workstream.tool -eq "claude") {
            Open-ClaudeWorkstream $workstream.name $workstream.purpose
        } elseif ($workstream.tool -eq "codex") {
            $searchMode = $workstream.purpose -eq "Investigate / search"
            Open-CodexWorkstream $workstream.name $workstream.purpose -SearchMode:$searchMode
        } else {
            throw "Unsupported workstream tool: $($workstream.tool)"
        }
    }

    "close" {
        if (-not $Task) {
            throw "Workstream name required for close."
        }
        if (-not $Tool) {
            throw "Tool required for close. Use claude or codex."
        }
        $result = Invoke-Manager @("close", "--tool", $Tool, "--name", $Task, "--force", "--drop-branch")
        if (-not $result.Success) {
            throw $result.Error
        }
        if ($result.Output) {
            $result.Output | Write-Output
        }
    }

    "close-pick" {
        $workstream = Select-ManagedWorkstream
        if ($null -eq $workstream) {
            exit 0
        }
        $result = Invoke-Manager @("close", "--tool", $workstream.tool, "--name", $workstream.name, "--force", "--drop-branch")
        if (-not $result.Success) {
            throw $result.Error
        }
        if ($result.Output) {
            $result.Output | Write-Output
        }
    }

    "menu" {
        Run-Menu
    }

    "prune" {
        $result = Invoke-Manager @("prune")
        if (-not $result.Success) {
            throw $result.Error
        }
        if ($result.Output) {
            $result.Output | Write-Output
        }
    }
}
