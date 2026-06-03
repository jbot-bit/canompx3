<#
.SYNOPSIS
  On-demand toggle for the Firecrawl Claude Code plugin (canompx3).

.DESCRIPTION
  Firecrawl is default OFF (committed .claude/settings.json sets it false,
  matching the explicit-only policy in .claude/rules/plugin-routing.md).
  This script flips it ON only when you need web scraping, via the highest-
  precedence layer: the untracked .claude/settings.local.json. Disable removes
  that override so the committed default (OFF) takes over again.

  Plugin enablement precedence (lowest -> highest):
    ~/.claude/settings.json        (user/global)   firecrawl: false
    .claude/settings.json          (project, git)  firecrawl: false  <- default OFF
    .claude/settings.local.json    (project-local) firecrawl: <this script>

  Only the local layer is touched. All other keys in settings.local.json are
  preserved (JSON round-trip, not text munging).

  NOTE: Plugin enable/disable is read at Claude Code startup. After toggling,
  you MUST restart the Claude session (or reload the window) for the change to
  take effect in-session. The script prints this reminder on enable/disable.

.PARAMETER Mode
  enable | disable | status   (default: status)

.EXAMPLE
  ./scripts/tools/firecrawl_mode.ps1 enable
  ./scripts/tools/firecrawl_mode.ps1 disable
  ./scripts/tools/firecrawl_mode.ps1 status
#>
[CmdletBinding()]
param(
    [ValidateSet('enable', 'disable', 'status')]
    [string]$Mode = 'status'
)

$ErrorActionPreference = 'Stop'

$PluginKey   = 'firecrawl@claude-plugins-official'
$RepoRoot    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # scripts/tools -> repo root
$LocalPath   = Join-Path $RepoRoot '.claude/settings.local.json'
$ProjectPath = Join-Path $RepoRoot '.claude/settings.json'
$UserPath    = Join-Path $env:USERPROFILE '.claude/settings.json'

function Read-Json([string]$Path) {
    # Returns $null if absent; tolerates UTF-8 BOM.
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $raw = Get-Content -LiteralPath $Path -Raw -Encoding utf8
    if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
    return $raw | ConvertFrom-Json
}

function Get-PluginFlag($obj) {
    # $null = key absent at this layer; $true/$false otherwise.
    if ($null -eq $obj) { return $null }
    if ($null -eq $obj.enabledPlugins) { return $null }
    $ep = $obj.enabledPlugins
    if ($ep.PSObject.Properties.Name -contains $PluginKey) { return [bool]$ep.$PluginKey }
    return $null
}

function Get-EffectiveState {
    # Highest layer that sets the key wins. Returns @{ value=<bool>; source=<path> }.
    foreach ($layer in @(
            @{ Path = $LocalPath;   Label = '.claude/settings.local.json (local)' },
            @{ Path = $ProjectPath; Label = '.claude/settings.json (committed)'   },
            @{ Path = $UserPath;    Label = '~/.claude/settings.json (user)'       }
        )) {
        $flag = Get-PluginFlag (Read-Json $layer.Path)
        if ($null -ne $flag) { return @{ Value = $flag; Source = $layer.Label } }
    }
    return @{ Value = $false; Source = '(no layer sets it -> default OFF)' }
}

function Show-Status {
    $eff = Get-EffectiveState
    if ($eff.Value) { $onoff = 'ON';  $color = 'Yellow' } else { $onoff = 'OFF'; $color = 'Green' }
    Write-Host ""
    Write-Host "Firecrawl effective state: $onoff" -ForegroundColor $color
    Write-Host "  decided by: $($eff.Source)"
    Write-Host "  per-layer:"
    Write-Host ("    user     ~/.claude/settings.json      : {0}" -f (_fmt (Get-PluginFlag (Read-Json $UserPath))))
    Write-Host ("    project  .claude/settings.json        : {0}" -f (_fmt (Get-PluginFlag (Read-Json $ProjectPath))))
    Write-Host ("    local    .claude/settings.local.json  : {0}" -f (_fmt (Get-PluginFlag (Read-Json $LocalPath))))
    Write-Host ""
}

function _fmt($v) {
    if ($null -eq $v) { return '(absent)' }
    if ($v)          { return 'true' }
    return 'false'
}

function Set-LocalOverride([bool]$Value) {
    $obj = Read-Json $LocalPath
    if ($null -eq $obj) { $obj = [pscustomobject]@{} }

    if ($null -eq $obj.enabledPlugins) {
        $obj | Add-Member -NotePropertyName 'enabledPlugins' -NotePropertyValue ([pscustomobject]@{}) -Force
    }
    if ($obj.enabledPlugins.PSObject.Properties.Name -contains $PluginKey) {
        $obj.enabledPlugins.$PluginKey = $Value
    } else {
        $obj.enabledPlugins | Add-Member -NotePropertyName $PluginKey -NotePropertyValue $Value -Force
    }
    Write-LocalJson $obj
}

function Remove-LocalOverride {
    # Disable = remove the local key so the committed default (OFF) governs.
    $obj = Read-Json $LocalPath
    if ($null -eq $obj -or $null -eq $obj.enabledPlugins) { return }
    if ($obj.enabledPlugins.PSObject.Properties.Name -contains $PluginKey) {
        $obj.enabledPlugins.PSObject.Properties.Remove($PluginKey)
    }
    # Drop an empty enabledPlugins object to avoid leaving cruft.
    if (($obj.enabledPlugins.PSObject.Properties | Measure-Object).Count -eq 0) {
        $obj.PSObject.Properties.Remove('enabledPlugins')
    }
    Write-LocalJson $obj
}

function Write-LocalJson($obj) {
    # LF line endings, no BOM (file is untracked + LF-consistent, so its exact
    # whitespace style is never diffed by git). Plain ConvertTo-Json: all keys
    # and values round-trip intact; PS 5.1's verbose indent is cosmetic only.
    $json = ($obj | ConvertTo-Json -Depth 20) -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($LocalPath, $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
}

switch ($Mode) {
    'status' {
        Show-Status
    }
    'enable' {
        Write-Host "Before:" ; Show-Status
        Set-LocalOverride $true
        Write-Host "After:" ; Show-Status
        Write-Host "Firecrawl ENABLED via local override. Restart the Claude session (or reload window) for it to load." -ForegroundColor Yellow
    }
    'disable' {
        Write-Host "Before:" ; Show-Status
        Remove-LocalOverride
        Write-Host "After:" ; Show-Status
        Write-Host "Firecrawl DISABLED (local override removed; committed default is OFF). Restart the Claude session for it to unload." -ForegroundColor Green
    }
}
