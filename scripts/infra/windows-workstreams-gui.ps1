param(
    [string]$Action = "",
    [string]$Task = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$launcherPs1 = Join-Path $repoRoot "scripts\infra\windows-agent-launch.ps1"

function Invoke-LauncherMode {
    param(
        [Parameter(Mandatory = $true)][string]$Mode,
        [string]$TaskName = ""
    )

    if ($env:CANOMPX3_WINDOWS_LAUNCH_ECHO_ONLY) {
        if ($TaskName) {
            Write-Output "MODE=$Mode TASK=$TaskName"
        } else {
            Write-Output "MODE=$Mode"
        }
        return
    }

    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $launcherPs1,
        "-Mode", $Mode
    )
    if ($TaskName) {
        $args += @("-Task", $TaskName)
    }
    Start-Process powershell.exe -ArgumentList $args | Out-Null
}

function Require-Task([System.Windows.Forms.TextBox]$TaskBox) {
    $taskName = $TaskBox.Text.Trim()
    if (-not $taskName) {
        [System.Windows.Forms.MessageBox]::Show(
            "Enter a workstream name first.",
            "AI Workstreams",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        ) | Out-Null
        return $null
    }
    return $taskName
}

if ($Action) {
    switch ($Action.ToLowerInvariant()) {
        "codex" { Invoke-LauncherMode -Mode "codex" -TaskName $Task; exit 0 }
        "claude" { Invoke-LauncherMode -Mode "claude" -TaskName $Task; exit 0 }
        "search" { Invoke-LauncherMode -Mode "codex-search" -TaskName $Task; exit 0 }
        "resume" { Invoke-LauncherMode -Mode "resume"; exit 0 }
        "list" { Invoke-LauncherMode -Mode "list"; exit 0 }
        "finish" { Invoke-LauncherMode -Mode "close-pick"; exit 0 }
        "clean" { Invoke-LauncherMode -Mode "prune"; exit 0 }
        "green-codex" { Invoke-LauncherMode -Mode "green-codex"; exit 0 }
        "green-claude" { Invoke-LauncherMode -Mode "green-claude"; exit 0 }
        default { throw "Unknown GUI action: $Action" }
    }
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form
$form.Text = "AI Workstreams"
$form.StartPosition = "CenterScreen"
$form.Size = New-Object System.Drawing.Size(560, 300)
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.MinimizeBox = $true
$form.BackColor = [System.Drawing.Color]::FromArgb(248, 249, 251)

$title = New-Object System.Windows.Forms.Label
$title.Text = "Open an isolated AI workstream"
$title.Font = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Bold)
$title.AutoSize = $true
$title.Location = New-Object System.Drawing.Point(18, 16)
$form.Controls.Add($title)

$subtitle = New-Object System.Windows.Forms.Label
$subtitle.Text = "Type a task name, then click Claude or Codex. Utility actions are on the right."
$subtitle.AutoSize = $true
$subtitle.Location = New-Object System.Drawing.Point(20, 46)
$form.Controls.Add($subtitle)

$taskLabel = New-Object System.Windows.Forms.Label
$taskLabel.Text = "Task name"
$taskLabel.AutoSize = $true
$taskLabel.Location = New-Object System.Drawing.Point(20, 84)
$form.Controls.Add($taskLabel)

$taskBox = New-Object System.Windows.Forms.TextBox
$taskBox.Location = New-Object System.Drawing.Point(20, 104)
$taskBox.Size = New-Object System.Drawing.Size(300, 24)
$taskBox.Font = New-Object System.Drawing.Font("Segoe UI", 10)
$form.Controls.Add($taskBox)

$hint = New-Object System.Windows.Forms.Label
$hint.Text = "Examples: operator-alerts, startup-brain, dashboard-cleanup"
$hint.AutoSize = $true
$hint.Location = New-Object System.Drawing.Point(20, 134)
$form.Controls.Add($hint)

function New-Button {
    param(
        [string]$Text,
        [int]$X,
        [int]$Y,
        [int]$W = 140,
        [int]$H = 34
    )
    $button = New-Object System.Windows.Forms.Button
    $button.Text = $Text
    $button.Location = New-Object System.Drawing.Point($X, $Y)
    $button.Size = New-Object System.Drawing.Size($W, $H)
    $button.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    return $button
}

$codexBtn = New-Button -Text "Open Codex" -X 20 -Y 176
$codexBtn.Add_Click({
    $taskName = Require-Task $taskBox
    if ($taskName) {
        Invoke-LauncherMode -Mode "codex" -TaskName $taskName
        $form.Close()
    }
})
$form.Controls.Add($codexBtn)

$claudeBtn = New-Button -Text "Open Claude" -X 170 -Y 176
$claudeBtn.Add_Click({
    $taskName = Require-Task $taskBox
    if ($taskName) {
        Invoke-LauncherMode -Mode "claude" -TaskName $taskName
        $form.Close()
    }
})
$form.Controls.Add($claudeBtn)

$searchBtn = New-Button -Text "Search Task" -X 20 -Y 216
$searchBtn.Add_Click({
    $taskName = Require-Task $taskBox
    if ($taskName) {
        Invoke-LauncherMode -Mode "codex-search" -TaskName $taskName
        $form.Close()
    }
})
$form.Controls.Add($searchBtn)

$greenCodexBtn = New-Button -Text "Green Codex" -X 170 -Y 216
$greenCodexBtn.Add_Click({
    Invoke-LauncherMode -Mode "green-codex"
    $form.Close()
})
$form.Controls.Add($greenCodexBtn)

$greenClaudeBtn = New-Button -Text "Green Claude" -X 320 -Y 216
$greenClaudeBtn.Add_Click({
    Invoke-LauncherMode -Mode "green-claude"
    $form.Close()
})
$form.Controls.Add($greenClaudeBtn)

$listBtn = New-Button -Text "List" -X 360 -Y 84 -W 150
$listBtn.Add_Click({
    Invoke-LauncherMode -Mode "list"
})
$form.Controls.Add($listBtn)

$resumeBtn = New-Button -Text "Resume" -X 360 -Y 124 -W 150
$resumeBtn.Add_Click({
    Invoke-LauncherMode -Mode "resume"
    $form.Close()
})
$form.Controls.Add($resumeBtn)

$finishBtn = New-Button -Text "Finish / Close" -X 360 -Y 164 -W 150
$finishBtn.Add_Click({
    Invoke-LauncherMode -Mode "close-pick"
    $form.Close()
})
$form.Controls.Add($finishBtn)

$cleanBtn = New-Button -Text "Clean Stale" -X 360 -Y 204 -W 150
$cleanBtn.Add_Click({
    Invoke-LauncherMode -Mode "prune"
})
$form.Controls.Add($cleanBtn)

$taskBox.Focus() | Out-Null
[void]$form.ShowDialog()
