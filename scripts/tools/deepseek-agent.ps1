# DeepSeek Agent Launcher (delegator)
#
# Backwards-compat shim for `opencode-agent.ps1`. PR #242 (2026-05-06) shipped
# this as an aider launcher; aider hid tool use and forced /add ceremony, so
# the harness was replaced with opencode (https://opencode.ai). This delegator
# preserves the `deepseek` shim entry point (`~/.local/bin/deepseek.bat` and
# the PS profile function) so existing muscle memory keeps working.

$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
& (Join-Path $dir "opencode-agent.ps1") @args
exit $LASTEXITCODE
