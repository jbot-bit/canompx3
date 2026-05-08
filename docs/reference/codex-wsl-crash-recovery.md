# Codex + WSL Crash Recovery

This runbook covers the primary Codex-on-Windows workflow for this repo:

- Windows Codex app
- repo stored under WSL home, for example `~/canompx3`
- WSL CLI and Windows app sharing state via `CODEX_HOME`

Use this guide when Codex feels like it is "making the machine crash" but the
actual failure may be in WSL, the Windows/WSL boundary, or a stale Codex build.

## Official Sources

- OpenAI Codex protected paths in `workspace-write`:
  - `https://developers.openai.com/codex/config-advanced#named-permission-profiles`
- OpenAI Codex Windows and WSL troubleshooting:
  - `https://developers.openai.com/codex/windows#troubleshooting-and-faq`
- OpenAI Codex Windows app update instructions:
  - `https://developers.openai.com/codex/app/windows#download-and-update-the-codex-app`
- OpenAI Windows app + WSL shared state:
  - `https://developers.openai.com/codex/app/windows#share-config-auth-and-sessions-with-wsl`
- Microsoft WSL filesystem guidance:
  - `https://learn.microsoft.com/en-us/windows/wsl/filesystems`
- Microsoft WSL VHD repair flow:
  - `https://learn.microsoft.com/en-us/windows/wsl/disk-space`

## What Is Expected vs Not Expected

Expected in Codex `workspace-write`:

- the repo root is writable
- `.git/` or `.codex/` may still be read-only inside the sandbox

Not expected:

- WSL terminals dying
- the distro remounting during active work
- repeated `uncleanly shut down` journal messages
- Codex trying to use a model that the installed build does not support

## Crash Signature From The May 8, 2026 Incident

Treat this pattern as a **real WSL reset signature**, not just a Codex
permissions quirk:

- `Operation canceled @p9io.cpp:258 (AcceptAsync)`
- `EXT4-fs ... unmounting filesystem`
- immediate remount of the distro filesystem
- `systemd-journald ... uncleanly shut down`

This repo also saw a separate Codex compatibility error in the same window:

- `The 'gpt-5.5' model requires a newer version of Codex. Please upgrade to the latest app or CLI and try again.`

## Recovery Ladder

Run the steps in order.

1. Update the Codex app from the Microsoft Store.
   The official OpenAI path is Microsoft Store -> `Downloads` -> `Check for updates`.
2. Verify the installed build:

   ```bash
   codex --version
   ```

3. Verify this repo is under the WSL filesystem, not `/mnt/c/...`.
4. Update WSL and restart the distro from Windows:

   ```powershell
   wsl --update
   wsl --shutdown
   ```

5. Reopen WSL and rerun the repo doctor:

   ```bash
   python3 scripts/infra/codex_local_env.py doctor --platform wsl
   ```

6. If the doctor only warns about sandbox-protected paths, treat that as
   expected for Codex `workspace-write`.
7. If the current-boot journal still shows the reset signature and terminals
   keep dying, follow the Microsoft repair flow for the WSL VHD.

## Shared Windows App + WSL State

OpenAI documents two ways to share state between the Windows app and WSL. The
preferred path for this machine is a shared `CODEX_HOME`.

Managed WSL launchers in this repo now auto-export the default shared path when
it exists. That keeps config, auth, and session history aligned across the
Windows app plus multiple supported WSL terminals. If you launch `codex`
outside those scripts, export the same path manually:

In WSL:

```bash
export CODEX_HOME=/mnt/c/Users/joshd/.codex
```

If this is your daily-driver setup, add the same line to `~/.bashrc` or
`~/.zshrc`.

This shares:

- Codex config
- cached auth
- session history

If your WSL username does not match the Windows username, set
`CANOMPX3_SHARED_CODEX_HOME` to the correct `/mnt/c/Users/<windows-user>/.codex`
path before using the managed launchers.

It does **not** mean the repo should move to `/mnt/c`. Keep the repo in the
WSL filesystem.

## Microsoft VHD Repair Flow

Use this only if WSL continues to reset or falls back to a genuinely read-only
root filesystem.

From Windows PowerShell, follow the Microsoft doc sequence:

1. Shutdown WSL:

   ```powershell
   wsl --shutdown
   ```

2. Mount the distro VHD with `wsl.exe --mount`.
3. Use `wsl.exe lsblk` to identify the device name.
4. Run:

   ```powershell
   wsl.exe sudo e2fsck -f /dev/<device>
   ```

5. Unmount with:

   ```powershell
   wsl.exe --unmount
   ```

Use the Microsoft doc above for the exact path and device-discovery details.

## Evidence To Collect Before Escalating

- `codex --version`
- `python3 scripts/infra/codex_local_env.py doctor --platform wsl`
- whether the failure was:
  - Windows Codex app
  - WSL CLI
  - VS Code extension
- `journalctl -b --no-pager | rg "AcceptAsync|unmounting filesystem|uncleanly shut down"`
- recent Codex log lines from:
  - `~/.codex/log/codex-tui.log`

## Fast Triage Rules

- If only `.git/` and `.codex/` are read-only, but the repo root is writable:
  - this is probably expected Codex sandbox behavior
- If the journal shows `AcceptAsync` plus an unmount/remount sequence:
  - this is a real WSL reset problem
- If Codex logs show `requires a newer version of Codex`:
  - this is a model/build mismatch and should be fixed separately from WSL
