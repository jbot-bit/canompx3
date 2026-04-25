# Claude Code Global Hardening — Design (2026-04-25)

Status: DESIGN — awaiting approval.
Target file: `C:/Users/joshd/.claude/settings.json` (global, cross-project).
Scope: client tool safety policy only. No repo code, pipeline, or trading logic touched.

## Why

Current global config has `Bash(*)` allow with only `EnterPlanMode` in deny. One poisoned PDF,
rogue MCP, or runaway subagent could force-push main, wipe uncommitted work, exfiltrate `.env`
or SSH keys, or upload strategy files to a public paste site. Hooks are advisory; sandbox
excludes git / gh / uv / ruff. The deny-list and secret-read-blocklist are the only mechanical
brakes.

## Changes (all merged into existing global settings — nothing removed)

1. `permissions.deny` — never-legitimate destructive commands (force-push, dd, mkfs, chmod 777).
2. `permissions.ask` — sometimes-needed destructive commands (rm -rf, reset --hard, clean -fd,
   checkout -- .). Prompts every time instead of silent block — preserves legitimate cleanup.
3. `sandbox.filesystem.denyRead` — block reads of SSH / AWS / .env / .pem / .key / secrets/**.
4. `sandbox.network.deniedDomains` — block pastebin.com, paste.ee, 0x0.st, transfer.sh, ix.io.
5. `fileCheckpointingEnabled: true` — enables /rewind after accidental edits.
6. `showClearContextOnPlanAccept: true` — offers clean-slate option after plan approval.
7. `attribution.commit` — strip stale Opus 4.6 version label so it cannot go stale again.

## Failure modes

- Sandbox denyRead / deniedDomains may need session restart to take effect.
- Bash deny / ask rules hot-reload.
- Subagent rm-rf loops will now hit an ask prompt. Intended behavior.

## Behavioral smoke tests (user-runnable after apply)

- `git push --force` → expect silent block (deny rule).
- `rm -rf /tmp/x` → expect permission prompt (ask rule).
- Next commit trailer → expect no model version number.

## Rollback

Single file, merge edit only. Remove the added keys to revert. File is in Windows file history.

## NOT included (deferred)

- `strictKnownMarketplaces` — schema marks it as managed / admin-only, does not take effect in
  user settings.
- Version-control the global settings file — worth doing separately, out of scope here.
- Plugin enable/disable toggles — not a safety issue.
