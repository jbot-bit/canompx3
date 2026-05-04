# git-mcp Q3 Mitigation — Implementation

**Date:** 2026-05-01
**Branch:** `tooling/git-mcp-eval`
**Companion verdict:** [`safety-precheck.md`](safety-precheck.md)

This document records the implementation that flips the
`safety-precheck.md` Q4 verdict from **GO-WITH-MITIGATION** to **GO**
contingent on the settings.json patch below being applied.

---

## Summary

A new PostToolUse hook (`mcp-git-guard.py`) closes both branch-flip
protection bypasses identified in the eval:

1. **PostToolUse(Bash) layer bypass** — closed by registering a second
   PostToolUse entry with matcher `mcp__git__.*`.
2. **Pre-commit layer bypass** — closed in-hook for the MCP path: the
   guard refuses `mcp__git__git_commit` on branch drift, since
   `repo.index.commit()` in upstream `mcp-server-git` never invokes
   client-side hooks.

A shared helper module (`_branch_state.py`) was extracted so both
guards consume one canonical implementation of `git_dir()`,
`current_branch()`, and `branch_at_start()`. This satisfies
`.claude/rules/institutional-rigor.md` rule 4 ("Delegate to canonical
sources — never re-encode") and prevents silent drift between the two
guards as they evolve.

---

## Files landed in this commit

| File | Lines | Purpose |
|---|---|---|
| `.claude/hooks/_branch_state.py` | ~80 | Shared canonical helpers. |
| `.claude/hooks/mcp-git-guard.py` | ~150 | New PostToolUse(`mcp__git__.*`) hook. |
| `.claude/hooks/branch-flip-guard.py` | refactored | Imports shared helpers; behaviour unchanged. |
| `tests/test_hooks/test_branch_state.py` | ~110 | Unit tests for the shared module. |
| `tests/test_hooks/test_mcp_git_guard.py` | ~250 | Unit + subprocess tests for the new hook. |

---

## Hook behavior matrix

| Tool name | Lock present, branch drift | Result |
|---|---|---|
| `mcp__git__git_commit` | yes | **BLOCK exit 2** (sole layer of protection — pre-commit bypassed) |
| `mcp__git__git_checkout` | yes | BLOCK exit 2 |
| `mcp__git__git_create_branch` | yes | BLOCK exit 2 |
| `mcp__git__git_reset` | yes | BLOCK exit 2 |
| `mcp__git__git_add` | yes | BLOCK exit 2 |
| `mcp__git__git_branch` | yes | BLOCK exit 2 |
| `mcp__git__git_status` | yes | exit 0 (read-only — no harm) |
| `mcp__git__git_diff*` | yes | exit 0 |
| `mcp__git__git_log` | yes | exit 0 |
| `mcp__git__git_show` | yes | exit 0 |
| any `mcp__git__*` | no drift | exit 0 |
| any `mcp__git__*` | missing/corrupt lock | exit 0 (fail-safe) |
| `Bash` / other MCP | any | exit 0 (matcher misfire defense) |

---

## settings.json patch — TO APPLY MANUALLY

This patch is **NOT applied by this commit** — same pattern as the
other two evals in this series. The user applies it after the hook
lands.

Add the new entry alongside the existing `"matcher": "Bash"` PostToolUse
hook (the order does not matter; matchers are independent):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "mcp__git__.*",
        "hooks": [
          {
            "type": "command",
            "command": "python C:/Users/joshd/canompx3/.claude/hooks/mcp-git-guard.py",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

Notes on the matcher pattern:

- Claude Code matches MCP tool names of the form `mcp__<server>__<tool>`
  literally; the regex `mcp__git__.*` covers every tool the official
  `mcp-server-git` exposes when registered under server name `git` in
  `.mcp.json`.
- If the server is registered under a different name (e.g. `git-tools`),
  update the matcher and the test fixtures together.
- The `timeout: 5` mirrors the existing `branch-flip-guard.py`
  registration. The hook runs two `git` subprocess calls; sub-second on
  a healthy repo, fail-safe on a slow one.

---

## Test results

```
$ python -m pytest tests/test_hooks/test_branch_state.py \
    tests/test_hooks/test_mcp_git_guard.py \
    tests/test_hooks/test_branch_flip_guard.py -v

35 passed in 2.36s
```

The subprocess test class (`TestSubprocessContract`) invokes the real
hook script with a real JSON event on stdin via `subprocess.run`,
proving the actual contract Claude Code uses (stdin -> exit code +
stderr). Other classes use `importlib.util.module_from_spec` +
`monkeypatch` for fast in-process coverage of the matrix above.

---

## Verdict flip

`safety-precheck.md` Q4 was **GO-WITH-MITIGATION**, contingent on
three deliverables:

1. New PostToolUse hook with matcher `mcp__git__.*` — **DONE** (this commit).
2. Hook MUST refuse `mcp__git__git_commit` on branch drift — **DONE**
   (verified by `TestBlockOnDrift::test_write_tool_blocks[mcp__git__git_commit]`).
3. Apply the settings.json patch documented above — **PENDING USER
   ACTION** (the eval pattern documents the patch but does not apply
   it automatically).

Once item 3 is applied (user runs the patch in the canonical worktree
`C:/Users/joshd/canompx3/.claude/settings.json`), the verdict flips to
**GO** for adopting `mcp-server-git` in `.mcp.json`.

Without item 3, the hook script exists but is not registered with
Claude Code; the bypass remains and the verdict remains
**GO-WITH-MITIGATION**.

---

## Forward pointers

- `tests/test_hooks/test_mcp_git_guard.py` — full coverage of the hook.
- `tests/test_hooks/test_branch_state.py` — coverage of the shared module.
- `.claude/rules/branch-flip-protection.md` — original protection-layer rule.
- `.claude/rules/institutional-rigor.md` — rule 4 (canonical-source delegation).
