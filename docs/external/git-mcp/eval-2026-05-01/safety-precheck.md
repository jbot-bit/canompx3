# git-mcp Safety Pre-check — 2026-05-01

**Worktree:** `C:/Users/joshd/canompx3-git-mcp-eval`
**Branch:** `tooling/git-mcp-eval` (scaffolded from `origin/main` @ `171fd73a`)
**Question under evaluation:** does adopting the official MCP git server (`mcp-server-git`,
https://github.com/modelcontextprotocol/servers/tree/main/src/git) into `.mcp.json` regress
the project's existing branch-flip protection stack?

**Verdict (TL;DR):** **GO-WITH-MITIGATION** — **MITIGATION IMPLEMENTED 2026-05-01.**
Both layers of branch-flip protection are **bypassed** by MCP git tool calls in their
current form. A new PostToolUse hook keyed on the matcher `mcp__git__.*` is required
before adoption — that hook (`mcp-git-guard.py`) plus a shared canonical-helper module
(`_branch_state.py`) landed alongside this doc; see Q4 § "Mitigation status" and
[`mitigation-hook-impl.md`](mitigation-hook-impl.md). Verdict flips to full **GO** once
the documented `settings.json` patch is applied to the canonical worktree.

Project rule reminders:
- `.claude/rules/branch-flip-protection.md` — guard is intentionally PostToolUse(Bash) +
  pre-commit step 0c as the backstop.
- `feedback_branch_flip_guard.md` — the protection stack exists for documented past
  multi-terminal incidents.
- Claude Code hook docs — `matcher` accepts regex; MCP tool names take the form
  `mcp__<server>__<tool>` and can be matched with `mcp__git__.*`.

---

## Q1 — Does `.claude/hooks/branch-flip-guard.py` fire on MCP git tool calls?

**Verdict: BYPASSED.**

**Evidence — registration (`.claude/settings.json`, lines 82-92):**
```json
"PostToolUse": [
  {
    "matcher": "Bash",
    "hooks": [
      {
        "type": "command",
        "command": "python C:/Users/joshd/canompx3/.claude/hooks/branch-flip-guard.py",
        "timeout": 5
      }
    ]
  },
```

The hook is registered ONLY against `"matcher": "Bash"` (line 84). Claude Code routes
PostToolUse events to a hook only when the tool name matches the matcher; an MCP tool call
emitted under the name `mcp__git__git_checkout` does not match `"Bash"` and therefore the
guard process is never invoked.

**Evidence — in-script tool_name guard (`.claude/hooks/branch-flip-guard.py`, lines
62-64):**
```python
tool_name = event.get("tool_name", "")
if tool_name != "Bash":
    sys.exit(0)
```

Even if a future settings.json change widened the matcher, the script itself early-exits
on any `tool_name != "Bash"`. Both layers of dispatch agree: this guard sees Bash calls
only.

**Implication:** an LLM driving a session via `mcp__git__git_checkout other-branch`
followed by file edits would experience no early "branch flipped" warning. The guard's
purpose (catch the flip after every Bash call so the user can course-correct early) is
defeated.

---

## Q2 — Does `.githooks/pre-commit` step 0c fire when commits are created via MCP `git_commit`?

**Verdict: BYPASSED (verified against upstream source, not UNVERIFIED).**

**Evidence — pre-commit step 0c (`.githooks/pre-commit`, lines 30-52):**
```bash
# 0c. Branch-flip guard — abort if HEAD branch != branch_at_start in session lock.
GIT_DIR_PATH=$(git rev-parse --git-dir 2>/dev/null || true)
if [ -n "$GIT_DIR_PATH" ] && [ -f "$GIT_DIR_PATH/.claude.pid" ]; then
    BRANCH_AT_START=$(python -c "
import json, sys
try:
    d = json.load(open('$GIT_DIR_PATH/.claude.pid'))
    print(d.get('branch_at_start', ''))
except Exception:
    pass
" 2>/dev/null || true)
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || true)
    if [ -n "$BRANCH_AT_START" ] && [ -n "$CURRENT_BRANCH" ] && [ "$BRANCH_AT_START" != "$CURRENT_BRANCH" ]; then
        echo "BLOCKED: Branch changed mid-session."
        ...
        exit 1
    fi
fi
```

Pre-commit hooks are invoked by `git commit` (the porcelain), not by libgit2 / GitPython
direct index/ref manipulation. The question is therefore empirical: does
`mcp-server-git`'s `git_commit` shell out to `git commit`, or write the commit object
directly?

**Upstream evidence — `mcp-server-git/src/mcp_server_git/server.py` (fetched live via
`gh api repos/modelcontextprotocol/servers/contents/src/git/src/mcp_server_git/server.py`,
lines 128-130):**
```python
def git_commit(repo: git.Repo, message: str) -> str:
    commit = repo.index.commit(message)
    return f"Changes committed successfully with hash {commit.hexsha}"
```

`repo.index.commit(...)` is GitPython's **direct index-to-commit-object** path: it writes
the tree, builds a commit object, and updates `HEAD` programmatically without invoking
the `git commit` porcelain. **No `pre-commit` hook fires.** This is a property of
GitPython's API, not a configuration choice — the function does not consult or run any
client-side hook.

For comparison, the same file shows the upstream is willing to shell out where
appropriate — e.g. `git_checkout` (lines 199-206) uses `repo.git.checkout(branch_name)`,
which is a thin wrapper around `subprocess`-invoked `git checkout`. But `git_commit`
deliberately uses the index API (likely for predictable hash return). The selection is
asymmetric and unfavourable for our hook stack:

| MCP tool | Upstream impl | Hooks fire? |
|---|---|---|
| `git_checkout` (line 205) | `repo.git.checkout(branch_name)` (shells out) | post-checkout hook would fire; **but** branch-flip-guard hook (Q1) does not, because matcher is `"Bash"`. |
| `git_commit` (line 129) | `repo.index.commit(message)` (direct API) | **pre-commit hook does NOT fire.** |
| `git_add` (lines 132-138) | `repo.git.add(...)` (shells out) | n/a |
| `git_reset` (line 141) | `repo.index.reset()` (direct API) | n/a |
| `git_create_branch` (line 196) | `repo.create_head(...)` (direct API) | n/a |

**Implication:** the pre-commit backstop — the layer the project specifically calls out as
catching branch flips that PostToolUse misses — does not run for MCP-driven commits.
Both layers are defeated.

---

## Q3 — Per-tool matcher mitigation: viable?

**Verdict: VIABLE, with documented gotchas.**

Claude Code's hook matcher accepts a regex. Adding a new PostToolUse entry alongside the
existing `"Bash"` matcher would catch MCP git events:

**Shape (NOT IMPLEMENTED in this task — design only):**
```json
"PostToolUse": [
  { "matcher": "Bash",            "hooks": [ ...existing branch-flip-guard.py... ] },
  { "matcher": "mcp__git__.*",    "hooks": [ ...new mcp-git-guard.py...           ] }
]
```

The new guard would:
1. Read `event["tool_name"]` and accept the `mcp__git__*` family (or filter to just the
   state-mutating subset: `mcp__git__git_checkout`, `mcp__git__git_create_branch`,
   `mcp__git__git_commit`, `mcp__git__git_reset`, `mcp__git__git_add`).
2. Read the same `.git/.claude.pid` lock file; compare `branch_at_start` to current
   branch via `git branch --show-current`; block on drift exactly as the existing guard
   does.
3. Be fail-safe (any error → `sys.exit(0)`), per the existing protection rule.

**Gotchas to call out before implementation:**

1. **Pre-commit cannot be salvaged the same way.** The PostToolUse mitigation closes the
   PostToolUse layer, but the pre-commit backstop is fundamentally bypassed by
   `repo.index.commit`. Mitigations:
   - (a) gate the new MCP-git hook to ALSO refuse `mcp__git__git_commit` when the lock
     file shows branch drift — this replaces the pre-commit layer for the MCP path.
   - (b) re-run drift check 0c logic inside the PostToolUse hook itself before allowing a
     commit through.
   - (c) restrict the `.mcp.json` allowlist to read-only tools (`git_status`,
     `git_diff_*`, `git_log`, `git_show`, `git_branch`) and continue using Bash for
     `commit`/`checkout`/`reset`. This is the simplest and tightest mitigation if speed
     gain on read-only ops is the actual goal.

2. **PostToolUse is inherently after-the-fact.** Same trade-off as the existing
   Bash-keyed guard (`branch-flip-protection.md` calls this out). For MCP `git_commit` the
   commit object already exists on the wrong branch by the time the hook fires — at
   minimum the guard should print a loud BLOCK message and refuse subsequent tool calls;
   ideally it should emit a `git update-ref` / `git reset` recovery hint. The existing
   Bash guard self-recovers because the user can `git checkout <original-branch>` and the
   PostToolUse fires again. The MCP path may need an explicit `mcp__git__git_reset`
   recovery suggestion in the BLOCK message.

3. **Matcher regex semantics.** Claude Code matches `mcp__<server>__<tool>` literally. If
   the `.mcp.json` server name is anything other than `git` (e.g. `git-tools`), the
   matcher must be updated. The verdict here assumes `.mcp.json` will register the server
   under the name `git`.

4. **Hook timeout.** Existing guard has `"timeout": 5`; the new one should match. The
   guard runs `git rev-parse` and `git branch --show-current` — sub-second on a healthy
   repo, but worktrees with a missing/locked `.git` should fail-safe to exit 0.

5. **Worktree `.git` path resolution.** The existing guard uses
   `git rev-parse --git-dir`, which resolves correctly from inside a worktree. The new
   hook must do the same — the canonical path is `worktree/.git/worktrees/<name>/`,
   which contains its own `.claude.pid` written by `session-start.py`.

---

## Q4 — Final verdict

**GO-WITH-MITIGATION** — **MITIGATION IMPLEMENTED**, see
`mitigation-hook-impl.md` and the commit landing this doc edit.

Adoption is acceptable IF AND ONLY IF, in the same change-set as the `.mcp.json` edit:

1. A new PostToolUse hook is registered with matcher `mcp__git__.*` (or the explicit
   state-mutating subset) that runs an MCP-aware variant of `branch-flip-guard.py`.
2. That hook MUST refuse `mcp__git__git_commit` on branch drift (the pre-commit backstop
   is bypassed by `repo.index.commit` and cannot be relied on for the MCP path).
3. OR — simpler — the `.mcp.json` allowlist for the git server is restricted to
   read-only tools (`git_status`, `git_diff_*`, `git_log`, `git_show`, `git_branch`),
   and write ops continue to flow through Bash.

Without (1)+(2) or (3): **NO-GO.** The branch-flip incident class documented in
`feedback_branch_flip_guard.md` would re-open silently, with no PostToolUse warning and
no pre-commit backstop.

**Recommended next task:** if the project elects to adopt the full toolset, write the
`mcp-git-guard.py` hook and the `.mcp.json` + `settings.json` patch in a single PR; gate
the merge on a manual test that proves a deliberately-flipped branch + `mcp__git__git_commit`
is BLOCKED. If the project elects the read-only restriction, write the `.mcp.json` with
an explicit `disabled` list and a CHANGELOG note pointing here.

### Mitigation status (2026-05-01)

- **(1) PostToolUse hook implemented**: `.claude/hooks/mcp-git-guard.py`
  with matcher `mcp__git__.*`. Companion shared module
  `.claude/hooks/_branch_state.py` extracted; `branch-flip-guard.py`
  refactored to consume it (both guards now share one canonical
  implementation per `.claude/rules/institutional-rigor.md` rule 4).
- **(2) `mcp__git__git_commit` blocked on drift**: verified by
  `tests/test_hooks/test_mcp_git_guard.py::TestBlockOnDrift::test_write_tool_blocks[mcp__git__git_commit]`
  (parametrised across all six write tools). Pre-commit-bypass class
  closed for the MCP path.
- **Tests**: 35/35 passing across `test_branch_state.py`,
  `test_mcp_git_guard.py`, and the original `test_branch_flip_guard.py`
  (refactor preserved original behaviour).
- **(3) settings.json patch**: documented inline in
  `mitigation-hook-impl.md` § "settings.json patch — TO APPLY
  MANUALLY". **Not applied in this commit** (per the eval-doc-only
  pattern shared with the other two evals); user applies the patch
  separately. Until then, verdict remains GO-WITH-MITIGATION rather
  than full GO.

**Cross-reference:** [`mitigation-hook-impl.md`](mitigation-hook-impl.md).

---

## Provenance / verification log

- Upstream `server.py` fetched 2026-05-01 via
  `gh api repos/modelcontextprotocol/servers/contents/src/git/src/mcp_server_git/server.py`
  (587 lines; SHA tracked by GitHub at fetch time). Quoted lines 128-130, 132-138,
  140-141, 185-197, 199-206 above are verbatim from that file.
- Local files quoted are the canonical project versions reachable from this worktree at
  `tooling/git-mcp-eval` (`171fd73a`); they are unchanged in this worktree.
- The temporary copy of the upstream file used for grep verification
  (`.tmp_upstream_server.py`) is gitignored / removed before commit; this verdict doc is
  the only artifact persisted.
