---
name: optimize-setup
description: >
  Repo-native Claude Code setup auditor. Grounds in THIS repo's actual config
  (settings.json hooks, .claude/agents, .mcp.json, skills) and the OFFICIAL hook
  docs before recommending anything — never generic. Finds wired-vs-on-disk hook
  drift, invalid hook-event keys, superseded/dormant hooks, and genuinely-missing
  high-EV automations. Read-only audit by default; applies a change only on
  explicit approval.
when_to_use: ["optimize my setup", "improve claude code setup", "audit hooks", "are my hooks wired", "dead hooks", "claude-code-setup", "automation recommendations", "what hooks am i missing", "hook hygiene"]
disable-model-invocation: true
---

# Optimize Setup — Repo-Native, Grounded

The official `claude-code-setup` plugin's `claude-automation-recommender` is built
for **bare** projects: it recommends generic hooks/MCP/subagents from a fixed
catalog. This repo is already configured far beyond that catalog (32+ hooks, 9+
agents, 5 MCP servers, full CI + pre-commit). Running the generic recommender here
produces false positives — it suggests things already present, often more
sophisticated.

This skill does what the plugin can't: **ground every recommendation in this
repo's real config + the official hook contract, then surface only genuine,
verified, highest-EV / smallest-diff changes.**

## Non-negotiables (why this skill exists)

1. **Official docs first.** Any claim about a hook event's behavior (which events
   inject context, valid event keys, stdin/stdout contract) MUST be verified
   against `https://code.claude.com/docs/en/hooks` — NOT from memory or filename
   intuition. The `post-compact-reinject` incident (2026-05-31) proved this:
   `PreCompact` *cannot* inject context (stdout discarded); post-compaction
   recovery only works via `SessionStart` `matcher:"compact"`. Filename said
   "post-compact"; the platform delivers it through SessionStart.
2. **Repo grounding before code.** Read the actual `settings.json`, `.mcp.json`,
   `.claude/agents/`, `.githooks/pre-commit`, and CI before recommending. Never
   propose adding what pre-commit/CI already enforce (e.g. ruff auto-format is in
   pre-commit step 2/8 — an edit-time format hook is redundant).
3. **Deletion is ~0 EV.** An on-disk-but-unwired hook already costs nothing at
   runtime. Deleting it saves nothing and loses intentional code. Report dormant
   hooks; do NOT delete them unless the user explicitly asks.
4. **Smallest diff wins.** Re-wiring a built-but-misfiled capability (a few lines
   in settings.json) beats authoring new automations.

## Audit procedure

### Phase 1 — Inventory (read-only, no claims yet)

```bash
# Hooks on disk
ls .claude/hooks/*.py | xargs -n1 basename | sort

# Hooks actually WIRED (referenced under a real event in settings)
python -c "
import json,re
d=json.load(open('.claude/settings.json'))
refs=set()
def walk(o):
    if isinstance(o,dict):
        [walk(v) for v in o.values()]
    elif isinstance(o,list):
        [walk(v) for v in o]
    elif isinstance(o,str):
        refs.update(re.findall(r'hooks/([A-Za-z0-9_\-]+\.py)', o))
walk(d.get('hooks',{}))
print('\n'.join(sorted(refs)))
"

# Other surfaces
python -c "import json; print(list(json.load(open('.mcp.json')).get('mcpServers',{})))"
ls .claude/agents/*.md | xargs -n1 basename
```

### Phase 2 — Resolve each on-disk-but-unwired hook (NO false kills)

For every hook on disk but not in the settings `hooks` block, classify before
judging:

- **`_`-prefixed** → shared module. Check who imports it:
  `grep -rl "import <stem>" .claude/hooks/`. If imported by a live hook → **KEEP**
  (load-bearing), not dead.
- **Imported by another hook** (e.g. `mcp-git-guard` ← `branch-flip-guard`) →
  **KEEP**.
- **Referenced in `settings.local.json`** → wired, just locally. Check it.
- **Truly unreferenced** → read its docstring + `git log -S "<name>" -- .claude/settings.json`
  to learn whether it was *removed* (and why) or *never wired*. A removal commit
  often names the real reason (invalid event key, superseded by another hook).

### Phase 3 — For any "should we wire this?" candidate: verify the event contract

Before adding ANY hook to settings, confirm against the official docs:
- Is the event key **valid and spelled exactly** right? (`PreCompact`, not
  `PostCompact`; `SessionStart`, etc.)
- Does that event **deliver what the hook needs**? Context-injecting events are
  **`UserPromptSubmit`, `UserPromptExpansion`, `SessionStart`** (stdout auto-added
  as context; or `hookSpecificOutput.additionalContext`). Pre-action events
  (`PreToolUse`, `PreCompact`) can only **block** via `decision:"block"` — their
  stdout is debug-log only.
- Windows emit footgun: a hook invoked directly by Claude Code has **cp1252**
  stdout on Windows → `UnicodeEncodeError` on arrows/em-dashes. Require
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (try/except) at
  the top, OR a `PYTHONIOENCODING=utf-8` wrapper. Smoke-test before wiring.

### Phase 4 — Report (only verified, repo-specific findings)

Output a table: each candidate, verdict (KEEP load-bearing / dormant-leave /
wire-this / redundant-with-pre-commit-or-CI), evidence, and diff size. Surface
**at most the 1-2 highest-EV** actions. End by asking for explicit approval before
editing `settings.json` (capital-adjacent repo → present first).

### Phase 5 — Apply (only on approval) + verify

After an approved edit:
```bash
python -c "import json; json.load(open('.claude/settings.json'))"   # JSON valid
# smoke-test the hook with a realistic stdin payload for its event + edge cases
echo '{"hook_event_name":"<Event>","source":"<src>"}' | python .claude/hooks/<hook>.py; echo "exit:$?"
uv run ruff check .claude/hooks/<hook>.py
python pipeline/check_drift.py    # canonical guardrail must stay 0-fail
```

## Anti-patterns (do NOT do)

- ❌ Recommending an auto-format / lint hook — pre-commit + CI already do ruff.
- ❌ Recommending context7 / GitHub / DB MCP — already in `.mcp.json`.
- ❌ Recommending code-reviewer / security subagents — already in `.claude/agents/`.
- ❌ Deleting unwired hooks to "clean up" — zero runtime cost, loses code.
- ❌ Wiring a hook by filename intuition without checking the event contract.
- ❌ Claiming a hook is fail-open without confirming the emit/print is inside a
  guard (an un-encodable char outside try/except still crashes).

## Reference

- Official hook docs: https://code.claude.com/docs/en/hooks
- Generic plugin (for contrast): `claude-code-setup` →
  `skills/claude-automation-recommender/SKILL.md`
- Incident that motivated this skill: `post-compact-reinject` wired correctly via
  `SessionStart matcher:"compact"` after #253 wrongly used invalid `PostCompact`
  key (2026-05-31).
