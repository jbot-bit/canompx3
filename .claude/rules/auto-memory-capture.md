# Auto-Memory-Capture — the repo learns without being asked

**Load-policy:** referenced from CLAUDE.md § Memory discipline. Read on demand
when editing the three hook files or the shared helper below.

**Authority:** closes the make-it-automatic gap from
`memory/project_next_auto_memory_capture_on_clear.md` — durable, non-obvious
session findings were lost on `/clear` because writing memory is manual and easy
to forget. This is the same treatment as plan-rigor v2: a hook *reminds Claude to
judge* whether to capture, without ever auto-writing memory files or
prose-guessing salience.

It does NOT auto-write memory. It NEVER parses transcript prose for "importance."
It gates on **factual git state only** — deterministic, fail-open, unfoolable by
narrative.

---

## The 3-event loop

```
SessionEnd(any reason w/ signal)  → write breadcrumb to .claude/hooks/state/
                                     + always append one telemetry line
SessionStart(next session, incl. the /clear that just ended the prior one)
                                  → read breadcrumb → emit additionalContext
                                     (CLAUDE-VISIBLE cue to JUDGE) → mark consumed
PreCompact(manual|auto)           → systemMessage (operator-visible warning)
```

Three files, one shared helper:

| File | Event(s) | Channel | Audience |
|---|---|---|---|
| `memory-capture-advisory.py` | PreCompact | `systemMessage` | **user** (terminal) |
| `memory-capture-advisory.py` | SessionEnd | (none) | state-writer only |
| `memory-capture-sessionstart.py` | SessionStart | `additionalContext` | **Claude** |
| `_memory_capture.py` | — | — | shared signal/breadcrumb logic |

### Why two channels (the load-bearing fact, verified against official docs)

- **`systemMessage` = USER-ONLY**, all events. It never reaches Claude.
- **`additionalContext` = CLAUDE-VISIBLE**, but only SessionStart (among our
  events) supports it. **PreCompact, SessionEnd, and Stop do NOT.**

So the *model-facing* goal (the repo actually learning) MUST run through
SessionStart `additionalContext`, not PreCompact. PreCompact's `systemMessage` is
a bonus operator-visible warning at the compaction boundary.

### Why it survives `/clear` (empirically proven, not inferred)

`/clear` is a hard context reset; `/compact` summarizes-and-continues. The whole
design rests on SessionStart `additionalContext` crossing the `/clear` boundary.
This was **proven by a Stage-0 logging probe** (2026-05-31): a uniquely-tagged
marker emitted on `SessionStart(source=clear)` was read in the freshly-cleared
session (seq counter advanced 1→2 across a real `/clear`, and the cleared session
reported reading seq=2). Probe log: `.claude/hooks/state/memory-capture-probe.log`
(probe itself removed after the window — never committed). No inferred fact ships.

---

## The signal threshold (factual git state only)

`signal_meets_threshold(sig)` is true iff ANY of:
- `commits >= 1` — commits made this session (`head_at_start..HEAD`, fallback
  `origin/main..HEAD`).
- `files >= 3` — tracked files changed (`git diff --name-only HEAD`).
- any `docs/runtime/stages/*.md` changed.
- any `.claude/rules/*` or `*doctrine*.md` / `*rule*.md` changed.

**Untracked (`??`) files do NOT count** — `git diff HEAD` shows tracked changes
only. This is intentional: brand-new unstaged files are often scratch/experiment;
durable work is staged or committed, and the commit-count signal catches the
committed case independently.

Below threshold → **silent**. A no-capture session is a VALID, no-bias outcome —
the cue explicitly says *"capture nothing if obvious/transient/already
documented."* This is a JUDGE prompt, never an instruction to write, and never a
nag (one-shot per breadcrumb, 24 h expiry).

---

## Lifecycle

- **SessionEnd** (signal met) writes `memory-capture-pending.json`
  (`consumed:false`) and always appends `memory-capture.log` (JSONL telemetry).
- **SessionStart** reads the breadcrumb; emits the cue iff present AND
  `not consumed` AND age < 24 h AND counts nonzero; then flips `consumed:true`
  (one-shot — fires at most once per breadcrumb).
- **PreCompact** warns once per session (dedup via `memory-capture-advisory.json`,
  capped 200 sessions).

State lives under `.claude/hooks/state/` (gitignored contents). The SessionStart
cue reads ONLY its breadcrumb — no `.claude.pid` dependency, so it is immune to
session-lock format changes and races with `session-start.py`. The two
SessionStart hooks run in **parallel** (different channels: this hook emits
stdout-JSON `additionalContext`; `session-start.py` prints stderr text) and never
touch each other.

---

## Fail-open guarantee

Every hook wraps `main()` in `try/except BaseException: sys.exit(0)`. Malformed/
empty stdin, missing git, git timeout, IO errors → exit 0, no stdout disturbance.
stdout on exit 0 carries ONLY valid JSON (when emitting) or nothing (when silent),
per the hooks contract.

---

## Verifying it is wired, not dead

Hook `command`s point at the **main checkout** (`C:/Users/joshd/canompx3/.claude/
hooks/…`), matching every existing hook — so a worktree session fires MAIN's
copy. These files therefore only fire **after they land in main on merge**
(`feedback_worktree_hooks_resolve_to_main_checkout_path_2026_05_31.md`). To
verify post-merge:
- `/hooks` menu — shows PreCompact / SessionEnd / SessionStart(×2) registered.
- `claude --debug` — logs hook firing. No restart needed (settings.json is
  file-watched).
- Live-fire smoke (any checkout): pipe a payload to each hook, e.g.
  `echo '{"hook_event_name":"SessionStart","source":"clear"}' | python
  .claude/hooks/memory-capture-sessionstart.py`.

---

## Related

- `memory/project_next_auto_memory_capture_on_clear.md` — the seed/goal.
- `memory/feedback_worktree_hooks_resolve_to_main_checkout_path_2026_05_31.md` —
  the wired-but-dead caveat.
- `.claude/hooks/session-start.py` — the parallel SessionStart sibling.
- `.claude/hooks/completion-notify.py` — the emit/state pattern templated here.
