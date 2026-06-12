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

## Baton staleness detection (4th check — closes the integrate-then-/clear gap)

The capture cue tells Claude to write NEW memory; it never re-checks EXISTING
batons. So a resume-baton could keep claiming `<sha> NOT on origin/main` long
after the work merged (n=1: this session's `f1fd7a90`/`6dadde5b` C11 baton). The
SessionStart hook now also runs two drift cues every start (independent of the
breadcrumb), both fail-open, emitted on the same `additionalContext` channel.

Grounded against official docs (`code.claude.com/docs/en/hooks`): SessionStart is
the sanctioned place for this — it **re-runs on resume** specifically so hooks can
refresh stale SHAs; multiple hooks' `additionalContext` is **concatenated**; cap
is 10 000 chars (our cue is < 1 KB). No built-in staleness mechanism exists, and
the leading memory plugin (claude-mem) is retrieval-only with no git validation —
so tier 1 is a deliberate, more-rigorous specialization for this git-native repo,
not a reinvented wheel.

- **Tier 1 — PROVEN (git-falsifiable).** `scan_stale_batons()`: a baton line says
  a SHA is *not merged / unmerged / local-only / verify-it-landed*, yet
  `git merge-base --is-ancestor <sha> origin/main` proves it IS on origin/main.
  The NOT-merged phrase and the SHA must co-occur **on the same line** (proximity
  guard — abstract merge-mechanics prose with no adjacent SHA never fires). Fires
  only on the git-proven contradiction; a genuinely-unmerged or unknown/GC'd SHA
  stays silent. This is a factual claim, phrased as such in the cue.
- **Tier 2 — ADVISORY (heuristic, not provable).** `scan_live_project_batons()`:
  recent (`mtime ≤ 72 h`) `metadata.type: project` batons whose `description:`
  asserts live status (`NEXT=/RESUME/OPEN/pending/…`). Scoped on three
  independent axes (type + recency + status language) to cut the ~500-file corpus
  to a handful; capped at 8. A *confirm-still-current* JUDGE nudge — never an
  assertion of staleness (it can't be proved, so it mustn't claim it). This
  mirrors claude-mem's "surface and let Claude decide" norm.

`MEMORY_DIR` is resolved HOME-based (`~/.claude/projects/<slug>/memory`, slug
derived from PROJECT_ROOT — not hardcoded), with a legacy repo-local fallback. A
live smoke test caught a hardcoded-repo-path bug here that 10 patched unit tests
missed → regression test `test_memory_dir_resolves_home_based_not_repo_based`
exercises the REAL resolver. Tests: `tests/test_hooks/test_baton_staleness.py`.

---

## MEMORY.md size guard + the HOT/COLD contract (5th SessionStart check)

**Authority:** 2026-06-12 — `MEMORY.md` silently truncated. The Claude Code
loader caps the always-loaded index at **~24.4KB** (tokenizer overhead lands the
real byte-cut ~1.7KB before a raw-byte cut). When the file crossed it, the
trailing lines — live deployed-lane state AND an open RESUME baton — **fell out
of context every session with no error.** Silent + positional: the worst failure
class, because the model can't know what it never saw.

This is the official-memory-tool / claude-mem / MemGPT-Letta pattern, which this
repo had **half-built**: a small always-loaded HOT index you retrieve *past*,
backed by a searchable COLD store. The cold store already exists (637 topic files
+ `recall` / `pinecone-assistant` / `ls memory/`). The index was just mis-shaped —
carrying *content* (single lines ran 1500-1900 chars) instead of *pointers*. The
fix finishes the pattern; it does **not** add new infra.

### The HOT/COLD contract (maintain automatically)

- **HOT = `MEMORY.md`** — the always-loaded index. Holds ONLY:
  - **`## Open batons`** — live/owed work (the `### Active live state` block stays
    pinned at the top so it can never be the line that truncates).
  - **`### Standing doctrine` + rigor/integrity/hygiene rollups** — always-on.
  - Every line is a **≤200-char pointer** (`- [topic](file.md) — ≤12-word hook`).
    Multi-link doctrine rollups (`→ [a] / [b] / [c]`) may exceed 200 chars
    *only* because they bundle several pointers — never because one line stores
    content. If a single finding needs >200 chars, the content belongs in its
    topic file, not the index.
- **COLD = `MEMORY_ARCHIVE.md` + the 637 topic files** — searchable, never loaded
  wholesale. "Demoted" ≠ "forgotten": cold tier is reachable via
  `recall` / `pinecone-assistant` / `ls memory/` on demand.
- **Demotion trigger:** a baton that closes (`✅ DONE`, `✅ PUSHED`, `ARCHIVABLE`)
  is demoted to `MEMORY_ARCHIVE.md` **the turn it closes**, leaving a 1-line
  pointer in HOT *only if* follow-up is still owed (re-open condition, owed Stage,
  etc.). Golden nuggets stay as permanent 1-line pointers.

### The guard (advisory cue, never auto-write)

`check_memory_index_size()` (in `_memory_capture.py`) reads `MEMORY.md` by **raw
bytes** (`read_bytes()` — the cap is byte-based, and Windows CRLF translation
would undercount a char-count → false-PASS). It fires when bytes ≥
`_MEMORY_WARN_BYTES` (21KB) OR lines ≥ `_MEMORY_WARN_LINES` (180) — both **below**
the ~24.4KB hard cap, so the cue lands while there is still headroom to act. A
post-truncation warning is useless: the content is already gone.

`memory-capture-sessionstart.py` emits the cue on the same `additionalContext`
channel as the baton-drift cues (concatenated, runs every start). Like every
other check it **only cues Claude to JUDGE** what to demote — it never auto-writes
or auto-demotes (the load-bearing doctrine of this whole system). Fail-open: any
read error → `None` → silent.

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
