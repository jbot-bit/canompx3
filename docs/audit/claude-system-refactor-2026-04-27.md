# Claude Code System Audit — 2026-04-27

**Branch:** `audit/claude-system-refactor-20260427`
**Mode:** DESIGN + AUDIT (no implementation)
**Scope:** `.claude/settings.json`, `.claude/hooks/*`, `.claude/rules/*`, `CLAUDE.md`, hook→prod-script call graph.
**Goal:** Cut tokens / hook overhead / duplicated logic / iteration latency without weakening institutional rigor, holdout discipline, source-of-truth chain, or audit guarantees.

Every claim below is grounded in a file path + line number. Items I could not verify with the reads I did are marked **UNSUPPORTED** and explicitly flagged not-actionable until verified.

---

## PASS 0 — System Map

### Hooks wired in `.claude/settings.json`

| # | Event | Matcher | Script | Timeout | Lines | Output channel | Token cost? |
|---|---|---|---|---|---|---|---|
| 1 | PreToolUse | `Edit\|Write` | `.claude/hooks/stage-gate-guard.py` | 5s | 421 | stderr (block msg) | Only when blocking (rare) |
| 2 | PreToolUse | `Edit\|Write` | `.claude/hooks/pre-edit-guard.py` | 5s | 32 | stderr (block msg) | Only on `gold.db` / `.env` (rare) |
| 3 | PreToolUse | `Read\|Bash` | `.claude/hooks/data-first-guard.py` | 3s | 377 | stderr (warn/block) | Counter-driven; mostly silent |
| 4 | PostToolUse | `Edit\|Write` | `.claude/hooks/post-edit-pipeline.py` | 90s | 239 | stderr (drift+test failures) | **HEAVY when failing** |
| 5 | PostToolUse | `Edit\|Write` | `.claude/hooks/post-edit-schema.py` | 30s | 98 | stderr (test+stale-SQL warnings) | Only on `init_db/db_manager/schema` |
| 6 | UserPromptSubmit | `""` | `.claude/hooks/stage-awareness.py` | 3s | 190 | stderr | **EVERY PROMPT** — 1-3 lines |
| 7 | UserPromptSubmit | `""` | `scripts/tools/bias_grounding_guard.py` | 3s | 91 | stderr (1 line) | Cooldown 20m, selective regex |
| 8 | UserPromptSubmit | `""` | `.claude/hooks/data-first-guard.py` | 3s | 377 (same as #3) | stderr | **EVERY PROMPT** — directive routing |
| 9 | UserPromptSubmit | `""` | `.claude/hooks/risk-tier-guard.py` | 3s | 119 | JSON `additionalContext` | Cooldown 20m + tier-change |
| 10 | Notification | `""` | `.claude/hooks/completion-notify.py` | 3s | 38 | beep | Zero (sound only) |
| 11 | Stop | `""` | `.claude/hooks/completion-notify.py` | 3s | (same script) | beep | Zero |
| 12 | PostCompact | `""` | `.claude/hooks/post-compact-reinject.py` | 5s | 78 | **stdout** (re-injected to context) | **HEAVY** on every compaction |
| 13 | SessionStart | `""` | `.claude/hooks/session-start.py` | 5s | 631 | stderr | **HEAVY** at startup (multi-line brief) |

**Settings of note** (`.claude/settings.json`):
- L19 `effortLevel: medium`
- L20 `alwaysThinkingEnabled: false` ✅ already off
- L4 `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE: 70` (compacts at 70% — aggressive; raises PostCompact frequency)
- L184-194 enabled plugins: `context7`, `commit-commands`, `explanatory-output-style`, `firecrawl` (each contributes its own commands/skills/agents to the listing)

### Orphaned script (NOT wired)

- **`.claude/hooks/bias-grounding-guard.py` (90 lines)** exists on disk but `.claude/settings.json:108-109` wires the **`scripts/tools/bias_grounding_guard.py`** copy (91 lines). Two near-identical files with the same `COMPACT_DIRECTIVE`. One is dead code.

### Files / docs that consume tokens at session level

- `CLAUDE.md` (136 lines) — auto-loaded every session. Already terse.
- `~/.claude/CLAUDE.md` (user global) — loaded too.
- `.claude/rules/auto-skill-routing.md`, `.claude/rules/workflow-preferences.md`, `.claude/rules/parallel-session-isolation.md` — included in initial system reminder per the SessionStart context I received this turn (all three appeared as `Contents of …`).
- `memory/MEMORY.md` (~150 lines per the policy header; truncates after 200).
- 22 files in `.claude/rules/` totalling **3,899 lines** — but `CLAUDE.md` only references a subset by path; the rest are loaded only when their owning hook/script reads them OR when CLAUDE.md says "auto-loads on X edit" (e.g. `backtesting-methodology.md`, `institutional-rigor.md`). I did **not** verify the autoload mechanism exists in any hook — flagging as **UNSUPPORTED** below.

### Skills / agents / commands

- 8 agents under `.claude/agents/` (~1,500 lines total) — only loaded when `Agent` tool is invoked; zero idle cost.
- 20 skills under `.claude/skills/` — only loaded when `Skill` tool fires.
- `.claude/commands` is a **regular file (zero bytes)**, not a directory — likely vestigial. Worth deleting.

---

## PASS 1 — Redundancy & Overhead Findings

### F1 — `data-first-guard.py` is wired to TWO events (verified)

`.claude/settings.json:65` (PreToolUse `Read|Bash`) AND `.claude/settings.json:119` (UserPromptSubmit `""`). Same 377-line script, dispatches by `hook_event_name` inside `main()` (`data-first-guard.py:368`). Not a bug — it's intentional dual-mode.

**But:** every user prompt pays the python-startup cost twice if the user immediately triggers a Read (once for prompt-routing, once for read-counting). On Windows this is ~150-300 ms × 2.

**Classification:** MERGE conceptually (single hook with `hook_event_name` switch is fine, but the script could short-circuit faster on UserPromptSubmit when no keywords match — it already does via `_should_emit_directives` cooldown, so the actual cost is mostly python startup, not work).

**Action:** KEEP, but re-evaluate after F2 fixes — UserPromptSubmit gets 4 hook firings; combining can reduce 4 python startups → 1.

### F2 — UserPromptSubmit fires 4 separate Python processes per prompt (verified)

`.claude/settings.json:93-133` registers four separate UserPromptSubmit hook entries:
1. `stage-awareness.py` (always emits 1-3 stderr lines if any stage exists)
2. `scripts/tools/bias_grounding_guard.py` (cooldown'd, selective)
3. `data-first-guard.py` (cooldown'd, selective)
4. `risk-tier-guard.py` (cooldown'd, selective; JSON additionalContext)

**Cost:**
- 4× python interpreter startup per prompt (Windows: ~600 ms-1.2 s combined wall clock).
- 4× JSON parse of the prompt event.
- `stage-awareness.py:118-120,173,181` emits stderr **every prompt** when a stage exists (no cooldown). Confirmed in this very session — the SessionStart context I received included `RISK TIER: high. Default to normal reasoning…` from risk-tier-guard.

**Token contribution per prompt** (rough — needs measurement):
- stage-awareness: ~50-150 tokens (always)
- risk-tier: ~30 tokens (only when fired ≈ 1× per 20 min)
- bias-grounding: ~25 tokens (only when fired)
- data-first: ~10-150 tokens (cooldown'd; bursts when matched)

**Classification:** **MERGE** — single Python entrypoint with all four logics inside, runs once. ~75% startup-cost reduction. Same correctness.

### F3 — Two PreToolUse `Edit|Write` entries that could be one matcher (verified)

`.claude/settings.json:39-58` registers stage-gate-guard and pre-edit-guard as **separate hook objects** with the same matcher. Functionally identical to one entry with two `hooks:[…]` items. Cosmetic, but the `--debug` Claude Code log gets noisier.

**Classification:** MERGE (cosmetic; ~0 token win, slight ergonomic win).

### F4 — Two PostToolUse `Edit|Write` entries (verified)

`.claude/settings.json:71-91` — same shape as F3. `post-edit-pipeline.py` (90s) + `post-edit-schema.py` (30s). They run sequentially per Claude Code's spec, so this is fine.

**Classification:** MERGE matcher block (cosmetic).

### F5 — `post-edit-pipeline.py` runs on EVERY pipeline/trading_app .py edit (verified)

`post-edit-pipeline.py:100` filters `if not (("pipeline" in file_path or "trading_app" in file_path) and file_path.endswith(".py"))`. Then:
- Phase 1: drift check `--fast` (≤ 30 s), debounced 30 s via `.last_drift_ok` (line 11-12).
- Phase 2: targeted pytest from `TEST_MAP` (line 46-77, 30 entries), 45 s timeout.
- Phase 3: `audit_behavioral.py`, 15 s timeout.

This is the **single biggest latency contributor** during multi-file refactors. Every 31st second invalidates the debounce, so a 5-file edit burst can run 1 drift + 5 pytests + 5 behavioral audits = **5–10 minutes** of hook latency that the user sits through.

**Token contribution:** `post-edit-pipeline.py:142-174` already truncates drift output to FAILED blocks only — good. But `result.stdout` on test failures is dumped raw (line 209) — can be hundreds of lines.

**Classification:**
- KEEP drift check post-edit (canonical safety).
- **Move targeted pytest + behavioral audit to PRE-COMMIT or a `Stop` hook** (run once at logical end, not after every edit). The pre-commit hook (`.githooks/pre-commit` per `CLAUDE.md` Guardrails section) is the natural home — I did not verify what it currently runs (**UNSUPPORTED** — needs `cat .githooks/pre-commit`).
- Cap raw test stdout to last ~60 lines + final summary.

### F6 — `session-start.py` is 631 lines and runs 5 git/uv/gh subprocesses every startup (verified)

`session-start.py:619-624` calls in sequence:
1. `_origin_drift_lines` — `git fetch` + ff-pull (10 s timeout, line 148)
2. `_env_drift_lines` — `uv sync --frozen --check` (10 s timeout, line 217-224)
3. `_action_queue_ready_lines` — imports `pipeline.work_queue.load_queue`
4. `_parallel_session_lines` — `git worktree list` + status of each (line 519-555)
5. `_main_ci_status_lines` — `gh run list` (5-min cache, line 268)

Plus the brief built by `scripts/tools/claude_superpower_brief.build_brief` (not read; **UNSUPPORTED** size claim).

The 5 s `timeout` in `settings.json:178` is **less than several internal subprocess timeouts** (`git fetch` is 10 s on line 148; `uv sync` is 10 s on line 222) — these can race. If the harness kills the hook at 5 s, the user sees no SessionStart context. **VERIFIED BY CODE** — needs runtime test to confirm whether this actually fires.

**Classification:**
- BUG: bump `settings.json:178` SessionStart timeout to **30 s** to match internal budget.
- KEEP all 5 subprocesses (each protects a real failure mode per inline rationale).
- TOKEN: token cost is from the `lines` list joined and printed (line 625). If `claude_superpower_brief` is fat, that's the win — needs measurement.

### F7 — `completion-notify.py` wired to BOTH `Notification` and `Stop` (verified)

`.claude/settings.json:135-158`. Script (`completion-notify.py:21`) gates on `if hook_event not in ("Notification", "Stop")`. Both events trigger a beep — likely double-beeps on some flows. Zero token cost; pure UX nit.

**Classification:** KEEP (sound is desired). Optional: deduplicate with a state file (last beep timestamp; skip if <2 s ago).

### F8 — Orphaned `.claude/hooks/bias-grounding-guard.py` (verified)

90-line script not referenced by `settings.json`. The wired one is `scripts/tools/bias_grounding_guard.py` (91 lines). Maintenance hazard — someone editing the orphan thinks they're tuning the live behavior.

**Classification:** DELETE the orphan (`.claude/hooks/bias-grounding-guard.py`).

### F9 — Empty `.claude/commands` file (verified)

`ls -la .claude/` shows `-r--r--r-- 1 joshd 197609 0 Mar 16 16:22 commands`. Zero-byte file where a directory probably belonged. Plugins now provide commands. Vestigial.

**Classification:** DELETE.

### F10 — Stage state files: dual `STAGE_STATE.md` + `stages/*.md` (verified)

`stage-awareness.py:107-115` and `stage-gate-guard.py:28-31` both read both. Comments say `STAGE_STATE.md` is "legacy, still read for backwards compat." Both hooks parse on every fire.

**Classification:** NEEDS REVIEW — confirm with user whether legacy `STAGE_STATE.md` can finally be retired, eliminating one parse path per hook fire.

### F11 — `data-first-guard.py` regex catalog overlaps with `risk-tier-guard.py` regex (verified)

- `data-first-guard.py:29` INVESTIGATION_KEYWORDS, :42 TRADING_QUERY, :54 SESSION_TIME, :97 DESIGN, :106 IMPLEMENT, :115 COMMIT, :123 RESEARCH, :132 ORIENT, :140 RESUME — **9 regex tables**.
- `risk-tier-guard.py:24` CRITICAL_RE, :34 HIGH_RE — keyword sets like `pipeline|check_drift|backtest|holdout|p.?value|fdr` overlap with `data-first-guard`'s RESEARCH_KEYWORDS.
- `bias_grounding_guard.py:24` TARGET_PROMPTS likewise overlaps.

Each regex compiles cheaply in Python, but conceptually the same prompt triggers 3 separate classifiers in 3 separate processes.

**Classification:** MERGE into single intent classifier (composes with F2). One regex pass → emit all relevant directives.

### F12 — `CLAUDE.md` references rules files but the autoload mechanism is **UNSUPPORTED**

`CLAUDE.md` says `.claude/rules/backtesting-methodology.md` and `.claude/rules/institutional-rigor.md` "auto-load on X edits". I did **not** find the auto-loader hook. Possibilities:
- Implemented in `claude_superpower_brief.build_brief` (not read).
- Implemented in a plugin.
- Documented intent that was never wired.

**Marked UNSUPPORTED.** If unwired, the prose in `CLAUDE.md` is a lie that confuses future-you. Action: verify by grepping for filenames in hook scripts.

---

## PASS 2 — Token Cost vs Local Cost

The user asked to separate true token cost from local CPU cost. Hooks emit on **stderr** (which Claude reads as additional context) and **stdout** with `hookSpecificOutput.additionalContext` JSON (also context). Both bill tokens. Wall-clock latency is local cost only.

| Source | Channel | Trigger frequency | Per-fire tokens (est.) | Daily token cost (high estimate) |
|---|---|---|---|---|
| `CLAUDE.md` | system prompt | every session | ~1,200 | low (cached) |
| `MEMORY.md` index | system prompt | every session | ~2,500 | low (cached) |
| `session-start.py` brief | stderr | every session | ~300-1,500 (UNSUPPORTED) | medium |
| `post-compact-reinject.py` | **stdout (re-injected)** | every compaction (~hourly at 70% threshold) | ~200-800 | medium |
| `stage-awareness.py` | stderr | every prompt | ~30-150 | **HIGH (every turn)** |
| `risk-tier-guard.py` | JSON additionalContext | ~1× per 20 min | ~30 | low |
| `data-first-guard.py` directives | stderr | bursts on routing matches | ~25-150 | low (cooldown'd) |
| `bias_grounding_guard.py` | stderr | ~1× per 20 min | ~25 | low |
| `post-edit-pipeline.py` failures | stderr | per failed edit | ~200-2,000 (raw test stdout) | spiky |
| `post-edit-pipeline.py` success | (silent) | – | 0 | – |
| Auto-loaded rule files (UNSUPPORTED) | unknown | unknown | unknown | unknown |

**Local-only (zero token) cost:**
- `git fetch`, `git status`, `gh run list`, `uv sync --check`: latency in `session-start.py`.
- `check_drift.py --fast`: 3-5 s latency in `post-edit-pipeline.py`.
- `pytest`: 5-30 s latency in `post-edit-pipeline.py`.
- All four UserPromptSubmit hooks: ~600 ms-1.2 s aggregate Python startup.

**Top 3 token wins (in priority order):**
1. **`stage-awareness.py` always-on emission** — add a per-stage-content-hash cooldown so identical state doesn't reprint on every prompt. Today it prints 1-3 lines on every turn even when nothing changed.
2. **`post-edit-pipeline.py` raw stdout dump** — line 209 `print(result.stdout, file=sys.stderr)` cap at last 80 lines + summary.
3. **`post-compact-reinject.py`** — verify `build_brief` size; if >500 tokens, slim to 1-page essentials.

**Top 3 latency wins:**
1. Move `pytest` + `audit_behavioral.py` out of `post-edit-pipeline.py` Phase 2/3 → `Stop` hook or pre-commit.
2. Merge 4 UserPromptSubmit hooks into 1 process.
3. Bump SessionStart timeout from 5 s to 30 s (currently truncates internal subprocess budgets).

---

## PASS 3 — Proposed Architecture

### A. Hook tier model (target)

| Tier | When | What runs | Tokens? | Latency budget |
|---|---|---|---|---|
| **edit** | PostToolUse Edit\|Write on `pipeline/`/`trading_app/` | drift `--fast` only | only on FAIL (truncated) | 5 s |
| **prompt** | UserPromptSubmit | single merged classifier (intent + risk + bias + data-first) | 0-150 tokens, cooldown'd | 300 ms |
| **commit** | `.githooks/pre-commit` (already exists) | pytest target file + behavioral audit + full drift | 0 tokens (local only) | 60 s |
| **push** | `.githooks/pre-push` (verify exists; **UNSUPPORTED**) | full pytest + drift `--full` | 0 tokens | 300 s |
| **session** | SessionStart | brief + 5 git/uv/gh checks | 300-800 tokens | 30 s (raise from 5 s) |
| **compact** | PostCompact | re-inject ≤ 1 page | ≤ 500 tokens | 5 s |
| **deploy/research** | manual (`/capital-review`, `/research`) | heavy audits | as needed | as needed |

### B. Command layer migrations

Move from hook-on-every-edit → on-demand:
- `audit_behavioral.py` → `/verify` skill (already exists per skill listing).
- Targeted pytest → pre-commit + manual `/verify quick`.
- Stale-SQL skill scan from `post-edit-schema.py` → keep (it's already only on schema files, ~100 lines, fast).

### C. Context control (what loads each session)

KEEP loaded every session:
- `CLAUDE.md`, `~/.claude/CLAUDE.md`, `MEMORY.md` index.
- `.claude/rules/auto-skill-routing.md`, `workflow-preferences.md`, `parallel-session-isolation.md` (small, behavioral).

LOAD ON DEMAND only:
- `.claude/rules/backtesting-methodology.md` (321 lines) — only when editing `research/` or `trading_app/strategy_*`.
- `.claude/rules/institutional-rigor.md` (94 lines) — only when editing production code.
- `.claude/rules/research-truth-protocol.md` (203 lines) — only on research prompts.
- `.claude/rules/quant-audit-protocol.md` (279 lines) — only on `/code-review` / `/capital-review` invocation.
- The other 18 rules files — only when their owning script reads them.

**Verification needed:** is on-demand loading already wired (per CLAUDE.md prose) or aspirational? Fix or remove the prose accordingly.

### D. Safety guarantees (what survives the cut)

| Guarantee | Today's enforcement | Post-refactor enforcement | Preserved? |
|---|---|---|---|
| 2026 holdout sacred | `RESEARCH_RULES.md` + drift checks | unchanged | ✅ |
| Source-of-truth chain | `CLAUDE.md` + `quant-audit-protocol.md` | unchanged (rule load is only the discovery mechanism, not the rule itself) | ✅ |
| Stage-gate before prod edits | `stage-gate-guard.py` PreToolUse | unchanged | ✅ |
| Drift check after prod edits | `post-edit-pipeline.py` Phase 1 | unchanged | ✅ |
| Targeted tests | `post-edit-pipeline.py` Phase 2 | **MOVED to pre-commit** | ✅ (still runs before any commit) |
| Behavioral audit | `post-edit-pipeline.py` Phase 3 | **MOVED to pre-commit** | ✅ |
| Stage awareness | `stage-awareness.py` every prompt | **content-hash cooldown** | ✅ (only re-emits on change) |
| Risk-tier hint | `risk-tier-guard.py` selective | unchanged (already cooldown'd) | ✅ |
| Bias grounding | `bias_grounding_guard.py` selective | unchanged | ✅ |
| Data-first | `data-first-guard.py` PreToolUse counters | unchanged | ✅ |
| `gold.db` write block | `pre-edit-guard.py` | unchanged | ✅ |
| Worktree mutex | `session-start.py` `_session_lock_lines` | unchanged | ✅ |
| Main-CI awareness | `session-start.py` `_main_ci_status_lines` | unchanged | ✅ |

---

## PASS 4 — Risk Analysis (per change)

### Change 1: Merge 4 UserPromptSubmit hooks → 1 process
- **Failure protected today:** prompt-time directive injection.
- **New failure mode:** if the merged script crashes early, all 4 directive families fail at once instead of 1.
- **Mitigation:** wrap each subroutine in `try/except` + `sys.exit(0)` (current scripts already do).
- **Verdict:** SAFE.

### Change 2: Move pytest/behavioral audit from PostToolUse to pre-commit
- **Failure protected today:** broken code mid-session is loud (red stderr after edit).
- **New failure mode:** developer iterates 10 broken edits silently; pre-commit fails at end. **Cost:** lost iteration time, surprise at commit.
- **Mitigation:** drift check stays post-edit (catches the most common silent breakage); pre-commit is mandatory and cannot be `--no-verify`'d (per memory).
- **Verdict:** ACCEPTABLE for senior engineer (user fits profile per memory). Verify pre-commit currently runs the tests; if not, add them as part of this change.

### Change 3: Add content-hash cooldown to `stage-awareness.py`
- **Failure protected today:** Claude is reminded of stage every prompt.
- **New failure mode:** stage changes mid-session but reminder is suppressed because content hash didn't update at the right moment.
- **Mitigation:** invalidate hash on any `stages/*.md` mtime change.
- **Verdict:** SAFE.

### Change 4: Bump SessionStart timeout 5 s → 30 s
- **Failure protected today:** none (raises a real correctness issue — internal subprocess budgets exceed harness timeout).
- **New failure mode:** slow startup masks a hung subprocess.
- **Mitigation:** internal timeouts already cap each subprocess (5-15 s). Aggregate worst-case is ~25 s.
- **Verdict:** SAFE and arguably a bug fix.

### Change 5: Truncate `post-edit-pipeline.py` raw test stdout
- **Failure protected today:** full pytest dump shows long traces.
- **New failure mode:** truncated trace hides root cause.
- **Mitigation:** keep last 80 lines (= where pytest summarizes failures + the failing assertion). User can re-run tests manually for full output.
- **Verdict:** SAFE.

### Change 6: DELETE `.claude/hooks/bias-grounding-guard.py` orphan + zero-byte `.claude/commands`
- **Failure protected today:** none.
- **New failure mode:** none.
- **Mitigation:** N/A.
- **Verdict:** SAFE.

### Change 7: Retire `STAGE_STATE.md` legacy path
- **UNSAFE without migration verification.** NEEDS REVIEW with user — confirm no agent or external tool still writes there.

---

## Minimum Set — Recommended Implementation Order

If approved, implement in this order, each as a separate commit so each is reversible:

1. **C6** — DELETE orphan `bias-grounding-guard.py` + empty `commands` file. (5 min, zero risk)
2. **C4** — Bump SessionStart timeout to 30 s in `settings.json`. (1 min, bug fix)
3. **C5** — Truncate `post-edit-pipeline.py:209` raw stdout to last 80 lines. (10 min, low risk)
4. **C3** — Add content-hash cooldown to `stage-awareness.py`. (30 min, low risk)
5. **C1** — Merge 4 UserPromptSubmit hooks into one entry point. (1-2 h, medium risk — needs test)
6. **C2** — Move pytest + behavioral audit from `post-edit-pipeline.py` to pre-commit. (1 h, **needs user sign-off** — workflow change)
7. **C7** — Retire `STAGE_STATE.md`. (NEEDS USER REVIEW first)

**F12 `CLAUDE.md` autoload prose** — verify before recommending: grep hook scripts for `backtesting-methodology` / `institutional-rigor`. If unwired, either wire it (preferred — these rules matter) or fix the prose.

---

## Decision Standard

A prop-desk risk manager would accept this refactor IF:
- All four code changes that touch directive emission are A/B tested by running the merged hook against the current 4 hooks on the same prompt set and diffing emitted directives. → required acceptance test.
- Pre-commit definitively runs the tests + behavioral audit before C2 lands. → verify `.githooks/pre-commit` content, add what's missing, only then remove from `post-edit-pipeline.py`.
- C7 (legacy STAGE_STATE.md) is gated on explicit user confirmation — too easy to silently break Codex's stage handoff.

If those conditions hold, every guarantee in the table above is preserved with measurable token + latency reduction. If not, do **not** ship.

---

## What I Did NOT Verify (do not act on these)

- **F12** — CLAUDE.md "auto-loads on X edit" mechanism. Need to grep for the loader.
- **C2 prerequisite** — actual contents of `.githooks/pre-commit`.
- **`claude_superpower_brief.build_brief` size** — could be the biggest token line item; not read.
- **Plugin contributions** — `context7`, `commit-commands`, `firecrawl`, `explanatory-output-style` each ship their own skills/commands/agents. Token cost of their always-loaded surfaces not measured.
- **`memory/MEMORY.md` actual line count** vs the 200-line truncation policy.

These are the next reads if you want a fully grounded "ready to implement" version of this plan.
