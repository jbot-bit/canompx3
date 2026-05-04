# CRG Maximization + Strategy Lineage Graph — v2

**Status:** plan, not yet executed. Replaces the v1 draft.
**Authored:** 2026-04-30. **Ground-truth re-verified:** 2026-04-30 (see § Verified facts).
**Anti-goal:** the v1 over-promised manual labour, glossed over absent MCP-prompt wiring,
and hid the parallelism by listing 4 terminals without naming what each one *can* do alone.

---

## What changed from v1 (read this first)

| v1 issue | v2 fix |
|---|---|
| PR-1 said "wire MCP prompts already-present" but `grep` finds **zero** references to `review_changes` / `debug_issue` / `pre_merge_check` in `.claude/agents/`, `.claude/skills/`, `.claude/commands/`. So PR-1 was *adding* them, not wiring them. | PR-1 reframed: "**add** MCP prompt invocations". Honest scope, identical lines. |
| PR-4 bundled AST scanner + builder + MCP tool + CLI in one ~700-line PR. Single review surface, single revert blast-radius. | **Split PR-4 → PR-4a (AST scanner, pure function, ~300 lines) + PR-4b (builder + MCP + CLI, ~400 lines)**. PR-4a is a leaf module; PR-4b imports it. |
| PR-5 promised "manual R3-R8 labels for ~70% of 82" — that's ~57 strategies hand-labeled by reading 360 research scripts and 17 hypothesis files. Bias-prone busywork. The v1 itself flagged "manual maintenance burden" in failure-mode matrix without acting on it. | PR-5 narrowed to **AST-infer R1/R2 only** (high-confidence 30%). Remaining 70% emit `confidence=unknown, slot=UNKNOWN`. PR-6 leaderboard surfaces this honestly as "untagged" — not "grandfathered" (which implied a quality judgment v1 hadn't earned). |
| v1 mentioned spawning 4 parallel Claude sessions but didn't specify which PRs Claude can spawn vs which the user spawns. | **Explicit: each terminal is a separate Claude session the user starts via `scripts/tools/new_session.sh`. This plan tells one Claude what to do per terminal. No Claude-spawning-Claude.** |
| v1 had no exit criteria for "done with the whole plan" — only per-PR. | Added § "Plan-level done criteria". |
| v1 anti-scope said "no `mechanism_prior` column" without defending it. | v2 § "Decisions deferred" acknowledges the column may eventually win, but ships YAML first because YAML is reversible (rm file) and column-add isn't. |
| v1 had no rollback plan if a PR exposes a worse problem mid-stream. | Added § "Halt conditions" — explicit rollback gates between PRs. |
| v1 mentioned `_crg_context_lines` PR #180 fix as motivating story but didn't lift the lesson into a check. | PR-2 acceptance now includes "agent file edits visible in `git diff` end-to-end before merge" — same gap class, same defense. |

---

## Verified facts (re-queried 2026-04-30 against live repo + gold.db, NOT FROM MEMORY)

**Re-verification timestamp:** 2026-04-30 after user said "ground in up-to-date repo truth". Every row below was re-executed; nothing was carried from prior context.

| Fact | Value | Verification command |
|---|---|---|
| `validated_setups` total | **82 (59 active + 23 retired)** | `SELECT COUNT(*), SUM(active), SUM(retired) FROM validated_setups GROUP BY status` |
| `validated_setups` columns | **74** | `DESCRIBE validated_setups \| wc -l` |
| `status` column values | **only 'active' and 'retired'** | `SELECT status, COUNT(*) FROM validated_setups GROUP BY status` |
| `mechanism_prior` column | **does not exist** | `cols=[r[0] for r in DESCRIBE]; 'mechanism_prior' in cols → False` |
| `oos_exp_r` column | **exists** | same check, True |
| `filter_type` distinct values | **23** (top 5: ORB_G5=12, ATR_P50=7, COST_LT12=7, X_MES_ATR60=6, OVNRNG_10=5) | `SELECT filter_type, COUNT(*) FROM validated_setups GROUP BY filter_type` |
| `_crg_usage_log.py` on main | **absent** (no commit anywhere) | `git log --all --oneline -- ".claude/hooks/_crg_usage_log.py"` empty |
| `.code-review-graph/usage-log.jsonl` in `.gitignore` | **present** (line 24) | `grep usage-log .gitignore` |
| `code_review_graph.eval.runner.CONFIGS_DIR` hardcoded | **YES at line 34** | `grep -n CONFIGS_DIR .venv/Lib/site-packages/code_review_graph/eval/runner.py` |
| Phase 3 agent files (5) | **all present** | `ls .claude/agents/{verify-complete,evidence-auditor,blast-radius,planner,ralph-loop}.md` |
| Skill dirs PR-1 expects | **all 3 present** (`.claude/skills/{quant-debug,open-pr,verify}/SKILL.md`) | `ls` |
| `review_changes`/`debug_issue`/`pre_merge_check` references in agents/skills/commands | **ZERO** | `grep -rn "review_changes\|debug_issue\|pre_merge_check" .claude/agents .claude/skills .claude/commands` empty |
| Spec mentions usage-log shim | **lines 153, 217, 257, 258, 263, 307** | `grep -n usage-log docs/plans/2026-04-29-crg-integration-spec.md` |
| Research scripts under `research/` | **360 .py files** | `find research -name "*.py" \| wc -l` |
| Hypothesis files under `docs/audit/hypotheses/` | **17 .md files** | `find docs/audit/hypotheses -name "*.md" \| wc -l` |
| `EVAL-BASELINE-2026-04-30.md` "NOT YET RUN" string | **present (1 occurrence)** | `grep -c "NOT YET RUN" docs/external/code-review-graph/EVAL-BASELINE-2026-04-30.md` |
| `mechanism_priors.md` slots R1-R8 enumerated | **YES, R1-R8 in § 4 "Signal → Role mapping"** | head 100 lines of file |
| `validated_shelf.deployable_validated_relation` | **defined and exported (lines 19, 34)** | `grep -n "deployable_validated_relation" trading_app/validated_shelf.py` |
| `edge_families.build_edge_families_for_instrument` | **defined at line 158** | `grep -n "build_edge_families_for_instrument" trading_app/edge_families.py` |
| `pbo.compute_family_pbo` | **defined at line 136** | `grep -n "compute_family_pbo" trading_app/pbo.py` |
| `strategy_fitness.compute_portfolio_fitness` | **defined at line 734** | `grep -n "compute_portfolio_fitness" trading_app/strategy_fitness.py` |
| `code_review_graph.eval.runner` importable + `run_eval` callable | **YES — `from code_review_graph.eval.runner import CONFIGS_DIR, run_eval` succeeds** | `python -c` import test |
| `code_review_graph.eval.configs/` ships 6 example yamls | **YES (express, fastapi, flask, gin, httpx, nextjs)** | `ls .venv/Lib/site-packages/code_review_graph/eval/configs/` |
| Literature extracts in `docs/institutional/literature/` | **present (Bailey×3, Carver×3, Chan×4, etc.)** | `ls docs/institutional/literature/` |

The `mechanism_prior` column is a genuine canonical gap — but adding it requires a schema migration, drift-check addition, and backfill. Sidecar YAML defers that decision until we *use* the data.

---

## Bias audit (what this plan is biased toward, declared upfront)

Per `docs/institutional/pre_registered_criteria.md` discipline of declaring biases:

1. **Bias toward shipping.** v1 wanted 6 PRs in 3 wall-clock days. v2 keeps the 7-PR shape (PR-4 split into 4a/4b) but explicitly flags PRs 4b, 5, 6 as **post-MVP** — they don't ship in this round unless 1, 2, 3, 4a all land cleanly first. **MVP = PR-1 + PR-2 + PR-3 + PR-4a.**
2. **Bias toward "CRG can find strats".** Honestly reframed in v1 already; v2 carries that forward — PR-4/5/6 are research-velocity tools (gap surfacing), not edge generators. No leaderboard ranking implies "use the top slot to build a strategy".
3. **Bias toward AST > regex.** PR-4a tests must include cases where AST disagrees with regex (e.g., column name appears as a string inside an unrelated SQL fragment). Regex-fallback edges remain visible and labeled, not silently merged.
4. **Bias toward "more telemetry is better".** PR-2 usage-log fail-silent must NOT escalate on log corruption. If the log file becomes a coordination bottleneck (e.g., file lock contention slows agents), the recovery is `rm` + restart — not "add more locking".
5. **Bias toward the v1 plan's structure.** v2 keeps PR boundaries because branches and worktrees are already mentally allocated. If the user prefers a flatter structure (one big PR per topic), v2 should be re-cut, not patched.

---

## PR table v2 (7 PRs, 4 terminals possible)

| PR# | Branch | Scope-lock files | Diff | Depends on | Terminal | Effort | Tier |
|---|---|---|---|---|---|---|---|
| **PR-1** | `chore/crg-mcp-prompts-add` | 5 markdown files (4 agents + 3 skills selectively) | ~50 lines | none | T1 | 2 hrs | MVP |
| **PR-2** | `chore/crg-usage-log-shim` | `.claude/hooks/_crg_usage_log.py` (NEW), `tests/test_hooks/test_crg_usage_log.py` (NEW), 5 agent MD files | ~120 lines | PR-1 (same agent files) | T1 after PR-1 | 4 hrs | MVP |
| **PR-3** | `feature/crg-eval-baseline` | `configs/canompx3-crg-eval.yaml` (NEW), `scripts/tools/run_crg_eval.py` (NEW), tests, `EVAL-BASELINE-2026-04-30.md` (UPDATE) | ~200 lines | none | T2 | half day | MVP |
| **PR-4a** | `feature/strategy-lineage-ast` | `trading_app/strategy_lineage_ast.py` (NEW), `tests/test_trading_app/test_strategy_lineage_ast.py` (NEW) | ~300 lines | none | T3 | 1 day | MVP |
| **PR-4b** | `feature/strategy-lineage-builder` | `trading_app/strategy_lineage.py` (NEW), `tests/test_trading_app/test_strategy_lineage.py` (NEW), `.claude/commands/strategy-lineage.md` (NEW), `trading_app/mcp_server.py` (extend +50 lines) | ~400 lines | PR-4a | T3 after PR-4a | 1-2 days | post-MVP |
| **PR-5** | `chore/mechanism-template-amend` | `docs/audit/hypothesis_registry_template.md` (UPDATE), `docs/mechanism_registry.yaml` (NEW), `scripts/tools/build_mechanism_registry.py` (NEW), tests | ~200 lines | none | T4 | 3 hrs | post-MVP |
| **PR-6** | `feature/mechanism-leaderboard` | `trading_app/mechanism_leaderboard.py` (NEW), tests, `.claude/commands/mechanism-leaderboard.md` (NEW) | ~200 lines | PR-4b + PR-5 | T4 after PR-5 (also waits on T3) | 4-6 hrs | post-MVP |

**MVP wall-clock with 3 terminals (T1, T2, T3):** ~1.5 days
**Full plan wall-clock with 4 terminals:** ~3 days
**Effort:** ~5 person-days unchanged

---

## Plan-level done criteria

The whole plan is done when **all** of these are true:

1. PRs 1, 2, 3, 4a merged to main. Commits are independently revertable per § "Per-PR independence".
2. `EVAL-BASELINE-2026-04-30.md` contains measured token-savings numbers. No "NOT YET RUN" string remains.
3. `.code-review-graph/usage-log.jsonl` shows ≥1 line written by an agent during a real session (smoke evidence).
4. `python -m trading_app.strategy_lineage_ast --scan research/` exits 0 and emits a column-reference report.
5. `pipeline/check_drift.py --fast` passes.
6. Spec line 217 + 308 promise visibly closed: `git log --oneline -- .claude/hooks/_crg_usage_log.py` non-empty.
7. The phrase "no fake numbers" (spec hard-non-negotiable §2) is honored: every claim in this plan that has a number traces to a verification command.

PRs 4b, 5, 6 are explicitly NOT required for "done". They are queued post-MVP work tracked in this same file.

---

## Halt conditions (rollback gates)

After each MVP PR merges, before starting the next:

| If observed | Halt | Why |
|---|---|---|
| Usage-log shim contention slows any agent >2× baseline | Stop. Revert PR-2. | Telemetry must not degrade execution. |
| Eval baseline reveals CRG provides <10% token savings vs. grep on the 10 ground-truth queries | Stop PR-4a. Reassess whether lineage graph is worth building. | If CRG itself isn't pulling weight, a CRG-style lineage tool likely won't either. |
| AST scanner false-positive rate >20% on the canonical column allowlist | Stop PR-4b. Iterate scanner. | Bad ground-truth poisons the gap query. |
| Drift check 16 (venv imports) fails after eval-runner monkey-patch | Stop PR-3. Vendor a config-path arg in a fork instead. | Indicates the patch is brittle to upstream change. |

Halt conditions are evaluated by reading actual output, not vibes.

---

## Branch / worktree allocation (one Claude per terminal)

Per `.claude/rules/parallel-session-isolation.md` and the branch-flip protection in CLAUDE.md.

```
T1 (current canompx3/):                                      → PR-1 → PR-2
T2 (canompx3-crg-eval-baseline/, spawned by USER):           → PR-3
T3 (canompx3-strategy-lineage/, spawned by USER):            → PR-4a → PR-4b
T4 (canompx3-mechanism-leaderboard/, spawned by USER):       → PR-5 → PR-6
```

**The user spawns each worktree** via `scripts/tools/new_session.sh <descriptor>`. **Each Claude
session works one terminal.** A Claude session does NOT spawn another Claude.

If only one terminal is run (this session), the lane is **T1 only: PR-1 → PR-2**. Everything
else is queued in this plan for later sessions.

---

## PR-1: MCP prompts → skills/agents (markdown only)

**Goal:** Make the 3 unused MCP prompts (`review_changes`, `debug_issue`, `pre_merge_check`) actually called.

**Verified state:** `grep -rn "review_changes|debug_issue|pre_merge_check" .claude/agents .claude/skills .claude/commands` returns zero. PR-1 is **adding**, not "wiring".

**Edits (5 files):**
- `.claude/agents/verify-complete.md` — preamble paragraph: "Before Gate 1, call `mcp__code-review-graph__review_changes` (if available) on the diff."
- `.claude/agents/evidence-auditor.md` — preamble: same `review_changes` call before structural ground-truth check.
- `.claude/skills/quant-debug/SKILL.md` — after user supplies error context: "Call `mcp__code-review-graph__debug_issue` with the error excerpt."
- `.claude/skills/open-pr/SKILL.md` — before push: "Call `mcp__code-review-graph__pre_merge_check`."
- `.claude/skills/verify/SKILL.md` — confirm not duplicating `verify-complete` agent. If duplicate, keep verify-complete authoritative; verify skill defers.

**Pattern:** prompts are markdown text. "Wiring" = telling the future Claude to call the prompt as the first action. No code.

**Acceptance:**
- `pipeline/check_drift.py --fast` passes
- `grep -rn "review_changes" .claude/agents .claude/skills` shows ≥2 hits
- Manual smoke: invoke `verify-complete` on a small diff, confirm preamble cites `review_changes` prompt
- Code review: SHIP

**Reuse:** none (markdown).

---

## PR-2: M3 usage-log shim

**Goal:** Close spec line 217 + 308. Without this, no instrumentation exists to measure adoption (closes spec hard-non-negotiable S1).

**Same gap class as PR #177's F1:** the spec promised a file, the file didn't ship, only the surrounding commit message did. PR-2 acceptance includes the same defense (PR-1 same agent files visible in `git diff` end-to-end before merge).

**New file `.claude/hooks/_crg_usage_log.py`:**
```python
def record_crg_call(agent: str, tool: str, query: str | None,
                    token_estimate: int | None) -> None:
    """Append one JSON line to .code-review-graph/usage-log.jsonl. Fail-silent."""
```
Fail-silent on any IOError (institutional-rigor §6 — log must never break agent execution).

**Wire-in points:** 5 agent MD files (`verify-complete`, `evidence-auditor`, `blast-radius`, `planner`, `ralph-loop`) — each gets one paragraph instructing Claude to call `_crg_usage_log.record_crg_call(...)` after every CRG invocation.

**Reuse:**
- `pipeline/check_drift_crg_helpers.py` — pattern reference for fail-silent + sentinel returns

**Tests (`tests/test_hooks/test_crg_usage_log.py`):**
1. Happy path — function writes one valid JSON line
2. Missing `.code-review-graph/` directory — function creates it
3. Read-only filesystem — function silently no-ops, no exception
4. Concurrent calls don't corrupt log (file-lock via `fcntl` on POSIX, no-op on Windows — Windows uses `O_APPEND` atomicity for line writes <512B)

**Acceptance:**
- Drift pass + 4/4 tests pass
- Smoke: run `verify-complete` on small diff, confirm 1 new line in `.code-review-graph/usage-log.jsonl`
- `git diff origin/main..HEAD -- .claude/agents/` shows wire-in paragraphs in all 5 files (closes the F1 gap class)
- Code review: SHIP

---

## PR-3: CRG eval baseline (closes "no fake numbers")

**Goal:** Spec hard-non-negotiable §2. Currently honored as "NOT YET RUN" in `EVAL-BASELINE-2026-04-30.md`. PR-3 closes that.

**Approach (per user choice in v1):** wrapper script that monkey-patches `CONFIGS_DIR`. Does not mutate venv.

**Halt-condition awareness:** if eval shows <10% token savings vs grep on the 10 ground-truth queries, **stop PR-4a** before starting (see § Halt conditions). The lineage graph's premise is that structural search beats text search; if the eval refutes that, lineage graph is rationalization.

**Files:**
- `configs/canompx3-crg-eval.yaml` — schema per upstream `fastapi.yaml`. Curated:
  - `test_commits`: 3 well-known commits (E2 lookahead fix, Holdout Mode A correction, Phase 2 drift checks) — diff scopes verified per `git show --stat`
  - `entry_points`: 5 canonical entry points (drift, discovery, eligibility builder, dst.orb_utc_window, outcome builder)
  - `search_queries`: 10 ground-truth pairs
- `scripts/tools/run_crg_eval.py` — monkey-patches `CONFIGS_DIR = REPO_ROOT/"configs"`, calls `run_eval("canompx3-crg-eval")`. ~30 lines.
- `tests/test_tools/test_run_crg_eval.py` — 3 tests (dry-run, monkey-patch, schema validation)
- `EVAL-BASELINE-2026-04-30.md` — UPDATE with measured numbers

**Reuse:** upstream `code_review_graph.eval.runner` — calls intact, only `CONFIGS_DIR` patched.

**Acceptance:** drift pass + 3/3 tests + real `EVAL-BASELINE.json` committed with measured numbers + halt-condition evaluated.

---

## PR-4a: AST scanner (pure function, NEW vs v1)

**Goal:** Extract a leaf module that is independently useful and independently revertable. Solves v1's PR-4 review-surface problem.

**File `trading_app/strategy_lineage_ast.py`** (~300 lines):
- `scan_python_for_column_refs(path: Path, column_allowlist: set[str]) -> list[ColumnRef]`
- `ColumnRef = NamedTuple(file: Path, lineno: int, column: str, evidence: Literal["ast_literal", "regex_fallback"])`
- Hybrid algorithm:
  1. Regex pre-filter on file content → narrow candidate identifiers
  2. `ast.parse` → walk `Subscript` (`df["col"]`), `Attribute` (`df.col`), `Constant` (string literals in SQL fragments)
  3. Cross-reference vs canonical column allowlist (caller supplies; for production, derived from `DESCRIBE daily_features` at build time)
  4. Edges labeled `ast_literal` (high confidence) or `regex_fallback` (weaker)

**Why a leaf module first:**
- Pure function, no DB, no side effects, no MCP server entanglement
- Can be tested in isolation without `gold.db` or canonical builders
- If PR-4b never lands, PR-4a is still useful as a contamination-detection helper

**Tests (~6 cases):**
1. `df["atr_20"]` → 1 ast_literal hit
2. f-string `f"SELECT {col} FROM"` with `col="atr_20"` const → 1 ast_literal hit
3. SQL fragment `"SELECT atr_20 FROM ..."` const → 1 ast_literal hit (Constant walker)
4. lookahead-banned column (e.g., `break_dir` on E2 cohort) → flagged with evidence label
5. file with no refs → empty list
6. malformed Python → returns empty + warning, does not raise

**Acceptance:**
- Drift pass + 6/6 tests pass
- `python -m trading_app.strategy_lineage_ast --scan research/` exits 0 and prints column-ref report
- No DB writes (`grep -rn "INSERT\|UPDATE\|DELETE\|CREATE" trading_app/strategy_lineage_ast.py` empty)
- Code review: SHIP

---

## PR-4b: Strategy lineage builder + MCP + CLI (post-MVP)

**Goal:** Surface unexplored mechanism slots, dead-feature columns, PBO-similar-but-mechanism-different strategy pairs.

**Files:**
- `trading_app/strategy_lineage.py` (~350 lines) — builder + query API. Imports PR-4a's scanner.
- `trading_app/mcp_server.py` — extend with `query_strategy_lineage(query_type, filter)` tool (~50 lines)
- `.claude/commands/strategy-lineage.md` — slash command
- `docs/runtime/strategy_lineage.jsonl` — regenerable snapshot, full overwrite per build

**Schema** (property graph in JSONL):
- Nodes: `strategy`, `family`, `column`, `mechanism_slot`, `research_script`
- Edges: `BELONGS_TO_FAMILY`, `PBO_SIMILAR_TO` (weight=logit_pbo), `REFERENCES_COLUMN` (evidence=ast_literal|regex_fallback), `CLAIMS_MECHANISM` (confidence=declared|inferred|unknown), `BUILT_BY_SCRIPT`

**Reuse (institutional-rigor §4 — never re-encode):**
- `trading_app/validated_shelf.py:19,49-60` — `deployable_validated_relation(con)` for strategy nodes
- `trading_app/edge_families.py:158-310` — `build_edge_families_for_instrument(con, instrument)` for family nodes + edges
- `trading_app/pbo.py:136-265` — `compute_family_pbo(con, family_hash, instrument)` for PBO_SIMILAR_TO edges
- `pipeline/build_daily_features.py:100-210` — schema source for canonical column allowlist
- `docs/institutional/mechanism_priors.md:65-82` — R1-R8 static enumeration
- `trading_app/strategy_lineage_ast.py` (from PR-4a) — column scanner

**Mechanism tag handling (interim before PR-6 lands):** all `CLAIMS_MECHANISM` edges shipped with `confidence=unknown, slot_id=UNKNOWN`. Renamed from v1's `grandfathered` because that word implied a quality judgment ("we used to allow it") rather than the truth ("we don't know yet").

**CLI:**
```
python -m trading_app.strategy_lineage build [--instrument MGC]
python -m trading_app.strategy_lineage gaps --type {mechanism|column|script}
python -m trading_app.strategy_lineage similar --family-hash <hash>
python -m trading_app.strategy_lineage cluster --feature atr_20
```

**Tests:** integration on MGC: build runs, JSONL written, all 8 mechanism slots emitted, ≥1 family edge, gap query non-empty.

**Acceptance:** drift + integration tests + read-only DB guard verified + code review SHIP.

---

## PR-5: Mechanism template + AST-inferred registry (narrowed)

**Goal:** Source mechanism tags **only where confident**. Defers manual labour until a strategy actually needs it (lazy evaluation principle).

**v2 narrowing vs v1:** AST-infer R1/R2 from `filter_type` for the ~30% of strategies where the filter encodes a known mechanism. **Do NOT hand-label R3-R8 for the remaining 70%.** Those emit `confidence=unknown`. PR-6 leaderboard surfaces this as "untagged" — actionable signal that more strategies need hypothesis files.

**Files:**
- `docs/audit/hypothesis_registry_template.md` — UPDATE with `mechanism_slots: [R1, R2, ...]` field declaration
- `docs/mechanism_registry.yaml` — NEW, ~100 entries (only those AST-inferable + the 17 with hypothesis files)
- `scripts/tools/build_mechanism_registry.py` — runs AST inference from `filter_type` strings + parses hypothesis-file frontmatter for declared slots
- `tests/test_tools/test_build_mechanism_registry.py` — schema + AST correctness + no-duplicate-strategy-id

**AST inference rules (from filter_type → mechanism slot):**
- `ORB_G*` → R1 (gating filter, opportunity-size mechanism)
- `ATR_P*`, `OVNRNG_*` → R1 (volatility regime gating)
- `PD_*`, `GAP_R*` → R2 (prior-day directional context)
- `COST_LT*` → no mechanism — economic filter, not a slot claim
- `X_MES_*` → R2 (cross-asset directional)
- everything else → unknown (do not guess)

**Acceptance:**
- Drift + 3/3 tests
- `len(yaml_entries)` matches AST-inferable count + hypothesis-file count (NOT 82 — that was v1's bias)
- `confidence=unknown` count visible and surfaced
- Code review: SHIP

---

## PR-6: Mechanism leaderboard (narrowed labels)

**Goal:** Per-mechanism aggregate scoring. Answers "which mechanisms actually work?" — honestly, with the unknown bucket visible.

**File `trading_app/mechanism_leaderboard.py`** — reads `docs/mechanism_registry.yaml`, joins to `validated_setups` (`expectancy_r`, `oos_exp_r`, `sample_size`) and `strategy_fitness.compute_portfolio_fitness()`. Per-slot rollup:

```
slot_id | count | mean_ExpR_IS | mean_ExpR_OOS | OOS_dir_match_rate | rolling_Sharpe | effectiveness_index | confidence_mix
```

`effectiveness_index = mean_ExpR_OOS*0.5 + dir_match_rate*0.3 + Sharpe_rolling*0.2`
`confidence_mix` = "{declared:N1, inferred:N2, unknown:N3}"

**Honest gap surfacing:** UNKNOWN slot must always render with its own row, count, and mean. If UNKNOWN dominates ExpR, that's a finding (most edge is in untagged strategies → tag them) not a defect.

**Reuse:**
- `trading_app/strategy_fitness.py:compute_portfolio_fitness()` — never re-encode
- `trading_app/validated_shelf.py:deployable_validated_relation()` — read path
- `docs/mechanism_registry.yaml` from PR-5

**Tests:** all 8 slots + UNKNOWN appear; effectiveness formula matches spec; confidence_mix correct; read-only DB guard.

**Acceptance:** drift + 4/4 tests + slash command runs + UNKNOWN row visible + code review SHIP.

---

## Anti-scope (explicit non-goals, expanded from v1)

Per institutional-rigor §5:

- **No `pipeline/` or `trading_app/` schema changes.** No `mechanism_prior` column on `validated_setups` (sidecar YAML only).
- **No live trading impact.** Drift checks confirm zero `trading_app/live/` edits.
- **No CRG schema or core changes.** PR-3 monkey-patches one variable in a wrapper script. Upstream issue logged separately.
- **No graph traversal library.** DuckDB views + JSONL sufficient. No networkx, no Neo4j.
- **No persistent graph state.** PR-4b's JSONL is a regenerable snapshot.
- **No backfill of `mechanism_prior` to validated_setups.** Sidecar YAML only.
- **No Claude-spawning-Claude.** Each terminal is a separate session the user starts.
- **No hand-labeling 70% of strategies in PR-5.** v2 explicitly narrows that.
- **No "leaderboard says use slot X to find new strats" recommendation.** Leaderboard surfaces gaps; humans interpret.
- **No automated promotion of UNKNOWN → declared.** Tagging requires a hypothesis file (institutional-rigor).

---

## Failure-mode matrix v2

| Mode | Detection | Recovery | Capital impact |
|---|---|---|---|
| Usage-log IO failure | shim returns silently | Agent execution unaffected | None |
| Usage-log contention | wall-clock timing per agent | Revert PR-2 (halt condition) | None |
| Eval baseline upstream signature change | wrapper script test fails | Pin `code-review-graph` version in pyproject | None |
| Eval shows CRG <10% savings | run output | Halt PR-4a (premise refuted) | None |
| AST scanner false positive | column not in `DESCRIBE` allowlist → discarded | Edges labeled `regex_fallback` | None |
| AST scanner FP rate >20% | acceptance test on canonical scripts | Halt PR-4b, iterate scanner | None |
| `edge_families` stale at lineage build | `COUNT(*) WHERE instrument=?` check | Builder calls canonical builder | None |
| Mechanism registry stale (new strategy promoted) | manual maintenance burden | Drift check (Phase 2.x follow-up) | None |
| Leaderboard ranks UNKNOWN top | by design — surfaces tagging gap | Fix by writing hypothesis files, not by hiding the row | None |
| PR-2 ships agent wires but file diff missing one agent | acceptance step "diff visible end-to-end" | Re-add file before merge (closes F1 gap class) | None |

---

## Verification commands

```bash
# Per-PR (run locally before opening PR)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe pipeline/check_drift.py --fast      # all PRs
.venv/Scripts/python.exe -m pytest tests/test_hooks/ -q                              # PR-2
.venv/Scripts/python.exe -m pytest tests/test_tools/test_run_crg_eval.py -q         # PR-3
.venv/Scripts/python.exe -m pytest tests/test_trading_app/test_strategy_lineage_ast.py -q   # PR-4a
.venv/Scripts/python.exe -m pytest tests/test_trading_app/test_strategy_lineage.py -q       # PR-4b
.venv/Scripts/python.exe -m pytest tests/test_tools/test_build_mechanism_registry.py -q     # PR-5
.venv/Scripts/python.exe -m pytest tests/test_trading_app/test_mechanism_leaderboard.py -q  # PR-6

# Plan-level done check
git log --oneline -- .claude/hooks/_crg_usage_log.py | head -1   # non-empty after PR-2
grep -c "NOT YET RUN" docs/external/code-review-graph/EVAL-BASELINE-2026-04-30.md  # zero after PR-3
.venv/Scripts/python.exe -m trading_app.strategy_lineage_ast --scan research/  # exit 0 after PR-4a
```

---

## Per-PR independence verification

| PR | Reverting it breaks… |
|---|---|
| PR-1 | Only the new prompt-call lines in skills/agents (markdown). No tests, no code. |
| PR-2 | Usage logging stops. PR #183 agents still function. |
| PR-3 | Eval baseline goes stale (file remains). No code path depends on the JSON output. |
| PR-4a | `python -m trading_app.strategy_lineage_ast` removed. Pure function, no callers if PR-4b not merged. |
| PR-4b | `/strategy-lineage` slash command + MCP tool removed. JSONL goes stale (gitignored). PR-4a still works. |
| PR-5 | `mechanism_registry.yaml` removed. PR-4b's `CLAIMS_MECHANISM` edges all become `confidence=unknown`. No crash. |
| PR-6 | Leaderboard slash command removed. PR-4b + PR-5 still work. |

Every PR independently revertable.

---

## Decisions deferred (with re-open trigger)

| Decision | Deferred because | Re-open when |
|---|---|---|
| Add `mechanism_prior` column to `validated_setups` | YAML reversible, column-add isn't | YAML drifts >2× per quarter OR a strategy promotion needs the column at write-time |
| Hand-label R3-R8 for untagged strategies | bias-prone; only 17/82 have hypothesis files | UNKNOWN slot dominates leaderboard ExpR (then tag what's worth tagging) |
| Vendor `code-review-graph` config-path arg upstream | monkey-patch is small, fork is large | upstream signature changes break wrapper |
| Run CRG-style lineage on `pipeline/` (not just `research/`) | scope creep | research-velocity gains validated on `research/` first |
| Add file-locking layer to usage-log | fail-silent + O_APPEND atomicity sufficient | observed corruption in real session |

---

## Open follow-ups (carry-forward, non-blocking)

Per `docs/plans/2026-04-29-crg-integration-spec.md` § "Open follow-ups":

1. D5 AST rebuild (advisory carry-forward)
2. D3 constants extension
3. D3 attribute-access detection
4. `reprice_e2_entry` re-add when promoted
5. Live-graph integration test

These are explicitly marked non-blocking in the spec. Tracked there, not duplicated here.

---

## Session execution rules

- **One Claude per terminal.** This Claude session executes whichever lane the user assigns.
- **Branch-flip protection:** every session starts with `git rev-parse --abbrev-ref HEAD && git status --short`. If branch is `main`, no production-code edits — switch to feature branch first via worktree.
- **Stage-gate-guard:** if any worktree is in RESEARCH/DESIGN, production edits blocked globally. Check before PR-2/PR-4a/PR-4b/PR-6 (the production-touching PRs).
- **2-pass implementation:** discovery (read affected files, blast radius, purpose) → implementation (write → drift + tests + behavioral audit → self-review → fix). One task at a time. Never batch without verification.
- **Done = tests pass (show output) + dead code swept (`grep -r`) + `check_drift.py` passes + self-review passed.** All four required.
