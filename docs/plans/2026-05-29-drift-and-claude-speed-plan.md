# Speed Plan — Drift, Commit, Ralph, and Claude Session Hygiene (2026-05-29)

**Status:** PLAN ONLY — no code written. Awaiting approval per "plan everything, build nothing."
**Author:** Claude (Opus 4.8) session, 2026-05-29.
**Constraint (non-negotiable, user-stated):** Code review and EVERY drift run must stay HONEST.
Speedups change *when/how often* a check runs, never *whether* the commit gate verifies it.
Pre-commit and CI ALWAYS run the full check set. Scoping/caching applies only to the inner loop.

---

## 1. Ground truth (live-measured this session, NOT memory)

| Measurement | Value | How measured |
|---|---|---|
| Full drift wall-clock | **277.7s** (~4m38s) | `Measure-Command { check_drift.py --quiet }` |
| Fast drift (`--fast`, skips 30 tagged-slow) | **184.6s** (~3m5s) | same |
| Import-only floor (interpreter + 15K-line module) | **0.68s** | `python -c "import pipeline.check_drift"` |
| Per-check cumulative (profiler, shared con) | **168.3s across 189 checks** | `scripts/tools/profile_check_drift.py` |

### The 84% — top 9 checks dominate (≈141s of 168s cumulative)

| Time(s) | Check | Nature |
|---|---|---|
| **70.7** | CRG D5: top-10 bridge nodes betweenness-centrality | Graph walk over whole codebase |
| **21.6** | FAST_LANE PROMOTE queue orphans | YAML/MD reconstruction from scratch |
| **12.2** | validated_setups C4 reproduction (MES/MGC) | DB; re-runs Chordia on 844 rows (ADVISORY) |
| **9.1** | Phase-4 SHA migration manifest integrity | SHA-recompute |
| **8.7** | Fast-lane status rollup parity | MD reconstruction (twin of #2) |
| **5.8** | Stage-file staleness vs landed commits | Walks ~29 stage files (ADVISORY) |
| **5.2** | AM3.3 audit-log theory_grant parity | Parses every audit yaml + lit MD |
| **3.6** | DSR reference-universe lock declared | YAML parse |
| **3.5** | CRG D1 + (3.0) D2 + (2.6) D3 | AST/graph walks |

The other ~158 "fast" checks sum to only ~27s.

### Key conclusions (each falsifiable, each measured)

1. **Interpreter startup is NOT the bottleneck** (0.68s). Killing it (daemonizing) buys ~0.7s. Not worth it.
2. **`--fast` is a weak lever** — saves only 93s because the bulk is now spread across the long tail too; and the single 70.7s CRG D5 isn't even the whole story. `--fast` ≠ honest commit speed.
3. **Parallelism is Amdahl-bounded by the 70.7s check.** Perfect parallelization of everything else still floors at ~71s. Parallelism is a *secondary* lever, useful only AFTER the top checks are cached/scoped.
4. **The honest win is per-check input-hashing + scoping.** A check that reads files X,Y,Z should: (a) only run in the inner loop when X/Y/Z changed (scoping), and (b) cache its PASS keyed on hash(X,Y,Z) so even the full commit-time run is instant when inputs are unchanged (caching). When inputs change, the key misses → it runs for real → honesty preserved by construction.
5. **The commit path stacks drift (278s) + a SECOND CRG graph rebuild (step 3b) + tests.** Commit slowness is drift + CRG-rebuild, not tests (~13s staged-aware).

---

## 2. Literature / doctrine grounding

- **Honesty constraint maps to `institutional-rigor.md` §§ 6, 8, 11** — no silent failures, verify-before-claim, never trust metadata. A cache that returns PASS without re-verifying changed inputs would violate § 11 ("reading a cache is not verifying"). Therefore the cache key MUST be the content hash of every input the check reads, and a key-miss MUST re-run the real check. This is the only design that satisfies the rule.
- **`integrity-guardian.md` § 3 Fail-Closed** — cache read errors, hash failures, or unknown inputs MUST fall through to running the real check (fail-closed = run it), never to a blind PASS.
- **Volatile Data Rule (CLAUDE.md)** — this plan cites LIVE measurements; the prior memory note's "4m33s / 25-of-188" figures were re-verified live before use (they held).
- **No new research thresholds touched.** This is pure infrastructure; no Criterion / DSR / Chordia math changes. Out of scope for `pre_registered_criteria.md`.

---

## 3. The honest speedup architecture

### Lever A — Per-check dependency declaration + content-hash cache (PRIMARY)

Extend the `CHECKS` tuple (currently `(label, fn, is_advisory, requires_db)`) with an optional
5th element: a `deps` descriptor naming the path globs / DB tables the check reads.

```
(label, fn, is_advisory, requires_db, deps)
# deps = CheckDeps(paths=["trading_app/dsr.py", ...], globs=["docs/audit/hypotheses/**"], db_tables=["validated_setups"])
```

A cache layer (`pipeline/_drift_cache.py`, new file) computes a cache key =
`sha256(check_label + sorted(hash(each dep file) + schema_version(each db_table)))`.
On a HIT (key file exists under `.git/.drift-cache/` and matches): return cached empty-violation
list instantly. On a MISS: run the real check, store result keyed on the new hash.

**Honesty proof:** the key is a pure function of the check's actual inputs. If ANY input changes,
the hash changes, the key misses, the check runs for real. There is no path where a changed input
returns a stale PASS. Cache corruption/read-error → fail-closed → run the check.

**Applies to commit-time too** (this is the big one): even the full pre-commit drift becomes
near-instant for checks whose inputs didn't change since the last green run. A one-line `dsr.py`
edit invalidates only the ~5 checks that read dsr.py; the other ~184 return cached PASS.

**Risk:** MEDIUM-HIGH. Wrong dep declaration = a check silently caches when it shouldn't.
Mitigation: (1) a meta-drift-check that runs a random 5% of cached checks "cold" each commit and
asserts the cached result matches the cold result (catches under-declared deps); (2) deps default
to "uncacheable" (always run) when unspecified — opt-in per check, starting with the top 9.

### Lever B — `--since <ref>` inner-loop scoping (SECONDARY, composes with A)

`check_drift.py --since HEAD` runs only checks whose `deps` intersect `git diff --name-only HEAD`.
Used by Ralph Step 1 audit gate and the post-edit hook. NEVER used by pre-commit/CI.

**Honesty proof:** scoping is inner-loop only. The commit gate (full set, now cache-accelerated via
Lever A) is unchanged. A scoped run is explicitly a pre-flight, not the verification of record.

### Lever C — Parallelize the file-only (non-DB) checks (TERTIARY)

After A+B cut the heavy checks, parallelize the remaining file/AST checks with a process pool.
DB checks (`requires_db=True`) stay SERIAL on the single shared read-only connection (DuckDB
connection is not thread-safe — confirmed in memory + code: `_shared_con` is one handle).

**Honesty proof:** parallelism changes execution order, not which checks run or their verdicts.
Each check is pure w.r.t. its inputs. Bounded by Amdahl — only worth doing after A removes the 70.7s spike.

### Lever D — Ralph stops double-running drift + report-loss fix (IMMEDIATE, separate)

- Ralph Step 1 audit gate → use `--fast` (or `--since` after Lever B). Step 3 post-fix verify stays full.
- Reorder ralph-loop.md Steps 4/5 so the `=== RALPH LOOP ITER ===` report emits BEFORE the 4-file
  bookkeeping writes — so the report is never lost if the agent runs out of turns (this session's bug).

---

## 4. Claude / PC session hygiene (separate track — addresses "slower the longer it's on")

### Finding (live, this session)
- 34 python + 38 node processes resident.
- **Duplicate MCP server generations**: `repo_state`, `research_catalog`, `strategy_lab`,
  `code-review-graph` each started 3× (08:56, 08:57, +). Stale sets from prior sessions never reaped.
- 2 orphaned `multiprocessing-fork` workers (parents 27984/47328 dead) — **KILLED this session**.
- Stale MCP servers hold read-only `gold.db` handles → lock contention → the exact "commit/drift
  slow because sibling holds the DB lock" class documented in
  `feedback_shared_index_db_lock_precommit_race_2026_05_28.md`.

### Proposed: `scripts/tools/reap_stale_processes.py` (new, opt-in, dry-run default)
Kills ONLY project-signature processes (MCP servers under this repo, fork-workers with dead parents,
pyright-langserver duplicates) that are (a) older than the current `.git/.claude.pid` session start,
OR (b) have a dead parent. NEVER touches a live bot/dashboard (signature match on
`webhook_server|bot_dashboard|--demo|--live` → hard-exclude). Dry-run prints; `--apply` kills.
Wire into `session-start.py` as an advisory reap (fail-open) so each new session cleans up the last.

**Honesty/safety:** read-only by default; capital-path processes hard-excluded; fail-open.

---

## 5. Proposed staging (each its own stage-gate; adversarial audit on check_drift.py edits)

| Stage | Scope | Files | Risk | Saves |
|---|---|---|---|---|
| 0 | Commit profiler harness as verification surface | `scripts/tools/profile_check_drift.py` (exists — just ensure committed) | TRIVIAL | — |
| 1 | Process reaper + session-start wire | `scripts/tools/reap_stale_processes.py`, `.claude/hooks/session-start.py` | LOW | PC/Claude responsiveness |
| 2 | Ralph `--fast` audit gate + report-loss reorder | `.claude/agents/ralph-loop.md` | LOW (doc) | ~90s/Ralph iter + reliability |
| 3 | Cache layer + deps on the TOP 9 checks only | `pipeline/_drift_cache.py` (new), `pipeline/check_drift.py` (CHECKS tuples for 9), meta-verification drift check | HIGH | ~120s commit-time when top inputs unchanged |
| 4 | `--since` scoping + post-edit-hook wire | `pipeline/check_drift.py`, `.claude/hooks/post-edit-pipeline.py` | MED | inner-loop near-instant |
| 5 | Parallelize remaining file-only checks | `pipeline/check_drift.py` runner | MED | tail compression |

Stage 3 is the headline win and the highest risk — it gets the full adversarial-audit gate
(`evidence-auditor`, independent context) per `adversarial-audit-gate.md` because it changes a
truth-layer verification path. The meta-verification drift check (random cold-recheck of cached
results) is REQUIRED to land in the same stage as the cache.

---

## 6. Expected end state (honest)

- **Commit-time full drift:** ~278s → ~60-90s typical (cache hits on unchanged top checks), still
  FULL coverage, still fail-closed. First commit after a big change still pays full cost (correct).
- **Ralph iteration:** ~26min → target ~6-8min (no double drift, scoped audit gate, report never lost).
- **Inner-loop (`--since`) drift:** ~278s → ~10-20s for a typical small edit.
- **PC/Claude:** stale MCP/fork accumulation reaped each session start.

No honesty lost anywhere: pre-commit + CI run the full set; cache keys are content hashes;
key-miss re-runs for real; meta-check catches under-declared deps; capital paths hard-excluded from reaper.
