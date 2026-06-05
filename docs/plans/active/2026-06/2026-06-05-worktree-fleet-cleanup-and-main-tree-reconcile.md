# Worktree-Fleet Cleanup + Main-Tree Reconcile (2026-06-05)

**Owner:** Claude (main checkout `C:/Users/joshd/canompx3`)
**Status:** TRIAGE LARGELY DONE — fleet 15→8 trees (7 removed 2026-06-05 PM); remaining 5 non-main trees ALL protected (live-C11 / C11-capital-deferred / uncommitted-work). main-tree reconcile DONE.

### FINAL STATE (2026-06-05 PM) — 8 trees, all removable ones removed
**Removed (7):** strict-c11-readiness-merge, precommit-drift-speed-replay, quick-gate-reliability (merged-clean); db-mcp-safe-access, ev-proof-pack-harness (content already shipped); ce7f-daily-radar, 025e-go-portal (detached, origin/rescue backup); plugin-routing-grounding (superseded by #325, 0 unique rule lines); peer-capability-parity-map (docs on HEAD, 0 unique); track-d-gate0-mbp1 (migration+research on HEAD). All backed up (branch ref and/or origin/rescue/*).

**REMAINING — do NOT auto-sweep, each needs a decision:**
| Tree | State | Why protected | Unblock when |
|---|---|---|---|
| `canompx3-wt-c11-cap-x075` | LIVE (idx today) | operator's active C11 audit @ `8d0b540e` | leave to operator |
| `canompx3-wt-audit-8d0b540e` | LIVE (idx today) | C11 verify tree @ `8d0b540e` | leave to operator |
| `daily-bug-scan-restore` | ~525 uniq lines, **capital** (account_survival.py) | head-on collision w/ live C11 | C11 lands → ship restore (supersedes scan) |
| `daily-bug-scan` | ~532 uniq lines, **capital**, BAD commit msg | superseded by restore | kill after restore ships |
| `1248` (detached) | **32 dirty = uncommitted live-pilot** (env_bootstrap.py, reconcile_live_fills.py, broker/tradovate env tests) | UNBACKED-UP valuable work; delete = the wipe incident | owner commits-or-discards |
| `institutional-audit-2026-06-03` | 11 dirty + new fdr.py, audit docs | uncommitted real work | owner commits-or-discards |
| `precommit-drift-speed` | 8 dirty (check_drift/_drift_cache/hooks) | uncommitted real work | owner commits-or-discards |

**KEY METHOD (load-bearing):** baseline is **HEAD, not origin/main**. Local main carries PR-merged work (e.g. #325) that origin/main lacks until pushed → a branch "unmerged vs origin/main" is often fully shipped vs HEAD. 3 branches (plugin-routing, parity-map, track-d) were superseded by HEAD with ZERO unique content — merging them would have REGRESSED rule files to older versions. Always `comm -23 <(git show BR:f|sort -u) <(git show HEAD:f|sort -u)` for true unique-content count.
**Trigger:** Operator git-cleanup session ("~20 worktrees, merge some, finish some, delete the rest" + "ship my shit properly, merged, not hanging around"). NOT the C11 cap-wiring the `/clear` baton pointed at (that's blocked — see §4).

---

## 0b. TRIAGE PROGRESS (2026-06-05 PM session — operator chose "full ship")

State at resume: **main == origin/main == `d2f5eb62`** (everything the plan listed as local-only/in-flight LANDED: cleanup `8f561e73`, baton-staleness `faf8c84e`, governor `cbb72f67`, stat-claim-gate `d2f5eb62`). Plan's old ahead/dirty table was STALE.

**KEY METHOD CORRECTION:** SHA-ancestry "unmerged" ≠ content-unmerged. In this multi-agent repo, work lands via re-commit/squash so a branch shows `ahead=N` while its FILES are byte-identical on origin/main. ALWAYS `diff <(git show BR:f) <(git show origin/main:f)` before merging — several "unmerged" branches were already shipped.

### REMOVED this session (verified safe):
- `strict-c11-readiness-merge`, `precommit-drift-speed-replay` — merged-clean (ahead 0, dirty 0). Branches deleted.
- `quick-gate-reliability` — merged, only HANDOFF.md auto-gen dirt (forced). Branch deleted.
- `db-mcp-safe-access` — **content already shipped** (`db_access.py`+`mcp_server.py` byte-identical on main). Worktree removed; branch ref kept (CI-only delta = fork noise).
- `ev-proof-pack-harness` — **harnesses already shipped** (`bootstrap_health_proof.py`/`bounded_benchmark_harness.py` identical). Worktree removed; branch KEPT — has ambiguous unmerged remainder in `live_readiness_report.py`/`check_drift.py`/`fast_lane_status.py` (stale-vs-superseded UNRESOLVED, needs real diff review).

### LIVE — HANDS OFF (appeared mid-session, NOT in plan's table):
- `canompx3-wt-c11-cap-x075` (session/joshd-c11-cap-x075-wiring, `8d0b540e`) — operator's ACTIVE C11 audit, merging to main imminently.
- `canompx3-wt-c11-verify` (detached `8d0b540e`) — C11 verify tree, live.
- `track-d-gate0-mbp1` (`f888fbec`) — appeared mid-session; not yet triaged.

### REMAINING (genuinely-new content, decision each):
- `peer-capability-parity-map` (parity-map doc NOT on main) — real new doc; rebase-or-ship.
- `plugin-routing-grounding` (`.codex/PLUGIN_ROUTING.md` NOT on main) — real; vet vs current routing rules.
- `daily-bug-scan` + `daily-bug-scan-restore` — BOTH have `daily_bug_scan.py` (+472) NOT on main. TWO variants of same feature; bad commit MSG (stray Codex chat) but REAL content. Pick better variant, ship one, kill other.
- `1248` (detached, ahead 0, **dirty 34** = live-pilot work: `env_bootstrap.py`, `reconcile_live_fills.py`, 27 live modules) — HIGH-VALUE uncommitted; needs owner/commit decision.
- `institutional-audit-2026-06-03` (dirty 13 + new `fdr.py`, audit docs) — real uncommitted; commit-or-discard.
- `precommit-drift-speed` (dirty 8: check_drift/_drift_cache/hooks) — real uncommitted.
- `025e` (detached `3ff16e40`, ahead 3, dirty 1) — go-portal perf; commits backed up on `origin/rescue/2026-06-03/detached-3ff16e40-go-portal`.
- `ce7f` (detached `a59791d5`, ahead 1) — daily-radar; backed up on `origin/rescue/2026-06-03/detached-a59791d5-daily-radar` → safe to remove.

---

## 0. Live coordination state (re-verify every resume — DO NOT trust stale)

- **Operator is committing/pushing on ANOTHER terminal.** In flight at plan-write time:
  `git -C C:/Users/joshd/canompx3-wt-governor-wiring push origin session/joshd-governor-wiring:main`
  (~6min pre-push full-drift gate running). **HOLD all git mutation on main until it lands.**
- origin/main @ plan-write = `ed3348a0` (governor wiring). Governor-wiring push NOT yet landed.
- local main = `566f29f7` = origin/main(`ed3348a0`) + 1 rebased commit `7abceff3`
  (baton-staleness detector — genuine, non-dup; `scan_stale_batons` not on origin).
- **Next sync trigger:** when `governor-wiring → main` lands, origin/main moves → rebase local `7abceff3` on top again, THEN continue.

---

## 1b. RESOLVED SPLIT (2026-06-05, after re-sync to cbb72f67 → local faf8c84e)

Dirty tree now cleanly separated into 3 buckets:
- **A — SAFE CLEANUP (commit alone):** 5 stale C11-throttle doc DELETIONS, all verified no surviving refs:
  `docs/audit/results/2026-06-04-c11-throttle-mechanism-revalidation.md`,
  `docs/handoffs/archived/2026-06-05-root-handoff-archive.md`,
  `docs/plans/2026-06-04-c11-throttle-live-mechanism-design.md`,
  `docs/runtime/stages/2026-06-04-c11-throttle-mechanism-revalidation.md`,
  `docs/runtime/stages/2026-06-04-refresh-c11-c12-control-state.md`.
- **C — STAT-CLAIM-GATE FEATURE (own commit, needs verify + missing pre-commit wiring):**
  `.claude/settings.json` (registers `branch-context.py` Edit|Write hook),
  `.claude/skills/code-review/SKILL.md`, `scripts/tools/check_claim_hygiene.py` (+88),
  `tests/test_tools/test_check_claim_hygiene.py` (new), `docs/runtime/stages/2026-06-05-stat-claim-literature-anchor-gate.md` (new).
  MISSING: pre-commit `[7/8]` filter broadening (dropped with the fast-path revert) — re-add as part of this commit.
- **B — DROPPED (stale, reverted to HEAD):** `dsr-cache-stale-pass-audit.md` edit (DSR fix ALREADY LANDED on origin/main —
  `pipeline/_dsr_policy.py` exists; doc edit was a pre-merge regression), `workflow-reliability-stage-ownership.md` edit (fast-path revert class).

### FOLLOW-UPS (tracked, not now)
- DSR stage doc on HEAD still says `mode: TRUTH AUDIT / status: OPEN` but fix is DONE+LANDED — correct to CLOSED (own doc-truth commit).
- C11 closeout-doc body stale "gate NOT closed" line — see §4.

## 1. Main-tree reconcile — current resolved state (staged, NOT committed)

Cleanup was started against a STALE base (before autopilot+governor+ledger landed). Reconciled by:
stash → rebase local commit onto origin/main → pop → resolve conflicts to upstream → triage deletions.

### SAFE cleanup (commit these)
- 4 stale C11-throttle docs deleted (verified NO surviving references):
  - `docs/audit/results/2026-06-04-c11-throttle-mechanism-revalidation.md`
  - `docs/handoffs/archived/2026-06-05-root-handoff-archive.md`
  - `docs/plans/2026-06-04-c11-throttle-live-mechanism-design.md`
  - `docs/runtime/stages/2026-06-04-c11-throttle-mechanism-revalidation.md`
  - `docs/runtime/stages/2026-06-04-refresh-c11-c12-control-state.md`

### MUST RESTORE before commit (stale deletions of LIVE-referenced things)
- [x] `.githooks/pre-push` — drift safety-net (restored)
- [x] `docs/runtime/stages/drift-path-scope-gate.md` — referenced by test_git_hooks_env (restored)
- [x] `docs/audit/results/2026-06-05-c11-firm-tier-economics.md` — referenced by ledger+scaling-axes (restored)
- [x] `docs/plans/2026-04-22-handoff-baton-compaction.md` — referenced by work_queue.py+test (restored)
- [x] `docs/plans/active/2026-06/2026-06-04-drift-precommit-speed-audit.md` — referenced (restored)
- [x] `docs/audit/results/2026-06-05-c11-scaling-axes-and-wiring-coherence.md` — referenced by ledger (restored)
- [x] `docs/plans/2026-06-04-c11-throttle-implementation.md` — referenced by live-mechanism-design (restored)
- [x] `.githooks/pre-commit` — DROPPED the stale fast-path revert; upstream fast-path kept (operator chose keep-fast-path)
- [ ] **`tests/test_tools/test_compact_handoff.py` (-173)** — STILL STAGED-DELETED, MUST RESTORE: tests live `scripts/tools/compact_handoff.py` + work_queue post-commit accretion-bug guard.
- [ ] **`tests/test_tools/test_git_hooks_env.py` (-129)** — STILL STAGED-DELETED, MUST RESTORE: tests live pre-commit venv/drift logic.

### STILL TO VET (net-diff vs HEAD exists)
- `.claude/settings.json` (+10) — collision file; vet before commit.
- `.claude/skills/code-review/SKILL.md` (+55/-10) — vet.
- `docs/runtime/stages/dsr-cache-stale-pass-audit.md` (+23) — vet.
- `docs/plans/active/2026-06/2026-06-03-workflow-reliability-stage-ownership.md` (+9) — vet.

---

## 2. Separate FEATURE (its OWN commit — NOT "cleanup")

**Stat-claim literature-anchor gate** (per `feedback_enforce_doctrine_at_commit_not_prose`). Half-wired:
- `scripts/tools/check_claim_hygiene.py` (+88) — checker logic
- `tests/test_tools/test_check_claim_hygiene.py` (untracked) — its test
- `docs/runtime/stages/2026-06-05-stat-claim-literature-anchor-gate.md` (untracked) — stage doc
- **MISSING WIRING:** the pre-commit `[7/8]` claim-hygiene filter broadening
  (`docs/audit/|institutional/|plans/|research/`) was DROPPED with the pre-commit revert.
  Committing checker without wiring = dead code. Re-add broadening as part of THIS feature commit, verify, then commit.

**Untracked C11 docs (vet, likely safe new docs):**
- `docs/audit/results/2026-06-05-c11-cap-x080-remediation-v1.md`
- `docs/plans/2026-06-05-c11-stage1-stage2-design.md`

---

## 3. Worktree-fleet triage (THE main job — 17 worktrees + main)

Caveat: most unmerged are `codex/*` = peer Codex work. Operator chose "all worktrees, merge unmerged codex too." Do NOT remove a worktree with a LIVE peer session in it; check dirty + recent mtime first. Deleting a live peer tree = the wipe in `feedback_peer_force_removed_my_worktree_mid_session`.

### MERGED & CLEAN — remove now (work already in origin/main)
- `.codex/worktrees/strict-c11-readiness-merge` (codex/strict-c11-readiness-merge)
- `.worktrees/precommit-drift-speed-replay` (codex/precommit-drift-speed-replay)

### MERGED but DIRTY — work in origin/main; verify dirt disposable then remove
- `.codex/worktrees/ce7f` (detached, ahead=0, **dirty=34 — inspect biggest pile**)
- `.codex/worktrees/precommit-drift-speed` (dirty=8)
- `.worktrees/quick-gate-reliability` (dirty=1)
- `.worktrees/tasks/.../institutional-audit-2026-06-03` (dirty=13)
- `canompx3-wt-c11-cap-x075` (session/joshd-c11-cap-x075-wiring, dirty=8) — YOUR session, merged

### UNMERGED — merge-or-kill decision each (ahead of origin/main)
| Worktree | branch | ahead | dirty |
|---|---|---|---|
| ev-proof-pack-harness | codex/ev-proof-pack-harness | 5 | 0 |
| daily-bug-scan-restore | codex/daily-bug-scan-restore | 4 | 1 |
| db-mcp-safe-access | codex/db-mcp-safe-access | 3 | 0 |
| .codex/.../025e | (detached) | 3 | 1 |
| plugin-routing-grounding | codex/plugin-routing-grounding | 2 | 0 |
| peer-capability-parity-map | codex/peer-capability-parity-map | 2 | 0 |
| track-d-gate0-mbp1 | codex/track-d-gate0-mbp1 | 2 | 0 |
| daily-bug-scan | codex/daily-bug-scan | 1 | 0 |
| **governor-wiring** | session/joshd-governor-wiring | 1 | 0 | ← operator pushing→main NOW; remove after lands |
| .codex/.../1248 | (detached) | 1 | 0 |

**Per-unmerged protocol:** `git -C <wt> log origin/main..HEAD --oneline` + last-commit mtime → if finished & wanted: rebase onto origin/main + push branch→main (operator GO for any capital/schema; Tier A docs/tooling direct) → remove worktree. If abandoned: confirm with operator, `git worktree remove`. Detached/ahead trees: identify the branch the commits belong to before removing (detached HEAD commits are reflog-only after removal).

---

## 4. C11 — SQUARED UP (stop flip-flopping; this is the truth)

**C11 is NOT "ruled out / dead NO-GO." The fix is IN ACTIVE DEVELOPMENT on a peer terminal.**

### What C11 is
The drawdown-magnitude survival gate for `topstep_50k_mnq_auto`. Express budget = 0.90 × $2,000 MLL = **$1,800**.
Uncapped baseline 90d DD = **$2,038.84** → ~$239 over budget → that's the only blocker. Not a strategy — the pass/fail check that stops the lane busting the funded account.

### The fix (the ORB risk cap)
New `risk_cap_pts` field (kept SEPARATE from empirical `p90_orb_pts` — honest, doesn't corrupt the stat).
Cap values per lane: **37.35 / 107.4 / 33.15 pts** (cap_x0.80 remediation) → DD drops to ~$1,594 (97.5%, $206 room) → **clears C11**.

### Audit gate — RESOLVED (was the flip-flop source)
Bracket-parity adversarial-audit gate (`9b3fc530`) is **CLOSED CONDITIONAL**, authoritatively:
- **Source of truth** = stage doc `docs/runtime/stages/2026-06-05-c11-s1-bracket-parity-audit.md` frontmatter:
  `status: DONE`, `verdict: CONDITIONAL-CLOSED`. Independent evidence-auditor (Stage 1c, agentId
  `aabe3bba320f7de8b`) RAN → CONDITIONAL; top finding FIXED Stage 1d (shipped test
  `test_f4_zero_risk_points_blocks_entry_via_safety_gate`, mutation-proven). **Stage 3 unblocked on audit axis.**
- Ledger (`6dadde5b`) `gate-closed-conditional` AGREES.
- **ONLY stale layer:** closeout doc *body* (`2026-06-03-bracket-risk-parity-closeout.md` lines 57/68/72/110)
  still says "gate is NOT closed" — written BEFORE the independent auditor ran. This lone holdout caused every flip.
  **FOLLOW-UP (own commit, after peer C11 lands):** add a 1-line provenance note pointing body→frontmatter resolution. NOT a verdict change.

### Why I (main checkout) must NOT touch C11 now — NOT because it's NO-GO
**A peer terminal is mid-build:** `session/joshd-c11-cap-x075-wiring` worktree (`canompx3-wt-c11-cap-x075`)
has it DIRTY/UNCOMMITTED right now (9 files: `prop_profiles.py`, `lane_allocator.py`, lane JSONs with
`risk_cap_pts`, 2 new tests `test_risk_cap_pts_wiring.py`/`test_check_drift_risk_cap_honesty.py`,
`c11-cap-x080-remediation-v1.md`). Editing C11 capital files on main = head-on collision. HANDS OFF until peer lands.

### Path to live (after peer cap work commits)
peer cap commits → re-run `account_survival` proving C11 passes at $1,800 → confirm wired in lane registry →
fix the stale closeout-doc body line → operator GO (Tier B capital; telemetry WAIVED for express-funded). Nothing armed yet.

---

## 5. Resume checklist (next session / after /clear)

### DONE this session (2026-06-05)
- ✅ Re-synced local main to origin/main `cbb72f67`; rebased baton-staleness commit (now `faf8c84e`).
- ✅ Committed SAFE cleanup `8f561e73` (5 stale C11-throttle doc deletions + this plan). Verified by content.
- ✅ Squared up C11 (§4): CLOSED CONDITIONAL, fix in-flight on peer terminal, NOT NO-GO-dead.

### STATE AT HANDOFF (verify with `git fetch` + `git status` first — may be stale)
- **local main = `8f561e73`, AHEAD 2 of origin/main `cbb72f67`** (2 LOCAL-ONLY commits: `faf8c84e` baton-staleness, `8f561e73` cleanup). NOT pushed. Per no-PR doctrine, integrate direct — but these are Tier A docs/hooks; pushing main is operator-run (auto-classifier blocks self-push to default branch).
- **Uncommitted = stat-claim-gate feature (C) + 2 not-mine files:**
  - C (own commit, NEEDS verify + missing pre-commit `[7/8]` wiring): `.claude/settings.json` (branch-context.py hook),
    `.claude/skills/code-review/SKILL.md`, `scripts/tools/check_claim_hygiene.py`, `tests/test_tools/test_check_claim_hygiene.py` (new),
    `docs/runtime/stages/2026-06-05-stat-claim-literature-anchor-gate.md` (new).
  - NOT MINE — leave alone: `HANDOFF.md` (post-commit hook's last-session auto-update of `8f561e73` — safe), `tests/test_hooks/test_baton_staleness.py` (peer trimming our baton test, -17/+6).
  - Untracked C11 docs (likely peer's, vet owner): `c11-cap-x080-remediation-v1.md`, `c11-stage1-stage2-design.md`.

### NEXT STEPS (priority order)
1. `git fetch`; re-read §0; check if peer's C11 cap work (`c11-cap-x075`) landed on origin/main.
2. **Feature C** — its own verified commit: re-add the pre-commit `[7/8]` claim-hygiene filter broadening
   (`docs/audit/|institutional/|plans/|research/`), run `check_drift.py` + the new test, then commit. Do NOT bundle the 2 not-mine files.
3. **§3 worktree triage (THE main job):** remove 2 merged-clean trees, verify+clear 5 merged-dirty, merge-or-kill 10 unmerged. Per-tree protocol in §3.
4. **Follow-ups:** DSR stage doc → CLOSED; C11 closeout-body stale line (§4) — after peer C11 lands.
5. **§4 C11:** stay OFF cap-wiring (peer owns it). Path-to-live in §4.
