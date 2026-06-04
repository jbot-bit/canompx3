# Project Self-Consistency & Completion Plan — 2026-06-04

**Goal (operator):** Get the project up to date with itself. STOP opening new work.
Finish / wire / complete what we already have. Then plan the rest.

**Method:** Grounded in live git + DB truth this session (not stage-file paperwork,
which lies). Worktree branch-name audit deliberately reads FILES and cherry-status,
not names, per `feedback_stranded_work_audit_local_only_is_the_risk_not_merge_state`.

> **RECONCILED 2026-06-04 (post-Codex-review):** a later Codex adversarial review
> (`needs-attention`) caught §A1 and §C2 as stale-against-live-git. Corrections
> applied below with evidence lines. Live truth at reconcile time:
> `git status -sb` → `main...origin/main` **0/0** (after a `--ff-only` to
> `de1f9089`); powered-OOS is **LANDED**, not stranded.

---

## A. DONE THIS SESSION (Tier A — already executed)

1. **Synced main to origin.** First sync this session: was 4 behind
   (`b73271f4`→`f29c38bf`, peer REGIME shadow Stages 2-3); discarded a stale local
   `regime-shadow-accumulation.md` strictly subsumed by origin's DONE version.
   **Then drifted to `behind 1`** as origin advanced to `de1f9089` (powered-OOS
   Stage 1). **Reconcile 2026-06-04:** `git pull --ff-only origin main`
   (`Updating 4badd61c..de1f9089`, fast-forward, exit 0) → **now `0/0`**.
   Evidence: `git status -sb` → `## main...origin/main`; `git log -1` → `de1f9089`.
2. **Refreshed C11/C12 control state** — both were `valid=False "db identity
   mismatch"` (DailyRefresh DB-rebuild envelope invalidation). Re-validated via
   `refresh_control_state.py`; both `valid=True`. C11 `gate_ok=False ($2,039 >
   $1,600)` is the **unchanged NO-GO**, not staleness.
3. **Pruned 4 stale-paperwork stage files** (code verified on main):
   fast-lane-2a3 (`9b13d0d1`), fast-lane-provenance (`ef4f0f29`),
   rebuild-outcomes-mnq (lag resolved, MNQ→2026-05-31), dotenv-cleanup (parses clean).

## B. VERIFIED HEALTHY — the "wired but dormant" suspicion, RESOLVED

Operator suspected committed-but-unwired dormant code. Audit (read-only agent + dry-run):

| Suspect | Verdict | Evidence |
|---|---|---|
| REGIME shadow accumulation | **WIRED + LIVE** | `daily_refresh.bat:30` calls the runner; dry-run scans 62 lanes / 0 errored / 0 would-append. 0 shadow rows is CORRECT forward-only boundary effect (forward_start 2026-06-03 > latest bars 2026-05-31..06-02), not a bug. First accrual lands when 2026-06-03+ bars arrive. |
| C11 ORB-cap parity (`9b3fc530`) | **ACTIVE** | cap consumed at `account_survival.py:370`, not just threaded. |
| Drift tree-cache (`5dab3861`) | **CONSULTED** | live read at `check_drift.py:16343`, not bypassed. |
| 59 new-in-30d modules | **ALL referenced** | zero fully-unreferenced modules on main. |

**Conclusion:** No dormant/dead committed code found on main. The one item that
*looked* dormant (REGIME shadow, 0 rows) is correctly gated, not broken.

---

## C. REMAINING WORK — grounded, NOT yet done (the "plan the rest")

### C1. Genuinely-incomplete stages (real code work)

| Item | Risk tier | Why NOT auto-done this pass |
|---|---|---|
| **stop-live-auto-on-startup** | **B (capital-adjacent)** | Adds `-StaleHours` auto-kill filter to `stop_live.ps1` + wires `-NoPrompt -StaleHours 4` into START_BOT.bat. Failure mode = **silently kills a parallel live session on launch**. Was deliberately deferred ("editing process-kill semantics on live-debut day is high-risk"). Needs: StartTime-guard correctness (tz, null StartTime, access-denied procs), PS process-mock tests, AND a real START_BOT launch test (`doctrine_bot_changes_must_be_front_and_back_end`). → own session, adversarial-audit-gated. |
| **sr-watchlist-part-c** | B (capital-truth) | NOT STARTED build: enroll 719 promotable-FIT candidates into SR monitoring, write a SEPARATE `sr_watchlist_state.json` the allocator NEVER reads. Diagnostic (CONTINUE/ALARM/NO_DATA split → is SR gate or forward-sample the allocation ceiling?). Fresh build, not a wiring completion. → `/design` then staged implementation. |

### C2. Stranded commits — RECONCILED 2026-06-04 (was stale; Codex-flagged)

| Branch | Commit | Status (live git 2026-06-04) | Evidence |
|---|---|---|---|
| ~~`session/joshd-powered-oos-reform`~~ | `bcfc581c` | **LANDED — not stranded.** Content landed on `main` via re-commit `de1f9089` (ancestor of HEAD). Files byte-identical to the old branch (`oos_holdout.py` blob `4e0ad69c`, test blob `3cbf8265`). Remaining ref is `origin/rescue/2026-06-03/session-joshd-powered-oos-reform` — a superseded duplicate, safe to delete. `bcfc581c` itself is NOT in main (content re-committed, not merged). | `git merge-base --is-ancestor de1f9089 HEAD`; `git rev-parse main:<f>` vs rescue ref |
| ~~`session/joshd-stale-work-radar` `774f856b`~~ | — | **DOES NOT EXIST in live git.** No local ref, no remote ref, SHA unknown to the repo. Do NOT treat as real work. The branch that *does* exist (`worktree-joshd-radar-review`, sid 2056cbf7) is a **dead, self-reverted** session — its HEAD `b602e6e5` is `revert(drift): Stage 3 ThreadPoolExecutor — audit FAIL + negative speedup`; heartbeat ~2.5h stale, holder PID dead. | `git branch -a` (no match); worktree lease + `git -C <wt> log` |

`track-d-gate0-mbp1` shows unpushed=2 but cherry-empty (already on main; stale tracking ref — no action).
**maximise-ops-fix's 24 "unpushed" are all already-on-main** (cherry-empty) — no action.

**Net: C2 is now empty of genuinely-unlanded work.** No verified stranded strand
remains. Any future "next strand" must be re-derived from live git, not from this
table's prior (now-corrected) claims.

### C3. Correctly blocked on operator (NOT /next-eligible — need a human decision)

- **regime-standalone-eligibility-monitor** — `SCOPED_AWAITING_PARAM_SIGNOFF`.
  Four-gate predicate params need sign-off before any code. Capital-path.
- **c11-80pt-cap-wiring** — PARKED, research-required. C11 is a measured NO-GO;
  cap alone is necessary-not-sufficient. Path to GO = pre-reg cap+stop≤0.50
  remediation, not config wiring. Do not arm live.

---

## D. RECOMMENDED ORDER (smallest blast radius first)

1. ✅ A1-A3 (done) + commit the consolidation.
2. ✅ **C2 stranded commits — RESOLVED 2026-06-04.** powered-OOS landed
   (`de1f9089`); the radar strand does not exist. Nothing to land. (Was: "land via
   2 small PRs" — obsolete.)
3. **C1 stop-live wiring** — own session, StartTime-guard + PS tests + live launch.
4. **C1 sr-watchlist Part C** — `/design` (it's a fresh build).
5. **C3** — surface to operator for the param/research decisions. Do not auto-start.

## D2. GUARD GAP FOUND THIS SESSION (operator-flagged — plan the fix)

**Symptom:** Operator flagged "another terminal in main" while I was committing on
shared main. Verified: 2+ live `claude` + 2 live `codex` processes; a peer
re-dirtied `HANDOFF.md` right after my commit. The guards did NOT stop me.

**Root cause (the real bug):** `worktree_guard --status` reported
`holder PID 31876 ... alive=False, peer_live=True` — the recorded holder PID is
DEAD but a live peer exists. The lease mis-resolves the holder: it can't name
WHICH live terminal holds main, so it never blocked a second session from
committing there. Same class as
`feedback_dead_pid_fresh_heartbeat_lease_trust_heartbeat_not_pid` and
`feedback_phantom_satisfied_venv_and_codex_npx_crashloop_masquerades_as_lease_churn`.

**Secondary gaps:**
- `shared-state-commit-guard` only fires on a sibling-session stage file claiming
  the EXACT `docs/runtime/` path — it guards file-content collision, not
  "two terminals both committing to main."
- No guard enforces "one session per main checkout" — it's convention only.
- branch-flip-guard correctly stayed quiet (branch name never changed) — not the
  relevant guard here.

**Fix (own session, Tier B — concurrency infra, needs tests):**
1. `worktree_guard` holder resolution: when recorded PID is dead but heartbeat is
   fresh + `peer_live=True`, resolve the ACTUAL live holder (scan live claude/codex
   PIDs against the heartbeat) rather than reporting a dead PID. Trust heartbeat
   over PID per the memory rule.
2. Consider a PreToolUse advisory when a `git commit` runs in the main checkout
   while `peer_live=True` — "you're committing on shared main with a live peer;
   prefer a worktree." Advisory, fail-open.
3. Operator workflow: isolated work should start via `START_WORKTREE.bat` /
   `new_session.sh`, NOT in the main checkout.

## E. EXPLICIT NON-GOALS this pass

- No production logic edits (consolidation = docs + state-refresh only).
- No touching sibling worktrees with live Codex/Claude peers.
- No arming C11 / live capital. NO-GO holds.
- No committing runtime DB artifacts. **RECONCILED 2026-06-04:** `live_journal.db.wal`
  was left *untracked but unignored* (a `git add -A` leak risk Codex flagged). Now
  **ignored** via `.gitignore` (added `live_journal.db.wal`, `live_journal.db-wal`,
  `live_journal.db-shm` sidecars next to the existing `live_journal.db` rule). The
  WAL file is left on disk (small + recently written → possible live writer; ignoring
  fully closes the repo-leak surface without touching a live journal). Verified:
  `git check-ignore -v live_journal.db.wal` → exit 0.
