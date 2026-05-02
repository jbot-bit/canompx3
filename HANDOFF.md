# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Pickup pointer (2026-05-02 PM — read this first)

- Active decision memo added on branch `codex/topstep-operator-arch-v2`:
  `docs/plans/active/2026-05/2026-05-02-topstep-operator-architecture-v2.md`
  This is a Topstep/canompx3 operator-architecture V2 memo with explicit
  regime-split scoring, unknowns register, null candidate, and kill criteria.

**Where to start next session:**

1. Read the survey: `docs/audit/results/2026-05-02-deployable-pool-edge-survey.md`.
   It is the operative truth for "what's the next high-EV thread."
2. The chordia_audit_unlock thread is half-done (5/8) and the remaining 3 are
   low-EV under current allocator + profile state — see survey § Decision.
3. **The high-EV next thread is NOT more chordia audits.** It is theory-grant
   feasibility for the prior-day-context filter family — see survey
   § "Higher-EV next threads (ranked)".

**Live capital state (verify before acting):**

- `docs/runtime/lane_allocation.json`: 3 DEPLOY lanes for `topstep_50k_mnq_auto`,
  rebalance_date 2026-05-02 06:07. (Audit log was updated at 11:23 — last
  rebalance is older than the audit log; a fresh rebalance is safe to run
  but produces no book change per dry-run evidence in the survey.)
- `docs/runtime/chordia_audit_log.yaml`: 6 audited rows (5 PASS_CHORDIA, 1 PARK).
  2026-05-02 added `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` PASS_CHORDIA at
  t=4.256 N=522 (canonical replay reproduced before commit). Result MD:
  `docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md`.
  Allocator effect: this lane enters `lanes[]` and demotes the RR1.0 sibling to
  paused via correlation gate.
- Downstream consumers now treat allocator `stale[]` rows as blocked alongside
  `paused[]`:
  - `trading_app/live/session_orchestrator.py` loads both buckets into the
    runtime block set.
  - `trading_app/pre_session_check.py` warns on deployed lanes that are stale
    in allocator output, not just paused.
- Allocator scoring path (`trading_app/lane_allocator.py:_per_month_expr`) now
  injects `cross_atr_{source}_pct` via canonical
  `trading_app.strategy_discovery._inject_cross_asset_atrs` before applying
  filters (2026-05-02). Without this, every `CrossAssetATRFilter` lane silently
  fail-closed and surfaced as STALE despite an active validated_setups cohort.
  Post-fix `lane_allocation.json` 2026-05-02 rebalance: stale[] 6 -> 0; the 6
  X_MES_ATR60 lanes correctly moved into Chordia gate paused (no strict-replay
  audit row exists yet). Same lane composition as before for deployed lanes.
  Audit-log row for `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` corrected:
  sample_size 1719 -> 1695 (was N_universe; the t-stat-bearing N_fired is
  1695, per `docs/audit/results/2026-05-01-chordia-revalidation-deployed-lanes.md`
  line 49). Validated_setups N=1508 is from a different cohort definition
  (win+loss only at promote-time 2026-04-11) and is not directly comparable.

**Tooling state:**

- `research/chordia_strict_unlock_v1.py`: hardened with `WF_START_OVERRIDE`
  cohort lower bound and stop_multiplier fail-closed guard.
  Pressure-tested. Auditing any default-stop strategy_id is one command:
  `python research/chordia_strict_unlock_v1.py --hypothesis-file <prereg>`.
- S-suffixed strategies (e.g. `*_S075`) require an `outcome_builder` rebuild
  at the target stop and a different runner — not yet built.

**Top-of-list candidates if you want to actually expand the book:**

- `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K` (OOS=0.357, all_yrs_pos=T) —
  blocked by MES profile filter, not by stats. Fix: open a MES profile.
- `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` (OOS=0.218, derT=3.74) —
  needs prior-day-context theory grant to drop hurdle from 3.79 to 3.00.
  Literature path: Chan Ch7 + Carver Ch9-10 extracts in
  `docs/institutional/literature/`.
- `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4` (OOS=0.230, all_yrs_pos=T, DSR=0.50) —
  blocked by MGC profile filter. Same fix as MES.

**What NOT to do (already disproved this session):**

- Don't audit `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` or
  `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` next — both correlation-pruned
  vs deployed RR1.0 OVNRNG_100 sibling (rho > 0.7) and ranking-pruned even
  before correlation gate per the dry-run rebalance. Doctrine completionism,
  near-zero portfolio EV.
- Don't trust `validated_setups.oos_exp_r` for any pre-2026-04-08 row
  without recomputing under Mode A — `research-truth-protocol.md` Mode B
  warning applies.

**Session commits (newest first):**

- (this commit) `audit(survey): deployable-pool edge map`
- `f25cc2fe` audit(chordia): MNQ_US_DATA_1000 RR1.0 VWAP_MID_ALIGNED → PASS_CHORDIA
- `5ea34d99` fix(chordia-audit): apply WF_START_OVERRIDE so audit cohorts match canonical promoter

**Working-tree note for next session:**

5 stage-marker files in `docs/runtime/stages/` show as deleted in working
tree (left over from prior closed-stage cleanup). They are NOT mine to
commit. Either commit them in a separate cleanup commit if intended, or
restore them via `git checkout -- docs/runtime/stages/`.

## Current Session (2026-05-02 — Chordia Theory Feasibility + Stale-Plan Reconciliation)

### What landed

- Follow-up fix for the session-router regression review:
  `scripts/tools/session_router.py` once again treats fresh mutating claims
  from the same checkout as routing conflicts. The case-variant
  same-checkout/self allowance remains in `pipeline/system_context.py`
  preflight/claim verification only. This restores the intended
  `codex-project.sh` behavior where a second mutating launch from the main
  checkout auto-routes into a managed worktree instead of staying in the
  shared root.
- Session-launcher self-block fix landed for WSL `/mnt/c` case drift:
  `pipeline/system_context.py` now treats case-variant mount paths that point
  to the same checkout as the same location when verifying/excluding fresh
  claims, and active-claim file keys now derive from directory identity rather
  than raw path casing. That location hardening remains in place for
  preflight/claim verification; the router-side same-checkout exclusion was
  reverted after review because it masked real same-checkout mutating
  conflicts.
- Read-only memo added:
  `docs/audit/results/2026-05-02-chordia-theory-feasibility-scan.md`
- First strict-threshold prereg added:
  `docs/audit/hypotheses/2026-05-02-mnq-usdata1000-vwapmid-o15-chordia-unlock-v1.yaml`
  targeting `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
  as a **no-theory** Chordia unlock audit (`testing_mode: family`,
  `pathway: A_fixed_family`, `K=1`, strict `t >= 3.79`).
- Three follow-on strict-threshold preregs added with the same no-theory
  pattern:
  - `docs/audit/hypotheses/2026-05-02-mnq-cmepreclose-xmesatr60-chordia-unlock-v1.yaml`
  - `docs/audit/hypotheses/2026-05-02-mnq-comex-ovnrng100-chordia-unlock-v1.yaml`
  - `docs/audit/hypotheses/2026-05-02-mnq-comex-costlt12-chordia-unlock-v1.yaml`
- Those four preregs were then normalized to the repo's active
  `metadata`/`execution`/`conditional_role` schema so the prereg front door
  can execute them without ad hoc routing.
- Generic bounded runner added:
  `research/chordia_strict_unlock_v1.py`
  - canonical `orb_outcomes` + `daily_features` replay
  - strict no-theory threshold via `trading_app.chordia.chordia_threshold`
  - explicit scratch-inclusive accounting (`pnl_r NULL -> 0.0`)
  - explicit `cross_atr_MES_pct` enrichment for `CrossAssetATRFilter` parity
  - writes result `.md` + `.csv` only; no writes to `validated_setups` /
    `experimental_strategies`
- Executed all 4 strict unlock preregs through
  `scripts/tools/prereg_front_door.py --execute` (initial run), then
  re-ran with `WF_START_OVERRIDE` cohort fix in
  `research/chordia_strict_unlock_v1.py::_load_universe`. The initial
  cohort included pre-2020-01-01 MNQ trades that the canonical promoter
  excludes via `WF_START_OVERRIDE['MNQ']=2020-01-01`
  (`trading_app/config.py:354`, micro-launch microstructure exclusion).
  After fix, audit `N_fired - scratch` reconciles to
  `validated_setups.sample_size` exactly on all 4 strategies.
  Corrected outcomes:
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
    -> `PASS_CHORDIA`, `t=5.158`, `N_IS=806` (was 889, t=5.547)
  - `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100`
    -> `PASS_CHORDIA`, `t=4.363`, `N_IS=522` (was 529, t=4.414)
  - `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12`
    -> `PASS_CHORDIA`, `t=4.294`, `N_IS=1252` (was 1281, t=4.202)
  - `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60`
    -> `PARK`, `IS t=4.211`, `N_IS=669` (was `FAIL_BOTH` t=3.716, N=700).
    Material reverdict: IS now clears strict 3.79, but OOS sign opposes
    IS at `N_OOS=49 >= 30` → PARK (IS-clean, no OOS confirmation).
  Note: `OVNRNG_100` and `COST_LT12` were already promoted into
  `lane_allocation.json` rebalance_date 2026-05-02 by the parallel
  rebalance — DEPLOY eligibility unchanged (still PASS_CHORDIA after
  cohort correction, only sizes revised in the audit ledger).
- Result artifacts written:
  - `docs/audit/results/2026-05-02-mnq-usdata1000-vwapmid-o15-chordia-unlock-v1.{md,csv}`
  - `docs/audit/results/2026-05-02-mnq-comex-ovnrng100-chordia-unlock-v1.{md,csv}`
  - `docs/audit/results/2026-05-02-mnq-comex-costlt12-chordia-unlock-v1.{md,csv}`
  - `docs/audit/results/2026-05-02-mnq-cmepreclose-xmesatr60-chordia-unlock-v1.{md,csv}`
- `docs/runtime/chordia_audit_log.yaml` updated with 2026-05-02 audit rows for
  those four strategies. Default `has_theory=False` remains unchanged; no new
  theory grants were added.
- Live-path follow-through hardening landed after the allocator truth swap:
  consumers that load blocked strategies from `lane_allocation.json` now treat
  `stale[]` the same as `paused[]` for entry blocking / warnings. This closes a
  downstream gap where stale-but-blocked lanes were visible in JSON but not
  consistently treated as blocked by every consumer.
- Canonical rebalance write for `topstep_50k_mnq_auto` completed successfully
  after path hardening in `trading_app/lane_allocator.py`:
  - new live `docs/runtime/lane_allocation.json` rebalance_date `2026-05-02`
  - live book now expands from 1 lane to 3 lanes:
    - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
    - `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100`
    - `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
- Read/write hardening landed around `lane_allocation.json` path resolution:
  - `trading_app/lane_allocator.py`
  - `trading_app/prop_profiles.py`
  - `trading_app/pre_session_check.py`
  The bug was WSL mount-path casing (`/mnt/c/Users/...` vs writable
  `/mnt/c/users/...`) causing rebalance write failure at the final save step.
- Reconciled stale plan split:
  `docs/runtime/handoff-2026-05-02.md` says "7-lane book / Stage 3 next",
  but live truth in `docs/runtime/lane_allocation.json` is already post-Chordia
  gate and now shows **1 DEPLOY lane** + 16 paused on rebalance_date
  `2026-05-01`.
- Result: do NOT resume the stale "audit the corrected 7-lane book" branch as
  if it were current. The active capital-EV thread is still
  `chordia_audit_unlock_pass_chordia_strategies`.

### Feasibility verdict from local literature only

- `COST_LT*` -> no theory grant; strict `t >= 3.79` only
- `X_MES_ATR60` -> UNSUPPORTED for theory grant from current local extracts
- `OVNRNG_100` -> class-grounded only, not enough for `has_theory: true`
- `VWAP_MID_ALIGNED` -> plausible, but not yet locally grounded enough for a
  doctrine grant

### Recommended next step

- One of the 4 remaining `PASS_CHORDIA`-without-audit names is now closed:
  - `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`
    -> `PASS_CHORDIA`, `t=4.362`, `N_IS=806` (744 win+loss, matches
    `validated_setups.sample_size`). OOS sign matches at `N_OOS=47`,
    `OOS_t=2.143`, `p=0.032`. Direction asymmetric: Long_t=2.390 vs
    Short_t=3.902; pooled gate clears strict, long-only would not.
- 3 remain:
  - `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075` -- BLOCKED on
    runner architecture: `chordia_strict_unlock_v1.py` now fails-closed
    on any non-default `stop_multiplier` because `orb_outcomes` is built
    at the default 1.0 stop. Auditing S-suffixed strategies requires an
    `outcome_builder` rebuild at the target stop and a different runner.
    Tracked separately as a deferred runner extension.
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` -- same-session sibling
    of already-deployed `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100`;
    portfolio-EV question (correlation gate) is downstream of audit.
  - `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` -- same-session sibling
    of already-deployed `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100`;
    same correlation question.
- Runner hardening landed this session:
  - `WF_START_OVERRIDE` cohort lower bound applied so audit replays
    reconcile to canonical promoter cohorts within the documented
    scratch-policy delta.
  - Fail-closed guard on non-default `stop_multiplier`. Pressure-tested
    with a fake `*_S075` prereg; runner refuses with exit code 2 and
    writes no result MD/CSV.
- Continue using the same front-door + bounded-runner flow established by
  `research/chordia_strict_unlock_v1.py`; do not invent a second harness.
- Only do more literature extraction if it could honestly upgrade
  `VWAP_MID_ALIGNED` or `OVNRNG_100` into `has_theory: true`.
- MCP work remains off the critical path for "more high-quality deployed
  trades" in this checkout.

### Execution surface now working here

- User built `.venv-wsl` in this checkout and executed the prereg front door
  successfully from WSL.
- Important harness lesson captured by code, not chat:
  `X_MES_ATR60` is NOT a plain `daily_features` filter. It requires
  `cross_atr_MES_pct` enrichment before canonical delegation. Without that,
  replay fail-closes to zero-fire and yields a false `SCAN_ABORT`.
- Important doctrine fix landed:
  the allocator live gate now reads strict-replay verdicts from
  `docs/runtime/chordia_audit_log.yaml` directly and fails closed to
  `MISSING` when no audit row exists. It no longer derives Chordia deploy
  truth from `validated_setups.sharpe_ratio * sqrt(sample_size)`.
- Post-fix measured behavior on 2026-05-02:
  - `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` now resolves to
    `FAIL_BOTH` from the audit row, not a live recomputed pass.
  - Remaining unaudited siblings resolve to `MISSING`, not `PASS_CHORDIA`.
  - Canonical rebalance for `topstep_50k_mnq_auto` still selects the same
    3 audited lanes, but the saved/report surfaces now reflect the gate:
    4 deployable, 49 paused, 6 stale; 33 pauses are explicitly
    `chordia gate:*`.
  - `docs/runtime/lane_allocation.json` now serializes structured `paused[]`
    and `stale[]` entries with `status`, `reason`, `chordia_verdict`, and
    `chordia_audit_age_days`, so stale-but-audit-failed lanes remain visible
    in the saved runtime state.
- Phantom-stat retraction:
  the originally cited `t=4.565` for
  `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` is UNSUPPORTED. Repo grep on
  2026-05-02 found zero committed matches. Do not cite it in future pre-regs
  or H3 controls.

---

## Prior Session (2026-05-01 EVE — Discovery-Loop Hardening + Chordia P1 Pivot)

### What landed

- **PR #198 MERGED** (Tier 1 discovery-loop): UserPromptSubmit hook detecting pasted agent narration + open-ended hardening verbs.
- **PR #199 MERGED** (Tier 2 discovery-loop): PreToolUse(Edit|Write) marker requirement before edits to `pipeline/`/`trading_app/`. Walks session JSONL transcript for REPRO / context_resolver.py / TRIVIAL artifact.
- **PR #200 CLOSED unmerged** (Tier 3 read-budget counter): closed on self-audit. n=1 trigger incident, arbitrary 16/26 thresholds, hard-cap-no-cooldown alert-fatigue risk, 1,100 lines of meta-tooling against ONE open money P1. Lesson captured in `memory/feedback_meta_tooling_n1_tunnel_2026_05_01.md`.
- **Tier 4 (rule docs only) DEFERRED** — small follow-up if the discovery-loop class actually recurs.

### Pivot to open P1: chordia_audit_unlock_pass_chordia_strategies

**ORIENT step done. NO audits run. NO pre-regs written.**

Live ground truth (queried via `compute_lane_scores` + `validated_setups`):
- 8/59 strategies are PASS_CHORDIA-without-audit
- t-stat range: 3.82 (MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075) → 4.58 (MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15)
- OOS ExpR range: +0.103R → +0.207R
- All `validation_pathway = family` → BHY FDR per Criterion 3
- OOS ≈ IS for all 8 (no obvious decay)

Triage doc: `docs/audit/results/2026-05-01-chordia-audit-unlock-triage.md` (full 8-row table + tier A/B/C correlation pruning + theory-grant matrix + honest +0.4R framing).

### Recommended next step (NOT executed)

Theory-grant feasibility scan: read Fitschen Ch 3 + Carver Ch 9-10 extracts in `docs/institutional/literature/` to determine which of the 4 filter classes (VWAP_MID, OVNRNG, X_MES_ATR, COST_LT) have citable theory grounds. One read pass, one memo, no code, no commits. Decides which strategies are PASS_PROTOCOL_A-eligible vs need-strict-t≥3.79.

Honest framing: highest single OOS ExpR is +0.207R, not +0.4R. Blended portfolio of 4-6 plausibly +0.10-0.18R after costs/correlation drag. +0.4R needs structurally new mechanism, not tuning these.

### Worktrees touched / cleaned

- `canompx3-tier2-edit-marker` — created, PR #199 shipped, REMOVED
- `canompx3-tier3-read-budget` — created, PR #200 closed, REMOVED, branch deleted local + remote
- `canompx3-handoff` — current worktree (this commit lands triage doc + memory entries + HANDOFF append)

### Parallel-session conflict warning

Main worktree (`C:/Users/joshd/canompx3`) at session-end has uncommitted Codex-session work touching `.codex/`, `HANDOFF.md`, `context/institutional.py`, `scripts/infra/codex-*.sh`, plus untracked `.agents/`, `CodePilot/`, `docs/external/code-review-graph/eval-2026-04-29/*.csv`, `scripts/tools/{repo_state,research_catalog}_mcp_server.py`. Per parallel-session-awareness rule, NOT touched. Fresh session must NOT git-add those without confirming ownership with the other terminal first.

---

## Prior Session (2026-05-01 PM — Codex Layer Alignment + Supercharge Roadmap)

### What landed

- Codex repo-skill discovery now matches official Codex behavior:
  `.agents/skills/` contains thin wrappers that forward to the canonical
  `.codex/skills/` sources.
- `.codex/config.toml` now enables small repo-local Codex hooks:
  - `SessionStart` adds startup hints for `/mnt/...` fallback sessions,
    missing `.venv-wsl`, and `.session/task-route.md`
  - `UserPromptSubmit` adds research/review grounding only when the prompt
    actually needs it
- Direct-session startup guidance was hardened around
  `python3 scripts/infra/codex_local_env.py doctor --platform wsl` and
  `setup --platform wsl` instead of brittle raw preflight commands.
- `canompx3_max` now points to `gpt-5.3-codex` instead of the older
  `gpt-5.1-codex-max`.
- `.codex/INTEGRATIONS.md` corrected: live repo MCP map is `gold-db` +
  `code-review-graph`, not `notebooklm`.
- Shared roadmap written:
  `docs/plans/2026-05-01-codex-supercharge-roadmap.md`

### Measured blockers still true

- This checkout is still a `/mnt/c/...` fallback surface.
- `python3 scripts/infra/codex_local_env.py doctor --platform wsl` reports:
  - WSL mount guard FAIL (`.git` write probe read-only)
  - `.venv-wsl` missing
  - preflight FAIL because uv cannot fetch Python/build deps from the current
    network-restricted state

### Next build order

1. Use a WSL-home clone such as `~/canompx3` for real Codex work.
2. Build `repo-state` MCP from:
   `context_resolver.py`, `task_route_packet.py`, `project_pulse.py`,
   `system_context.py`, `context_views.py`.
3. Build `research-catalog` MCP over `docs/institutional/`,
   `docs/audit/hypotheses/`, `docs/audit/results/`, and existing context
   catalog tooling.
4. Build `strategy-lab` MCP only after those two, using `gold-db` as the truth
   substrate instead of inventing a second state layer.

## Prior Session (2026-05-01 PM — Allocator Chordia Gate)

### Current state at handoff

**PR #197** open: `fix/allocator-chordia-gate` → `main`. 3 commits:
```
a37bf7df  fix(test): test_lane_ctl picks first available lane
67129d04  fix(chordia-gate): code-review findings — drift path, malformed yaml, deprecation
49688f05  feat(allocator): Chordia gate prevents silent rebalance-bypass class
```

**First action on resume**: `gh pr view 197 --json state,mergeable,statusCheckRollup`.
- Green → merge (squash fine).
- Red → read `gh run view <id> --log-failed` before any edit.

### What this PR does

Stage `allocator_chordia_gate` (P1 from `action-queue.yaml`). Refuses DEPLOY for
any strategy failing Chordia 2018 t-stat. Two-layer defense:
1. `apply_chordia_gate(scores)` inline at top of `build_allocation()` —
   demotes FAIL_BOTH / FAIL_CHORDIA / MISSING / stale-audit (>90d) to PAUSE.
2. Drift check #134 `check_lane_allocation_chordia_gate` — refuses any FAIL_*
   lane in `lane_allocation.json` (catches hand-edits / code reverts).

**Verdict taxonomy** (`trading_app.chordia.chordia_verdict_label`):
- PASS_CHORDIA (t≥3.79, sizing-up eligible) | PASS_PROTOCOL_A (3.00≤t<3.79 + theory)
- FAIL_CHORDIA (3.00≤t<3.79 no theory) | FAIL_BOTH (t<3.00) | MISSING (no data)
- Only PASS_* permits DEPLOY (`chordia_verdict_allows_deploy`).

**Architectural decision**: no new gold.db table. `validated_setups.sharpe_ratio` +
`sample_size` recompute the t-stat live every rebalance. Persisted artifact:
`docs/runtime/chordia_audit_log.yaml` — per-strategy `has_theory` doctrine grant +
`audit_date` (for staleness). YAML's `verdict` field is a HUMAN LEDGER ONLY; the
allocator ALWAYS recomputes from `validated_setups`.

**Live impact** (verified 2026-05-01 against gold.db):
- Pre-gate `topstep_50k_mnq_auto`: 4 lanes (EUROPE_FLOW / COMEX_SETTLE /
  NYSE_OPEN / TOKYO_OPEN, all DEPLOY).
- Post-gate: **1 lane** — `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` (PASS_PROTOCOL_A,
  t=3.412 IS / 3.511 live recompute).
- 8 strategies sit at PASS_CHORDIA but are blocked by missing-audit gate. ← UNLOCK PATH

### Reviews completed

- `evidence-auditor` (capital-class grounding) — YELLOW; YAML-vs-live verdict
  discrepancy on H2/H4 (both still block DEPLOY identically). Addressed inline.
- `general-purpose` engineering review (agentId af59919cf2bf3c285) — caught 3
  ship-blockers, all fixed in `67129d04`:
  - H1 fail-open class: drift check used CWD-relative path. Now `PROJECT_ROOT`.
  - H2 docstring lie: malformed YAML actually raised. Now catches + WARNs.
  - M1 dead API: `chordia_gate()` deprecated (kept for legacy test_chordia.py).

### What's NOT done — ranked by EV

**A. 8 PASS_CHORDIA strategies need doctrine audits  ← highest EV**

The user's stated aim this session: "+0.4R+", "sky is the limit, can't get there
if you don't aim". Honest framing: H3 NYSE_OPEN ExpR ≈ +0.079R; +0.4R is 5×.
Not reachable by tuning the same setup harder.

The 8 PASS_CHORDIA strategies are statistically eligible (t≥3.79) but blocked
by missing audit. **Auditing each = the concrete unlock**. Pathway-A revalidation
is the institutional method (see `RESEARCH_RULES.md`); do NOT auto-write audit
rows without running the actual audit. Pre-reg per Phase 0 grounding required.

Enumerate:
```python
from datetime import date
from trading_app.lane_allocator import compute_lane_scores
scores = compute_lane_scores(date.today())
[s.strategy_id for s in scores
 if s.chordia_verdict == "PASS_CHORDIA" and s.chordia_audit_age_days is None]
```

**B. `WorkQueue` pydantic schema drift** (~10 min, infra hygiene)

`scripts/tools/project_pulse.py` blows up on schema validation. `pipeline/work_queue.py:165`
rejects `class=audit`, `class=stage`, `status=open` in `action-queue.yaml`. Symptom:
`/orient` skill fails its pulse step. Fix: extend Literal enums in
`pipeline/work_queue.py` OR canonicalize YAML to existing values. Check recent
queue commits for direction.

**C. `stage-gate-guard.py` worktree-awareness**

Hook reads `Path("docs/runtime/stages")` from main worktree's cwd, ignoring the
worktree of the edited file. Same class as `feedback_crg_worktree_repo_root_resolution.md`.
Workaround used this session: mirror stage file into main's `stages/` dir. Real
fix: resolve `STAGES_DIR` via the worktree-of-edited-file's git root. Causes
friction every multi-worktree session.

**D. Code-review polish (M2/M3/M4 from review af59919cf2bf3c285)** — ship-OK
- M2: `apply_chordia_gate` 23-kwarg LaneScore rebuild → use `dataclasses.replace`
- M3: integration test compute → save → drift round-trip (catches schema-name typos)
- M4: `Literal` typing on verdict; shared `CHORDIA_DEPLOY_VERDICTS` constant
  (currently `("PASS_CHORDIA", "PASS_PROTOCOL_A")` is duplicated across files)

**E. Action-queue housekeeping** (post-merge)

Flip `allocator_chordia_gate` entry status from `open` to `closed`. Set `notes_ref`
to PR #197 URL + `docs/audit/results/2026-05-01-chordia-revalidation-deployed-lanes.md`.

### Don't pigeon-hole

User explicitly flagged this. On resume: `/orient` first (after fixing B above
if blocking), check live `action-queue.yaml`, two-track decide. Do NOT auto-route
to "next adjacent task" — pick by EV.

### Friction noted (not fixed this session)

- Cross-worktree stage-file mirror dance (Item C above).
- `claude` interactive launch from sandboxed Bash detaches and the child window
  dies → can't reliably spawn fresh Claude sessions in sibling worktrees from
  inside Claude. User opens the terminal manually.

---

## Prior Session (2026-05-01 — Allocator orb_minutes Hardcode Fix)

### What landed this session

**Capital-class structural fix** for `trading_app/lane_allocator.py`. Allocator
hardcoded `orb_minutes = 5` at 5 query sites; O15 strategies in the live
`topstep_50k_mnq_auto` profile were scored against O5 aperture data, not their
validated O15 aperture. Trailing ExpR / trailing_n / months_negative / DD-budget
P90 ORB sizes were wrong for these lanes.

- Audit (fresh-context): agents `a1e76860ea5635815` + `ab19fcc9e39814af2`
- Audit doc: `docs/audit/results/2026-04-30-allocator-orb-minutes-hardcode-audit.md`
- Interim mitigation: PR #188 (paused 2 O15 lanes pending structural fix)
- Allocator rerun on 2026-04-18: 6 → 7 lanes, aperture mix 4 O5 / 2 O15 / 1 O30
- SINGAPORE_OPEN_O15 ExpR corrected: 0.2407 → 0.1332 (-45%); p90 37.8 → 60.1pts (+59%)
- Tests: 47/47 pass, drift 117/117 PASS
- One hardcode kept by design: `_compute_session_regime` (rationale comment in code)

**Operator decision pending after merge:** new lane composition includes 3 lane
swaps (NYSE_OPEN COST_LT12 → ORB_G5; TOKYO_OPEN COST_LT12 → COST_LT08; US_DATA_1000
ORB_G5_O15 → VWAP_MID_ALIGNED_O15) plus a new O30 lane (SINGAPORE_OPEN_ATR_P50_O30)
that fits budget under correct DD math. Glance at validation history before
flipping live JSON.

## Prior Session (2026-04-29 — PARKed Pathway-B Additivity Triage)

### What landed this session

**Additivity audit for D2 + D4 PARK candidates** (research/2026-04-29-parked-pathway-b-additivity-audit.py)
- Result: `docs/audit/results/2026-04-29-parked-pathway-b-additivity-triage.md`
- Verdict matrix:
  - **D2 B-MES-EUR** → **PASS_ADDITIVITY** (worst_rho +0.32, N=387). Optionality preserved; Phase E admission still gated on OOS power floor (~Q3-2026).
  - **D4 B-MNQ-COX** → **FAIL_ADDITIVITY** (worst_rho **+0.81** vs deployed `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`, subset 100%, N=407). Confirms HANDOFF flag (+0.7733) and exceeds canonical `RHO_REJECT_THRESHOLD = 0.70`. **D4 cannot deploy as additive even if OOS power clears.** Two recovery paths: (1) re-spec as a same-cell *replacement* of the deployed lane (head-to-head pre-reg required); (2) park indefinitely until the deployed COMEX_SETTLE ORB_G5 anchor decays out.
- Brute-force OOS accrual on D4's existing predicate is wasted under the current canonical threshold.

**Production-code change (config.py only):** Two `_HYPOTHESIS_SCOPED_FILTERS` registrations + `strict_gt: bool = False` opt-in field added to `OvernightRangeFilter` and `GARCHForecastVolPctFilter`.
- `OVNRNG_PCT_GT80` (`overnight_range_pct > 80` strict) — for D2 audit.
- `GARCH_VOL_PCT_GT70` (`garch_forecast_vol_pct > 70` strict, direction="high") — for D4 audit.
- Both filters NOT in `BASE_GRID_FILTERS`, NOT routed in `get_filters_for_grid()` for any (instrument, session). Reachable only via canonical `ALL_FILTERS` lookup or Phase-4 hypothesis-injection. Verified by 36-cell test sweep in `TestStrictGtVariants`.
- Default `strict_gt=False` preserves existing `>=` semantics on all 5 prior `OvernightRangeFilter`/`GARCHForecastVolPctFilter`-shaped instances (regression-tested).

### OOS power-floor accrual ETA (read-only check, 2026-04-29)

DB max `trading_day = 2026-04-26`. D2 OOS density is 0.177 trades/cal-day
(20 trades over 113 cal days), giving a **projected ETA to N_OOS_on=50
of 2026-10-09 (early Q4-2026)** — ~6 months further than the
"Q3-2026" wording in the original D2 pre-reg + 2026-04-28 HANDOFF.
D4 (predicate as written, even though Path-3-reframed) accrues at 0.327
trades/cal-day and would hit N=50 by ~2026-06-01, but RULE 7 still
blocks deployment as a new lane regardless of OOS N. See triage doc
"Addendum 2".

### Decision gate

D2 stays in PARK queue for OOS accrual (additivity-clean).
D4 is now **dual-gated**: OOS power floor + RULE 7 additivity. Existing predicate cannot pass both. Decision moves from "wait for N" to "respec or park". Recommend respec as replacement-lane Pathway-B if user wants D4 alive; otherwise close.

D-0 v2 (PARK_ABSOLUTE_FLOOR_FAIL) excluded from this triage — different failure class. B-MNQ-EUR does not exist as pre-reg (verified 2026-04-29).

### Verification

- `tests/test_trading_app/test_config.py`: 203/203 pass (186 prior + 17 new strict-gt regression tests).
- `pipeline/check_drift.py`: 20 pre-existing violations from local `certifi`/`annotated_types`/`click` env gaps; verified unchanged pre/post edit via `git stash` baseline. No new violations introduced.
- Audit runner exit 0 on canonical `gold.db`. D2 N=387, D4 N=407 — both above noise floor.
- Read-only against canonical layers; no writes to `validated_setups`, `lane_allocation.json`, or live config.

---

## Prior Session (2026-04-28 — Phase D D4 B-MNQ-COX Pathway B K=1 — PARK_PENDING_OOS_POWER)

### What landed this session

**Phase D D4 — B-MNQ-COX Pathway B K=1** (`research/2026-04-28-phase-d-mnq-comex-settle-pathway-b`,
commits `f7e2c921` pre-reg + `733883ed` runner + this verdict commit)
- Pre-reg: `docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml`
- Runner: `research/phase_d_d4_mnq_comex_settle_pathway_b.py`
- Result: `docs/audit/results/2026-04-28-mnq-comex-settle-pathway-b-v1-result.md`
- **Verdict: PARK_PENDING_OOS_POWER** — all 7 KILL criteria PASS, all non-conditional
  C5/C7/C9/Sharpe gates PASS, but C6/C8 are GATE_INACTIVE_LOWPOWER (N_OOS_on=17,
  power=0.106). Per Amendment 3.2: UNVERIFIED ≠ KILL. Cell parks until N_OOS_on ≥ 50
  (~Q3-2026).
- Numbers: N_IS_on=199, ExpR_IS_on=+0.2453, ΔIS=+0.2286, Welch t=3.18, p=0.00161,
  bootstrap p=0.00190, Sharpe_ann_IS=+1.692, DSR_PB=0.9998, dir_match=True
  (ΔOOS=+0.2538), 6/6 years positive IS_on.
- Pre-reg locked → no post-hoc threshold rescue when OOS accrues.
- **RULE 7 flag preserved:** lane-correlation +0.7733 vs deployed COMEX_SETTLE
  ORB_G5. CANDIDATE_READY/PARK verdict is necessary, not sufficient, for Phase E
  admission. Portfolio additivity audit required before any capital deployment.

### Decision gate

D4 (B-MNQ-COX) → PARK_PENDING_OOS_POWER. Real-money exposure remains unchanged
because (1) OOS is underpowered for C6/C8 verification, AND (2) RULE 7 portfolio
overlap with deployed COMEX_SETTLE ORB_G5 is +0.773 — additivity unproven.

D2 (B-MES-LON) and D3 (B-MNQ-NYC) still pending user GO. Each candidate carries
the same likely outcome (PARK_PENDING_OOS_POWER given identical N_OOS power-floor
situation).

### Verification

- Pre-commit gauntlet (8/8) PASSED on both pre-reg and runner commits.
- Independent SQL reproduction of Phase B numbers confirmed exact match before
  pre-reg commit (delta_IS=0.2286 to 4dp).
- Bootstrap p computed on B=10000, block=5 (Phipson-Smyth correction applied).
- KILL_BASELINE_SANITY PASSED: |reproduced - expected| = 0.000014 R.

---

## Prior Session (2026-04-28 — Phase D D-0 v2 clean re-derivation — PARK_ABSOLUTE_FLOOR_FAIL)

### What landed this session

**Phase D D-0 v2 clean re-derivation (MNQ COMEX_SETTLE E2 CB1 OVNRNG_100 RR1.5):**
- Pre-reg: `docs/audit/hypotheses/2026-04-28-phase-d-d0-v2-garch-clean-rederivation.yaml` (SHA `823b0127`, PR #170 merged)
- Runner: `research/phase_d_d0_v2_backtest.py` (new, scratch-policy: realized-eod, garch predictor, bootstrap p)
- Result: `docs/audit/results/2026-04-28-phase-d-d0-v2-garch-backtest.md`
- **Verdict: PARK_ABSOLUTE_FLOOR_FAIL**
  - Relative Sharpe uplift: **+16.48%** ≥ 15% threshold → PASS
  - Absolute Sharpe diff: **+0.0283** < 0.05 floor → FAIL
  - Bootstrap p: **0.50** > 0.05 → FAIL (underpowered)
  - H1 requires ALL THREE gates; 2/3 fail → PARK, not KILL
- Action-queue item `phase_d_d0_clean_rederivation` closed with override note: gate eval deferred to 2026-05-15 or later; do NOT promote tainted D-0 v1 or D-0 v2 PARK to deployment.
- `phase_d_d0_backtest.py` remains TAINTED (do not modify); v2 script lives alongside.

### Decision gate

D-0 v2 is PARKED — clean predictor does not rescue the regime-amplification thesis with statistical confidence at IS N. **2026-05-15 gate eval has no clean D-0 v2 PASS basis.** Daily shadow accumulation should remain parked.

D2/D3/D4 pre-regs remain pending user GO (see prior session below).

### Verification

- PR #170 merged, 5029 tests passed, 28 skipped (CI SUCCESS).
- `python pipeline/check_drift.py`: run on prior session; no pipeline/ or trading_app/ files touched this session.

---

## Cross-session loss notice (2026-04-28 — crg-calibration session)

**Phase D YAML uncommitted WIP lost.** During calibration session on PR #171 (`crg-calibration` branch), I stashed `docs/audit/hypotheses/2026-04-28-phase-d-d0-v2-garch-clean-rederivation.yaml` (single-char ` M` modification on top of committed `2c8e52ab9889`) with label `phase-d-yaml-WIP-from-other-session-preserved-for-crg-calibration` to switch branches. Stash later disappeared from `git stash list` and from `git reflog stash` — no command in my session removed it. Suspected cause: parallel-session interference or background hook reaping. **Committed version on `prereg-phase-d-d0-v2-garch` is intact (519 lines, blob `2c8e52ab9889`, branch tip `823b0127`).** If the WIP edits were load-bearing, fsck unreachable objects (~5,600) may still contain the blob for ~30 days — search by yaml content. Otherwise re-edit from committed baseline. (Note: PR #170 D-0 v2 work has since landed on main, so the lost WIP may have been superseded by that work.)

---

## Prior Session (2026-04-28 — doctrine fix + Phase D dispatch — branch `triage-e2-lookahead-9-candidates`)

### What landed this session

**Root-cause doctrine fix for the 2026-04-21 postmortem § 5.1 retro-audit:**
- `.claude/rules/backtesting-methodology.md § 6.1`: removed `rel_vol_{s}` from safe list (the description contradicted `pipeline/build_daily_features.py:1600-1660` which computes numerator as `break_bar_volume`).
- `.claude/rules/backtesting-methodology.md § 6.3`: added `rel_vol_{s}` to E2 banned list with canonical-source cite + Chan Ch 1 p.4 cite.
- `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md § 9`: closure note for 2026-04-28 follow-up. Documents two found drifts: rel_vol § 6.1 wording, and `late-fill-only` annotation as statistically unsafe for signal discovery (Chan p.4 selection bias).
- `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md`: added "Re-derivation dispatch" section. Priority 1 = Phase D D-0 clean re-derivation with HARD DEADLINE 2026-05-15 (predictor swap: `garch_forecast_vol_pct` first, `atr_20_pct` second). Priority 2 = historical predictor-tainted scripts deferred unless downstream citation appears.
- Memory: `phase_d_daily_runbook.md` flagged D-0 locks as research-provisional pending clean re-derivation.
- Stage file: `docs/runtime/stages/triage-e2-lookahead-9-candidates.md`.

### Decision gate

Phase D D2/D3/D4 still pending user GO. **Phase D D-0 clean re-derivation must land before 2026-05-15** to avoid the gate eval consuming contaminated baseline. Recipe is in registry dispatch § Priority 1 — pre-reg amendment + predictor swap, K=1 framing preserved.

### Verification

- `python pipeline/check_drift.py`: 114 passed, 0 skipped, 10 advisory (0 violations on check 124).
- `pytest tests/test_pipeline/test_check_drift_e2_lookahead.py`: 10/10 pass.
- No `pipeline/`, `trading_app/`, schema, or canonical-config files touched.

---

## Prior Session (2026-04-28 — drift check 124: 9 e2-lookahead-policy annotations)

### What landed this session

**Drift check 124 clearance** — 9 scripts surfaced by `check_e2_lookahead_research_contamination()` annotated:
- TAINTED (7): `phase_d_d0_backtest.py`, `break_delay_filtered.py`, `break_delay_nuggets.py`, `l1_europe_flow_pre_break_context_scan.py`, `mnq_comex_unfiltered_overlay_v1.py`, `mnq_l1_europe_flow_prebreak_context_v1.py`, `shadow_htf_mes_europe_flow_long_skip.py` (reclassified after Opus audit)
- CLEARED (1): `audit_sizing_substrate_diagnostic.py` (implements the gate itself)
- NOT-PREDICTOR (1): `output/confluence_program/phase0_run.py`
- Registry rows 19-27 added; row 27 reclassified `not-predictor → tainted` after real-data audit on MES EUROPE_FLOW O15 E2 IS (N=1719) showed 42.6% trades have `entry_ts < break_ts` → `break_dir='long'` selector is post-entry on those rows
- Drift check 124 now passes (0 remaining unannotated)
- Pre-existing test failure `test_pulse_integration::test_text_output_is_scannable` (61>60 lines) confirmed pre-existing

### Decision gate

Phase D D2/D3/D4 still pending user GO (HANDOFF below).
`phase_d_d0_backtest.py` is TAINTED — D-0 pre-reg result needs clean re-derivation before D-0 claim can be cited (high-EV next step: re-derive on `garch_forecast_vol_pct` per Carver Ch 9-10).
`shadow_htf` ledger continues recording (zero-capital observational); re-pre-register required before any deployment use.

---

## Prior Session (2026-04-28 — Phase D D1 + cleanup + hardening)

- **Tool:** Claude Code (autonomous)
- **Date:** 2026-04-28
- **Pickup point:** Phase D D1 verdict landed (B-MES-EUR PARK_PENDING_OOS_POWER); D2/D3/D4 pending user GO

### What landed this session

1. **Stage hygiene cleanup** (`chore/close-landed-stages-2026-04-28`, commits `55c458d7` + `2a474ea7`)
   - Closed 2 stage files for already-merged PRs #158 + #152
   - Added drift check 121 `check_stage_file_landed_drift` (advisory) — surfaces stages
     where `updated:` is >7d old AND ≥3 commits reference the slug, exactly the class
     of bug that caused this session's "thought we sorted this already" confusion.

2. **Phase D D1 — B-MES-EUR Pathway B K=1** (`research/2026-04-28-phase-d-mes-europe-flow-pathway-b`,
   commits `d58e5ce2` + the verdict commit)
   - Pre-reg: `docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml`
   - Runner: `research/phase_d_d1_mes_europe_flow_pathway_b.py`
   - Result: `docs/audit/results/2026-04-28-mes-europe-flow-pathway-b-v1-result.md`
   - **Verdict: PARK_PENDING_OOS_POWER** — all 7 KILL criteria PASS, all non-conditional
     C5/C7/C9/Sharpe gates PASS, but C6/C8 are GATE_INACTIVE_LOWPOWER (N_OOS_on=9, power=0.106).
     Per Amendment 3.2: UNVERIFIED ≠ KILL. Cell parks until N_OOS_on ≥ 50 (~Q3-2026).
   - Pre-reg locked → no post-hoc threshold rescue when OOS accrues.
   - DSR_PB = 0.9845, Welch p = 0.000602, bootstrap p = 0.000500, |t| = 3.47, era stable 7/7.

### Decision gate at session-end

D1 (B-MES-EUR) → PARK. D2/D3/D4 still pending user GO before pre-regs are written:
  - **D2** B-MES-LON: MES LONDON_METALS O30 RR2.0 long + ovn_range_pct_GT80 (Chan Ch7 + Fitschen Ch3)
  - **D3** B-MNQ-NYC: MNQ NYSE_CLOSE O5 RR1.0 long + ovn_range_pct_GT80 (Chan Ch7 + Fitschen Ch3)
  - **D4** B-MNQ-COX: MNQ COMEX_SETTLE O5 RR1.0 long + garch_vol_pct_GT70 (Carver Ch9-10) —
    CAVEAT: B6 lane-correlation +0.773 vs deployed COMEX_SETTLE ORB_G5 — likely portfolio overlap, run anyway but flag for Phase E.

All 4 candidates expected to land at PARK_PENDING_OOS_POWER given identical N_OOS power-floor situation.
Real promotion to Phase E requires N_OOS_on ≥ 50 (Q3-2026 timeframe at current trade rates).

### Decision gate at session-end

Phase B (verification) returned **4 of 4 candidates as PATHWAY_B_ELIGIBLE** — they fail Pathway A discovery DSR (K_family=1850-2700) but pass Pathway B K=1 DSR with theory-citation per Phase 0 Amendment 3.0. All 4 are mechanism-grounded (Chan Ch7 + Fitschen Ch3 for ovn_range_pct_GT80; Carver Ch9-10 for garch_vol_pct_GT70). C8 dir-match and C6 WFE are UNVERIFIED for all 4 (N_OOS = 9-17, below the 50 power floor) — this is the legitimate Phase 0 verdict, not failure.

**Decision needed:** approve Phase D (write Pathway B K=1 pre-regs for D1-D4) or redirect.

### Plan progress (full plan: `docs/plans/2026-04-28-edge-extraction-phased-plan.md`)

- ✅ **Phase A** — Contamination sweep (commit `96bba7a7`)
  - E2 break-bar look-ahead fix in `research/comprehensive_deployed_lane_scan.py`
  - Mode A revalidation of all 59 active validated_setups
  - 18-script E2 LA contamination registry written
  - PR #48 "monotonic-up universal" memory entry flagged TAINTED
  - RULE 16 added to backtesting-methodology-failure-log
- ✅ **Phase B** — Per-candidate verification (this session)
  - DSR Eq.2 + Eq.9 effective-N per cell (Pathway A FAIL, Pathway B PASS)
  - Per-year era stability per cell (all 4 PASS C9)
  - Lane-correlation matrix vs deployed 6 lanes (all |corr| ≤ 0.36, additive)
  - Result: `docs/audit/results/2026-04-28-phase-b-candidate-evidence.md`
- ⏸ **Phase C** — Instrument-family discipline (pending — autonomous)
- ⏸ **Phase D** — Pathway B K=1 pre-regs (REQUIRES YOUR GO)
- ⏸ **Phase E** — Capital integration (REQUIRES YOUR GO + capital-review skill)
- ⏸ **Phase F** — Adjacent edge hunts (deferred until D returns)

### Phase B candidates (all PATHWAY_B_ELIGIBLE)

| ID | Cell | N_IS | ExpR_IS | SR_ann | DSR_PB | Mechanism |
|---|---|---|---|---|---|---|
| B-MES-EUR | MES EUROPE_FLOW O15 RR1.0 long + ovn_range_pct_GT80 | 186 | +0.143 | 0.89 | 0.985 | Chan Ch7 + Fitschen Ch3 |
| B-MES-LON | MES LONDON_METALS O30 RR2.0 long + ovn_range_pct_GT80 | 183 | +0.242 | 0.94 | 0.993 | Chan Ch7 + Fitschen Ch3 |
| B-MNQ-NYC | MNQ NYSE_CLOSE O5 RR1.0 long + ovn_range_pct_GT80 | 160 | +0.219 | 1.53 | 0.999 | Chan Ch7 + Fitschen Ch3 |
| B-MNQ-COX | MNQ COMEX_SETTLE O5 RR1.0 long + garch_vol_pct_GT70 | 199 | +0.245 | 1.69 | 0.999 | Carver Ch9-10 |

### KILLED in this session

- 3 prior-scan "candidates" using `rel_vol_HIGH_Q3` (E2 break-bar look-ahead, 40-43% post-entry data verified)
- 11 prior-scan OOS-flipped survivors (the OOS flips were the look-ahead artifact's signature; 0 flips on clean scan)
- `dow_thu` / `is_monday` / `is_friday` survivors (no mechanism per Aronson EBTA Ch6)
- PR #48 "monotonic-up universal" TAINTED (rel_vol regression on E2 = post-entry data)

### Files added/changed this session

- `research/phase_b_candidate_evidence_v1.py` (new — Phase B evidence script)
- `docs/audit/results/2026-04-28-phase-b-candidate-evidence.md` (new — verdict per candidate)
- `HANDOFF.md` (this update)
- Memory: `MEMORY.md` flag for PR #48 + registry pointer

### How to pick up

1. Read `docs/plans/2026-04-28-edge-extraction-phased-plan.md` (master plan)
2. Read `docs/audit/results/2026-04-28-phase-b-candidate-evidence.md` (Phase B verdict)
3. User decision: approve Phase D pre-reg writing for B-MES-EUR (highest EV) or redirect
4. If approved: write `docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml` per `docs/prompts/prereg-writer-prompt.md`
5. Phase C (instrument-family discipline) can run in parallel — autonomous, no capital touched

### Prior commit context (earlier today)

- **Commit:** 96bba7a7 — research(phase-a): contamination sweep + scan hardening + phased plan

## Session decisions (2026-04-28 — docs lifecycle hardening)

- Added `docs/INDEX.md` as the daily docs front door with a strict 9-entry entrypoint list (authority, workflow, runtime, active plans).
- Applied plan lifecycle metadata frontmatter to all `docs/plans/**/*.md` files: `status`, `owner`, `last_reviewed`, `superseded_by`.
- Restructured plans into date-bounded active/archive topology:
  - active plans now live under `docs/plans/active/2026-04/` (8 files)
  - inactive plans moved under `docs/plans/archive/{2026-02,2026-03,2026-04,undated}/`
  - `*.tasks.json` moved out of root into date-bucketed archive folders
- Added `scripts/tools/list_stale_active_docs.py` to report stale active plan docs by `last_reviewed` age threshold (default 14 days).
- Updated `docs/governance/document_authority.md` with a binding lifecycle policy for plan metadata, active/archive placement, and stale-check command.
- Follow-up refinement: stale-doc scanner now scans only `docs/plans/active/` (not full plan tree) and `docs/INDEX.md` now includes a concise Claude Code daily flow to keep usage project-specific and lightweight.

## Session decisions (2026-04-27 — orphan-branch recovery)

- **Recovered registry hygiene from `codex/control-plane-unify`** (original `9550a668`, 2026-04-24). Manually applied the additive `context/registry.py` subset of the 18-file orphaned commit: registers existing canonical control-plane infrastructure (`docs/runtime/action-queue.yaml`, `pipeline/work_queue.py`, `scripts/tools/work_queue.py`, `pipeline/system_authority.py`, `pipeline/system_context.py`) into FALLBACK_READ_SET + 2 task manifests' canonical_files/doctrine_files; adds `## Control-Plane Truth/Notes/Plane` H2 sections to the 3 markdown render functions. All referenced files VERIFIED present in main. Re-rendered `docs/context/*.md` via `scripts/tools/render_context_catalog.py`. Other 17 files in original commit had heavy CRLF + functional drift; not recovered (separate decision).
- **`codex/control-plane-unify` branch retained locally** — original audit's "DROP — superseded" verdict was wrong on 2 of 3 cited grounds; branch carries unrecovered substantive content; revisit when needed.

## Session decisions (2026-04-27 — sizing-substrate audit closure)

- **Stage-1 sizing-substrate diagnostic SUBSTRATE_WEAK.** 2/6 lanes PASS (EUROPE_FLOW + TOKYO_OPEN, both via rel_vol_session, both UNSTABLE per Carver Ch.7 fn78 → stage2_eligible=False). Tier-A 0/18 STRONG NEGATIVE: deployed binary filters carry no continuous predictive substrate. Pre-reg `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`; result `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`.
- **Institutional code+quant audit (evidence-auditor) returned PASS_WITH_RISKS.** Verdict upheld; 5 load-bearing findings. Closures in this session:
  - Finding A (MED, ATR_P50 vol_norm = raw identity): identity test added; effective unique cells = 42 (not K=48) documented in result MD header.
  - Finding B (LOW, COST_LT12 inline formula): anti-drift equivalence test asserts `derive_features` matches canonical `CostRatioFilter`.
  - Finding C (MED rule violation, pooled-finding YAML absent): front-matter added (`flip_rate_pct: 67.0` lane-level, `heterogeneity_ack: true`).
  - Finding D (MED mandatory rule, RULE 13 pressure-test absent): `feature_temporal_validity` extended for RULE 1.1 (hard-banned: `double_break`, `mae_r`, `mfe_r`, `outcome`, `pnl_r`) + RULE 6.3 (E2 break-bar suffixes via `entry_model=E2` gate, canonical authority `trading_app/config.py:3540-3568`); 4 pressure tests + E1 control test added.
  - Finding E (INFO, monotonicity gate `inc or dec`): verified correct, no fix.
  - Findings F/G (INFO): accurate label / pre-specified design, no fix.
  - **Finding H (LOW, deferred):** pre-reg YAML uses `testing_mode: diagnostic_descriptive` which is not in the canonical {`family`, `individual`} set per `pre_registered_criteria.md`/`research-truth-protocol.md`. Doctrine-formalization gap; doesn't affect operative methodology. Future doctrine review to either (a) add `diagnostic_descriptive` as a third canonical value or (b) revise the pre-reg with `corrigendum.md` subdoc.
- **Cross-walk note:** `rel_vol_session` is the SAME column (`daily_features.rel_vol_{ORB_LABEL}`) used by PR #51's universal-scope monotonic-up regression (β₁=+0.278/+0.330/+0.300; t=+9.6/+11.8/+7.5 across MNQ/MES/MGC at 5m E2 RR1.5). Stage-1's 2 PASS cells are lane-level cuts of that universal base, attenuated by stricter cell gates + Carver fn78. Theory grounded in `fitschen_2013_path_of_least_resistance.md` + `chan_2013_ch7_intraday_momentum.md` (already extracted). Hong-Stein attention cite NOT REQUIRED and would be invented.
- **Doctrine entry — `mechanism_priors.md` §7 PARKED:** continuous-sizing substrate of deployed binary filters. Reopen gate: AFML Ch.19 sigmoid bet-sizer (NOT in `resources/`) + per-lane fresh mechanism citation + new pre-reg.
- **Follow-on routing:** do NOT author Path-β `rel_vol_session`-conditioner pre-reg (duplicates PR #51 at lane scope, breaches cumulative-trial Bailey MinBTL bound). Route to existing PR #51 candidate-activation plan in flight per `memory/amendment_3_2_and_cpcv_parked_apr21.md`.
- **Branch finalization:** `chore/freshness-bumps` was scope-bled with 17 unrelated sizing commits. Split via non-destructive label move: `git branch research/sizing-substrate-stage1-2026-04-27` at HEAD, then `git reset --hard 993daccb` on the freshness branch. All sizing commits preserved on the new branch. 8/8 pre-commit checks green on every commit. Neither branch pushed.
- **Test count:** 29 → 35 (all pass).

## Session decisions (2026-04-27 — orphan-branch recovery)

- **Recovered `_stage_file_is_closed()` helper from `codex/stage-hygiene-active-detection`** (original `eb40b35a`, 2026-04-24, "fix(context): ignore closed runtime stage files"). Adds 2-line call in `_parse_stage_file` + new function — closed/completed stage files (status field set OR `## Execution Outcome` H2 present) drop out of `_list_active_stages`, freeing edits previously blocked by stale closed stages. Pre-checks PASS: zero callers in main, additive only, integration via `_list_active_stages` → `stage-gate-guard.py` + `pipeline/system_brief.py` is sound. Manual cherry-pick (file-level `git checkout` had massive CRLF noise); applied just the substantive 2-line + 1-function diff. Branched from `origin/main` as `recover/stage-hygiene-active-detection`. Original commit's 6 stage-doc frontmatter touches dropped (audit-confirmed those files are gone from main).
- **`codex/stage-hygiene-active-detection` branch retained locally** until recovery PR is merged.

## Session decisions (2026-04-27 — PR #124 re-extraction + cleanup)

- **PR #147 merged (170b6085)** — re-extract of abandoned PR #124 onto fresh main. Original PR #124 (`chore/pass-three-drift`, commit `6c279810`) was based on `cebefd92` (pre-PR-#126 + pre-PR-#130-renormalization); CI failed on the wall-clock-time-bomb test fix landed by PR #126, and local rebase developed 4 production-code conflicts (`pipeline/check_drift.py`, `trading_app/live/alert_engine.py`, `trading_app/live/webhook_server.py`, `tests/test_pipeline/test_work_queue.py`) plus 79 CRLF-only files from missing PR #130 renormalization on its base. PR #124 closed with cross-link.
- **Drift check #120 live** — `check_magic_number_rationale(trading_app_dir)` AST-walks `trading_app/live/` for UPPER_SNAKE_CASE numeric constants `abs(value) > 10` and requires `Rationale:` / `rationale` (case-insensitive) within ±10 lines OR membership in `RATIONALE_WHITELIST` (initially empty). Cites Robert Carver, *Systematic Trading*, Ch. 4 (parameter-justification discipline). 9 existing constants retagged to satisfy.
- **Criterion 11 / 12 control state refresh** — both `valid=False reason=db identity mismatch` cleared via `python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto`. C11 now operational 83.1% / age=0d / paths=10000; C12 5 CONTINUE / 1 ALARM (L4 NYSE_OPEN COST_LT12 SR=33.27 vs threshold=31.96 — pre-existing, expected blocked-lane state).
- **Worktree + branch cleanup** — `canompx3-pt` worktree force-removed (held 79 CRLF-only files + 1 stale HANDOFF auto-update stub, no real WIP); `canompx3-pr124` temp worktree removed after PR #147 merge; local branches `chore/pass-three-drift{,-v2}` deleted; remote branches `chore/pass-three-drift{,-v2}` deleted on origin. 3 obsolete stashes dropped (PR #124 cleanup wash, PR #138 mutex already-landed, PR #135 pre-merge HANDOFF state).
- **PR #99 (Codex DRAFT) standing-instruction acknowledgment** — DEFERRED. Hook treated retry as duplicate-attempt. Standing instruction in HANDOFF still holds: leave for Codex.

## Session decisions (2026-04-26 evening + late-evening sweep)

- **Env hardening landed** (PR #136, merged) — venv pinned to `uv.lock`, `_env_drift_lines()` reports drift at session-start, `requirements.txt` regenerated from lock.
- **Same-worktree mutex landed** (PR #138, merged `e306284b`) — atomic `O_EXCL` lock at `<git-dir>/.claude.pid` prevents two Claudes in one worktree (closes the `cannot lock ref 'HEAD'` race). 4 scenarios tested.
- **Architecture decision: per-session clones REJECTED.** Researched against 2026 industry consensus (Anthropic, Augment Code, Penligent, Upsun) — worktrees + runtime isolation is the recommended pattern. `_discover_git_common_root()` in `pipeline/paths.py:33-65` already implements canonical worktree-shared-DB.
- **Phase 1B / 1C DEFERRED** — cross-worktree push lock and runtime isolation audit, both low-priority/advisory. Not blocking.
- **Hook-output trim DEFERRED** — needs production-code edits to `scripts/tools/task_route_packet.py`; properly stage-gated.
- **Recovery PRs #133/#134 CLOSED** — stash content already preserved at `archive/stash-2026-04-26-l1-europe-flow-pr1a-v2` and `archive/stash-2026-04-26-l6-wip-pre-correction`. PRs closed with archive-tag citations; `recover/l1-europe-flow-prereg` and `recover/l6-us-data-2026-diagnostic` deleted local + remote.
- **Evidence-auditor obligation discharged** — the stage `open-pr-review-debt.md` listed 7 deferred PRs. As of 2026-04-26: #72/#74/#59/#8 CLOSED, #12 MERGED, #36 CLOSED (re-extracted as PR #135 which MERGED). #99 remains DRAFT (Codex). All 6 non-DRAFT entries resolved historically without auditor pass — recorded as discharged, not pending.
- **Conservative zombie sweep** — deleted 4 `[gone]`-flagged local branches (`chore/handover-2026-04-26-pr125-126`, `ralph/burndown-v5.3-rebased`, `ralph/crit-high-burndown-v5.3-high-tier`, `research/pr48-sizer-rule-oos-backtest`). `ralph/crit-high-burndown-v5.2` retained because `canompx3-ralph-burndown` worktree still uses it.

## Stages closed this session (work landed, file deleted)

| Stage file | Marker commit |
|---|---|
| hwm-stage1-gap1-none-reason-contract-guard | `8a40a142` (#129) |
| hwm-stage2-tracker-integrity | `8a40a142` (#129) |
| hwm-stage3-orchestrator-pre-session | `8a40a142` (#129) |
| hwm-stage4-inactivity-monitor | `8a40a142` (#129) |
| hwm-warning-tier-notify-dispatch | `8a40a142` (#129) |
| ralph-iter-174-f4-bracket-naked | `87dffa38` |
| osr-debt-frame-audit | `eaa4e055` (#106) |
| debt-burndown-import-side-effects | `2e9f4145` (#128) |
| live-overnight-resilience-hardening | `e02c529d` (audit-verified `2f45a3e8`) |
| ralph-iter-179-pass-one-hardening | `0d54d52e` (#107) |
| ralph-iter-181-r4-signals-rotation | `36276381` (#110) |
| ralph-iter-185-f7-fill-poller-timeout | `36276381` (#110) |
| hardening-three-fixes | `290006e5` (#137) |
| open-pr-review-debt | (this PR — obligation discharged historically) |
| pass-three-magic-number-drift-check | `170b6085` (#147 — re-extracted from abandoned #124) |

## Active stages (1 remaining)

1. **`pr48-mgc-shadow-only-overlay-contract.md`** (IMPLEMENTATION) — operator observation pending; do not action without observation result.

## Next Steps — Active

1. **PR #99** (DRAFT, Codex's `wt-codex-mnq-hiroi-scan`) — research(mnq) MNQ COMEX geometry hardening + COMEX family stamp. Standing instruction: comment + leave for Codex.

2. **Action queue** — fully closed as of 2026-04-26. `cross_asset_session_chronology_spec` marked CLOSED (spec frozen at `docs/plans/2026-04-25-cross-asset-session-chronology-spec.md` via PR #101 / `87ebc885`); re-opens when a future cross-asset prereg explicitly cites and gates on it. No P1 ready items remain.

3. **Operator action** — MGC shadow observation; result unblocks `mes_q45_exec_bridge` action-queue item.

## Parallel session activity (DO NOT TOUCH)

- **`canompx3`** — `chore/brain-skill` (orphan branch with merged-content commits + ralph-loop session committing here). Ralph terminal active 2026-04-27 — do not write to this worktree.
- **`canompx3-ralph-burndown`** — `ralph/crit-high-burndown-v5.2` (branch `[gone]` on remote; HEAD `6740d938` predates merged Ralph PRs #107/#110). Stale; do not write to it. Local branch retained because of this worktree.
- **`canompx3-ghost`** — checked out on `main` (this handoff PR's worktree); idle after this session.
- **`canompx3-pt`** — REMOVED 2026-04-27 after PR #124 closed.
- **MGC shadow observation** — operator action; observation result unblocks `mes_q45_exec_bridge` action-queue item.

## Blockers / Warnings

- Memory entries calling F-1 "dormant" are FACTUALLY WRONG — F-1 is **active** and fail-closing in signal-only. Day-1 XFA seeds `_topstep_xfa_eod_balance = $0.0` (canonical per `topstep_scaling_plan.py:51-53`); 6 deployed MNQ lanes ALLOW through `risk_mgr.can_enter()`. Verified empirically (E2E probe in PR #100/B6 work).
- `build_live_portfolio()` is **DEPRECATED** — `--all --signal-only` without `--profile` hard-fails all 3 orchestrators. Only `--profile` path works.
- Drift checks pre-existing: #4 (`work_queue.py` schema parser false-positive on `table 'the'`) and #59 (MGC — 1 day with `!= 3 daily_features` rows). Neither blocking.
- Working-tree debt: `requirements.txt` regenerated by `uv export` may be dirty in `canompx3-ghost` (main) worktree. Separate concern; needs its own commit + uv-lock-vs-export reconciliation.
- ~53 zombie local branches remain after 2026-04-26 conservative sweep (4 deleted). Most have merged PRs but no `[gone]` flag because origin still hosts the branch. Aggressive sweep deferred — would benefit from `clean_gone` command + cross-reference of `git branch --list` against `gh pr list --state all`.

## Durable References

- `docs/runtime/action-queue.yaml` — P1/P2 action queue
- `docs/runtime/decision-ledger.md` — durable decisions
- `docs/runtime/debt-ledger.md` — known debt
- `docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md` — HWM Stages 1-4 design (landed in #129)
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` — O-SR grounding (still pending: cusum_monitor → SR Eq 10)
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` — ORB premise
