Classify the current task and control stage execution: $ARGUMENTS

Use when: starting non-trivial work, mode unclear, or stage-gate-guard hook blocked an edit.
Triggers: "stage-gate", "classify", "what mode", "gate check", "what stage am I in"

## STEP 1: CHECK EXISTING STATE

Read `docs/runtime/STAGE_STATE.md` if it exists.
- Active stage? → Ask: "Continue [stage description], or reclassify?"
- No file or stale? → Proceed to Step 2.

Stale detection (drift-first):
1. If file exists, check `git log --oneline --since="[updated timestamp]" -- [scope_lock files]`
2. Any commits touching scope files since last update? → STALE (drift)
3. No drift but >4 hours since update? → STALE (age fallback)

## STEP 2: TRIVIAL CHECK

Is this task trivial? ALL conditions must be true:
- Touches ≤2 files
- Changes are mechanical (typo, comment, rename, format, single bug fix)
- No config cascade, no schema change, no new canonical source usage
- Acceptance is obvious ("it compiles" / "test passes")

**NEVER TRIVIAL — these files always require full staging:**
- Pipeline core: build_daily_features, build_bars_5m, ingest_dbn, init_db, run_pipeline
- Config/session/paths: dst.py, asset_configs.py, cost_model.py, paths.py, config.py
- Validation: strategy_discovery, strategy_validator, outcome_builder, entry_rules
- Live trading: live_config.py, trading_app/live/*
- Pipeline gates: check_drift.py, health_check.py
- Protected scripts: build_edge_families, audit_behavioral

If any target file is in the NEVER TRIVIAL list → skip to Step 3 (full classification).

If genuinely trivial → write minimal state and stop:
```yaml
# docs/runtime/STAGE_STATE.md
---
task: "[one-liner]"
mode: TRIVIAL
scope: [file1.py]
updated: [ISO timestamp]
terminal: [main|worktree-name]
---
```
Done. Hook will pass for non-core files. No further ceremony.

## STEP 3: CLASSIFY MODE

Exactly ONE of:

| Mode | Signal |
|------|--------|
| REBASE/RESUME | Returning to interrupted work, "where was I", stale state |
| TRUTH AUDIT | Need to verify what's real before deciding anything |
| DESIGN | Planning, exploring, "how should we", "what if", "4t" |
| IMPLEMENTATION | Ready to code: "build it", "go", "do it", plan is approved |
| VERIFICATION | Checking work: "verify", "test", "did it work" |
| TOO BROAD | Fails the breadth check below |

## STEP 4: BREADTH CHECK (non-trivial tasks only)

Count truth domains touched:
- [ ] Code (pipeline/, trading_app/ Python)
- [ ] DB/Data (gold.db schema, row contents, staleness)
- [ ] Config (SESSION_CATALOG, COST_SPECS, ACTIVE_ORB_INSTRUMENTS, live_config)
- [ ] Artifacts (model bundles, pipeline outputs, cached files)
- [ ] Docs/Shared (TRADING_RULES.md, HANDOFF.md, memory, specs)

**TOO BROAD if 2+ domains AND any of:**
- Non-local dependency (change in domain A forces change in domain B)
- Unclear acceptance (can't define "done" independently per domain)
- Unresolved canon/policy (rules still being debated)
- Downstream contamination (assuming later-stage output during upstream work)
- Multiple independently testable deliverables (could be separate commits)

If TOO BROAD → dispatch /task-splitter. Stop here.
If 2+ domains but tightly coupled with clear joint acceptance → proceed. Note the coupling.

## STEP 5: DISPATCH + APPROVAL FLOW

| Mode | Action |
|------|--------|
| REBASE/RESUME | → /resume-rebase |
| TRUTH AUDIT | → dispatch preflight-auditor agent (standalone) |
| DESIGN (quick, ≤2 stages) | → dispatch planner agent → present plan → **wait for approval** |
| DESIGN (full architecture) | → /4t or /brainstorm (they write STAGE_STATE on approval) |
| IMPLEMENTATION | → preflight-auditor → if CLEAR → Step 6 |
| VERIFICATION (stage checkpoint) | → /verify done (reads acceptance from STAGE_STATE) |
| VERIFICATION (pre-commit) | → /verify quick |
| VERIFICATION (deep audit) | → /verify full |
| TOO BROAD | → /task-splitter → present Stage 1 → **wait for approval** |

### DESIGN → APPROVAL → STATE WRITE

1. Planner returns structured stage output (not prose)
2. Present Stage 1 to user exactly as planner structured it
3. **Wait.** User must say "go", "approved", "do it", "looks good", or equivalent.
4. **IMMEDIATELY on approval:** write the approved stage to `docs/runtime/STAGE_STATE.md` using full schema:
   - `mode: IMPLEMENTATION`
   - `stage_purpose:` from planner output
   - `scope_lock:` from planner's file list — exact paths
   - `acceptance:` from planner's acceptance criteria — exact commands
   - `proven:` / `unproven:` / `blockers:` carried from planner output
5. Confirm: "Stage [N] approved and locked in STAGE_STATE.md. Scope: [files]. Running preflight."
6. Dispatch preflight-auditor. If CLEAR → proceed to execution. If BLOCKED → stop, report.

**The structured STAGE_STATE.md IS the approval record.**
**Do NOT proceed to implementation until STAGE_STATE.md contains the approved stage.**

## STEP 6: STAGE COMPLETION CHECKPOINT

When stage work is done:
1. Show evidence (command output, not claims)
2. Update STAGE_STATE.md: mark current stage DONE, describe next stage
3. **Commit policy:** commit STAGE_STATE.md WITH the stage's code changes in the same commit.
   - Do NOT commit STAGE_STATE.md alone
   - Do NOT commit on every in-flight update — only at checkpoints
4. **STOP. Do not auto-proceed.**
5. User says "next" → write next stage to STAGE_STATE.md → preflight → execute
6. If ALL stages are done → delete `docs/runtime/STAGE_STATE.md` (cleanup — prevents stale state lingering)

## HARD RULES
- No implementation without an approved stage in STAGE_STATE.md
- No editing files outside scope_lock
- No skipping preflight for non-trivial implementation
- No auto-proceeding past a checkpoint
- If user says "just do it" → still write TRIVIAL or minimal STAGE_STATE, preflight can be skipped
- STAGE_STATE.md commit policy: update freely, commit only at checkpoints bundled with code
