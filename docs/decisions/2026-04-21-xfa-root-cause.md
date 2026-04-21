# XFA / F-1 Hard Gate Root-Cause — 2026-04-21

**Worktree:** `deploy/live-trading-buildout-v1` (Phase 2 of directive)
**Stage:** Workstream A step 1 (Task #8)
**Verdict:** **MEMORY FRAMING IS WRONG.** F-1 is NOT code-dormant. The bot is deliberately running in `signal_only=True` mode by safest-default CLI policy. Live flip requires (a) explicit `--live` CLI flag, (b) interactive "CONFIRM" entry, (c) broker credentials in env, (d) real broker account bound to profile.

---

## Memory claim tested

Memory `MEMORY.md`:
> "Live: topstep_50k_mnq_auto, 2 copies, 6 lanes, signal/demo against TopStep 20092334. Real XFA NOT connected (F-1 hard gate dormant)."

Directive framing echoed this: "F-1 hard gate off dormant."

## What the code actually shows

### F-1 canonical code is LIVE-WIRED (not dormant)

**Canonical source:** `trading_app/risk_manager.py` (L36–L261) + `trading_app/topstep_scaling_plan.py` + `trading_app/live/session_orchestrator.py` (L46–L130, L336–L349, L578–L580, L1240).

1. `RiskLimits.topstep_xfa_account_size` (L42) — canonical F-1 knob. When `None`, F-1 is disabled; when set (e.g. `50000`), F-1 ladder is enforced.
2. `session_orchestrator._resolve_topstep_xfa_account_size(prof)` (L68) — reads `prof.is_express_funded` + `prof.account_size` from `prop_profiles.ACCOUNT_PROFILES` and returns the XFA size automatically.
3. `SessionOrchestrator.__init__` (L336–L349) passes that value into `RiskLimits`, so F-1 activates for every XFA-flagged profile.
4. `_apply_broker_reality_check` (L94–L130) runs at first bar, queries broker account metadata, and calls `risk_mgr.disable_f1(reason)` if the live account is actually a TC (Trading Combine) or missing metadata. This is the auto-disable feature from commit `306d16a0`.
5. `RiskManager.can_enter` (L194–L261) — the runtime F-1 check. Fail-closed: refuses entry when EOD XFA balance is unknown, when projected lots exceed the ladder cap, or when ladder lookup fails.

### Git history — F-1 has been LIVE-READY since 2026-04-14

Selected commits on `origin/main` (reachable from `f567cfe6`):

| Commit | Date range | Summary |
|---|---|---|
| `13700958` | 2026-04-11 | audit: Criterion 11 F-1 BLOCKER is a FALSE ALARM (per-trade ceiling bug) |
| `527ac13b` | 2026-04-11 | fix(scaling-plan): close F-1 false alarm — canonical per-instrument contract aggregation |
| `f9f6683d` | pre-2026-04-14 | fix(scaling-plan): F-1 enforce TopStep XFA Scaling Plan ladder per-day [stage 7/8] |
| `5fae4d0c` | 2026-04-14 | feat(risk): wire F-1 TopStep XFA Scaling Plan + fix safety lifecycle persistence |
| `ea039194` | 2026-04-14 | docs(handoff): F-1 wiring + refresh_data fix shipped 2026-04-14 |
| `ebc2b30f` | 2026-04-15 | fix(risk): skip F-1 EOD balance refresh when orphans present at rollover |
| `306d16a0` | 2026-04-15 | feat(risk): detect TC vs XFA broker account and auto-disable F-1 for TC |

HANDOFF excerpts confirm this: "F-1 is now LIVE-READY" (L3558), "F-1 is now fully broker-aware" (L3455), "239 F-1 scope tests pass" (L3466).

### What IS dormant is the bot's mode flag, not F-1

`scripts/run_live_session.py` L412–L415:
```python
# Default to signal-only if no mode specified (safest default)
if not args.signal_only and not args.demo and not args.live:
    log.info("No mode specified — defaulting to --signal-only (safest)")
    args.signal_only = True
```

`scripts/run_live_session.py` L427–L438 (the `--live` branch):
```python
else:  # --live
    if args.all:
        print("--all + --live not supported. Use --instrument X for live trading.")
        sys.exit(1)
    if not args.auto_confirm:
        confirm = input("\n⚠  LIVE MODE — real money orders will be placed.\n   Type CONFIRM to proceed: ").strip()
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(0)
    else:
        log.warning("LIVE MODE — auto-confirmed (dashboard launch)")
    signal_only = False
    demo = False
```

`SessionOrchestrator.__init__` (L164–L177) receives `signal_only: bool`. Downstream effects when `signal_only=True`:
- No `order_router` is constructed (L1099-style branches; see explicit invariant check L1817–L1819).
- No fills are routed to broker (L1776 + L2032 + L2199 early returns).
- Orders never touch a real account.

Dashboard path (`trading_app/live/bot_dashboard.py` L1484) passes `--signal-only` explicitly; default user-launched sessions that omit `--signal-only`/`--demo`/`--live` fall into the safest-default arm above.

### Broker credentials — unverified state

`trading_app/live/rithmic/auth.py` L49–L63 reads `RITHMIC_USER` / `RITHMIC_PASSWORD` / `RITHMIC_APPKEY` from env. If absent, a broker factory call would fail. This worktree does **not** have those env vars set (per shell env), so even flipping `--live` would not actually connect.

**Not verified in this audit:** whether the production machine / systemd service / docker image that runs the bot has Rithmic env vars set. Cannot verify from a local worktree.

---

## Classification per directive

Directive § Phase 2.1 asks: "is it (a) a code gap, (b) a flag/config switch, (c) a credential issue, (d) a risk gate deliberately paused, (e) an ops dependency on broker build, or (f) unknown?"

| Class | Present? | Evidence |
|---|---|---|
| (a) code gap | **NO** | F-1 fully wired since 2026-04-14; 239 tests pass; broker-aware auto-disable since 2026-04-15 |
| (b) flag/config switch | **YES (primary)** | `--live` / `--demo` / `--signal-only` CLI arg; signal-only is the safest default |
| (c) credential issue | **LIKELY (contributory)** | `RITHMIC_USER` / `RITHMIC_PASSWORD` / `RITHMIC_APPKEY` required; production env state not verified from this worktree |
| (d) risk gate deliberately paused | **YES (co-primary)** | `signal_only=True` default is an explicit ops safety choice; `--live` path requires human "CONFIRM" input |
| (e) ops dependency on broker build | **PARTIAL** | Broker code exists (`trading_app/live/rithmic/`, `projectx/`, `tradovate/`); account binding to real XFA is the unfinished ops step, not a code dependency |
| (f) unknown | NO | all routes traced |

**Net:** root cause is (b) + (d) + contributory (c). Live orders have three independent gates:
1. User chooses `--live` explicitly (not the default).
2. User types literal "CONFIRM" (or uses `--auto-confirm` via dashboard).
3. Broker credentials are loaded into the runtime environment.
4. The target profile's `is_express_funded=True` implies F-1 will auto-enforce per ladder; auto-disable handles TC accounts.

F-1 itself is ready. What is waiting is an ops decision + credential provisioning, not a code change.

---

## Directive abort trigger

Directive § Phase 2 abort rules:
> "If root cause = (c) credentials or (d) deliberate pause, STOP workstream A and surface — don't fake a connection."

Root cause includes (d) deliberate-pause primary and (c) credentials contributory. **Both abort conditions are present.** Workstream A HALTS at the end of Phase 2. Phase 4 (Task #9 — A2 XFA wiring implementation) does NOT execute in this autonomous run.

---

## Material correction to repo memory

The following are factually wrong in current docs/memory and should be corrected by the user the next time those files are edited:

1. `MEMORY.md` line: "Real XFA NOT connected (F-1 hard gate dormant)." → **F-1 is NOT dormant. F-1 is live-ready. The bot is deliberately in signal-only mode.** The correct framing: "Bot in signal_only mode by default (safest-default CLI policy). Live flip requires `--live` + human CONFIRM + Rithmic env creds. F-1 scaling plan auto-enforces for XFA profiles when live, auto-disables for TC accounts per broker-reality check."
2. Any cross-referenced "F-1 XFA dormant" references in `chatgpt_bundle/04_DECISION_LOG.md`, `chatgpt_bundle/06_RD_GRAVEYARD.md`, `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md`, `docs/audit/2026-04-15-topstep-scaling-reality-audit.md` should be updated — they are stale relative to the 2026-04-14/15 wiring sprint.

This is a MEMORY DRIFT discovery. No memory write is performed by this deploy-live agent; corrections are user-owned.

---

## What this means for the deploy-live buildout

- **Workstream A (Task #7):** HALTED at end of Phase 2. Phase 4 does NOT run this session. User must decide, outside this autonomous run:
  - Whether to flip `--live` on a specific profile.
  - Whether to provision Rithmic / TopStep-approved-copier credentials on the production machine.
  - Whether TopStep LFA routing is bound through Rithmic or Tradovate (NOT ProjectX — Stage 1 critical finding).
- **Workstream C (Tasks #10–13):** Can proceed with **C1 (Task #11) — PR #48 gate audit** because C1 is a pure data / doc exercise, gated only on itself. C2 (Task #12 shadow wiring) is still gated on Phase 4, so it will also remain blocked after C1 completes.
- **Workstream B (Tasks #15–21):** Can proceed with credential-independent sub-tasks:
  - B1 (Task #16 Rithmic auth wiring) has the same credentials dependency — **also halted**.
  - B2 (Task #17 routing) is a routing-table / config file exercise, **can proceed** — it consumes Stage 1 prop-rules findings, including the TopStep LFA-via-ProjectX ban.
  - B3/B4/B5/B6 downstream of B1 — halted.
- **Workstream D (Task #14):** out of scope.
- **Workstream E (Tasks #2–6):** completed.

Autonomous advance path, given this verdict: **→ Phase 3 (C1 PR #48 gate audit)**. Then Phase 5.1b (B2 routing) if there's runway. No Phase 4, no Phase 5.1a, no Phase 6.

---

## Open items for the user

1. Confirm production-machine Rithmic / broker credentials status.
2. Decide whether to flip a profile to `--live` (and which one — `topstep_50k_mnq_auto` is the memory-referenced signal/demo profile).
3. Confirm TopStep LFA routing choice (Rithmic or Tradovate — NOT ProjectX).
4. Update MEMORY.md and stale F-1-dormant references per § Material correction.
5. After those decisions, re-enter the deploy-live worktree for workstream A Phase 4 + B1 — but those tasks are no longer autonomous-safe; they require ops gatekeeping.

---

## Self-review

- [x] Every code quote above references a specific file + line range. All quotes verified in this worktree at commit `4119f8e9`.
- [x] Git history verified via `git log --oneline --all | grep -iE "f-1|xfa|scaling plan"` on this worktree.
- [x] The memory-vs-reality contradiction is surfaced explicitly, not silently routed around.
- [x] Abort trigger is applied per directive letter. No workstream A wiring is attempted.
- [x] No training-memory fallbacks used. Every claim cites either code, commit, or file path.

**Certificate status:** N/A (Phase 2 produces a decision doc, not a G1–G10 gate certificate).
