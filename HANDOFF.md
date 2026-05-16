# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code (Opus 4.7)
- **Date:** 2026-05-16 PM
- **Commit:** e697b5e9 — docs: close MEMORY/canonical reconcile + add live-debut carry-over capture list
- **Files changed:** 2 files
  - `HANDOFF.md` (Next Step 2 sub-item closed)
  - `docs/runtime/next-session-go-live-plan.md` (new carry-over capture list section)
- **Prior:** bf6cda7a (Monday pre-session checklist — path fixes + corrected HWM diagnosis)
- **Session summary:** Sat 2026-05-16 PM, markets closed weekend. Picked up from `35a13e86`: (a) HWM DEFERRED unchanged, (b) live preflight cannot run (markets closed), (c) carry-over operationalized via capture list, (d) confirmed already DONE in `bf6cda7a`. Bonus: closed stale "MEMORY 3 vs canonical 2 reconcile" sub-item — live-verified both surfaces agree at 4 deployed MNQ lanes. Memory: added `feedback_auto_loaded_rules_context_spike_on_read.md` (~30K-token rule injection on file Read).
- **Did NOT start:** rank-3 AUDIT_GAP_ONLY VWAP_MID_ALIGNED_O30 pre-reg authoring (deferred to next session — CTX hit 47% from rule auto-load before design work could begin).

## This Session (2026-05-13 PM)
- Token-efficient code review (Sonnet) found a LOW `BrokerDispatcher.supports_sequential_bracket_ids()` delegation gap — committed `a6e79c6b`. Also refreshed 316 `validated_setups.last_trade_day` rows (2026-05-07 → 2026-05-12) via inline python (Sonnet violated integrity-guardian § 2; canonical migration `scripts/migrations/backfill_validated_trade_windows.py` reproduces identical state; `--dry-run` shows `drifted=0`).
- Adversarial audit (Opus) on the Sonnet commit caught: (a) stale class docstring claiming `BrokerDispatcher` is wired as top-level router (zero production construction sites — Stage 2 multi-broker plan never wired), (b) zero companion tests for either delegation method, (c) inline-python integrity violation. Fixed in `aef0cf2e` (docstring + 4 mutation-proof tests).
- Blast-radius audit on `aef0cf2e` found three more same-class API-parity bugs on `BrokerDispatcher`: `is_degraded` was `@property` but base/orchestrator call it as method (`TypeError` would fire if dispatcher ever wired), `degraded_accounts()` missing override (would silently return `{}`), `verify_bracket_legs()` missing override (would misread as MISSING legs CRITICAL alarm). Fixed in `612bf331` (property→method + 2 overrides + 5 mutation-proof tests).
- Net: 9 new tradovate tests (63→72), all 126 drift checks pass, 247 sibling tests (`session_orchestrator` + `copy_order_router`) green. Class now fully API-aligned with `BrokerRouter` base + `CopyOrderRouter` peer. Production unaffected — `BrokerDispatcher` has zero live callsites.
- Memory: `feedback_code_review_dead_class_detection.md` added (grep for `ClassName\(` construction sites before grading dead-code severity).

## This Session (2026-05-16)
- **Tool:** Claude Code (Opus 4.7)
- **Date:** 2026-05-16 (Sat BNE / Fri 15:22 CT)
- **Summary:** First real-money `topstep_50k_mnq_auto` MNQ live session. Preflight 7/7 (broker auth, portfolio load, daily features, contract resolution, notifications, journal, copy-trading dry-run). Bot connected to ProjectX Market Hub, subscribed to MNQ quotes, ran ~38min in wait-for-bar before `Ctrl+C`. Zero trades — all 4 lane session windows had passed by start time; 3/4 lanes also BLOCKED by Criterion 12 SR alarms (1 PRIME_SHADOW: US_DATA_1000).
- **Status:** Rig wired correctly end-to-end. No exceptions, no broker drops, no risk-manager fires, clean shutdown. Capital outcome: $0 P&L.
- **Verification:** Preflight self-tests `notifications PASS / brackets PASS / fill_poller PASS`. `is_market_open_at` correctly resolved Friday RTH-late as OPEN. `Daily features row: atr_20=321.875, atr_vel=Stable`. F-1 XFA scaling active.
- **Observations for next session:** (a) Dashboard `/api/bars-recent?instrument=MNQ` returned `"bars":[]` — chart panel renders empty despite feed connected; likely tick→1m aggregation handoff bug. NOT capital-control. (b) HWM file (`data/state/account_hwm_21944866.json`) timestamp is fresh (2026-05-15T20:46:01Z) but `hwm_dollars=0.0` / `last_equity=0.0` — tracker shell exists but was never populated with broker equity during the 2026-05-16 debut. Operator-visible concern: "never populated", not "stale". Equity-population path investigation DEFERRED — run one real Monday session first; revisit only if `hwm_dollars` remains 0.0 after broker activity. (c) Bot did not write a `logs/live/live_<ts>.log` file — output was stdout-only; canonical plan's "tail the log file" instruction was never validated against a real `--live` run. CARRY-OVER: both (a) and (c) need ≥1 more live run to characterize before a fix stage is justified.

## Next Steps — Active
1. **MGC LONDON_METALS — DO NOT RE-LITIGATE.** Verdict frozen at `docs/audit/results/2026-05-12-mgc-london-metals-mode-a-k1-revalidation.md`. Reopen only if new evidence clears one of the prereg kill criteria (K1 t_IS≥3.00 with theory grant, or K3 N_IS_on≥100). Do not re-run Phase A on alternative apertures as a back-door — that pattern is the trap.
2. **Highest-EV next is MNQ.** Live: 4 deployed MNQ lanes per `docs/runtime/lane_allocation.json` (verified 2026-05-16). Concrete candidate: rank-3 AUDIT_GAP_ONLY VWAP_MID_ALIGNED_O30 pre-reg authoring per Chordia v2 readouts. (Prior "MEMORY 3 vs canonical 2 — reconcile" sub-item RESOLVED 2026-05-16: both surfaces now agree at 4 post-Chordia-K=20 rebalance per `memory/live_lanes_2026_05_14_four_deployed_post_chordia_k20.md`.)
3. **Pre-existing carry-over (still open):** Track D MNQ COMEX_SETTLE Gate 0 runner design (Databento top-of-book table + bounded runner for DESIGN_ONLY prereg); deployment-coverage decision on 78 ROUTABLE_DORMANT strategies (`docs/audit/results/2026-05-12-deployment-coverage-orphans.md`).
4. **NUGGET 5 PARKED 2026-05-13.** Agent-control-plane evaluation (Paperclip / amux / Cogpit / OctoAlly / LONA / reasoning sidecar) marked PARKED in `docs/plans/2026-05-12-agent-control-plane-evaluation.md`. Reopen only if worktree/branch/PR cleanup exceeds 2 hrs/week for two consecutive weeks. Existing worktree-manager + 5 MCPs + 11 subagents + 27 skills + 17 hooks already constitutes a control plane; NUGGET 4 (commit `b90c6291`) addressed the actual bottleneck (session-start context load). Do not re-evaluate without the reopen trigger firing.
5. **Monday pre-session checklist (BEFORE first real MNQ trade window opens):**
   (a) HWM tracker for account 21944866: file timestamp is fresh (2026-05-15T20:46:01Z) — NOT 20.6d stale. The real defect is `hwm_dollars=0.0` / `last_equity=0.0` (shell created but never fed broker equity). DEFERRED: do not investigate equity-population path until one real Monday session has completed; revisit only if `hwm_dollars` remains 0.0 after broker activity. No pre-session action required.
   (b) Re-run `python scripts/run_live_session.py --instrument MNQ --profile topstep_50k_mnq_auto --preflight --signal-only` — expect "Preflight: 7/7 passed". Operator-run; requires live broker auth.
   (c) CARRY-OVER (open, deferred): two log-surface gaps from 2026-05-16 debut still need ≥1 more live run to characterize before a fix stage is justified — (i) `/api/bars-recent` returns `[]` despite feed connected (chart panel empty — likely tick→1m aggregation handoff), (ii) bot did not write `logs/live/live_<ts>.log` to disk under `--live` (output was stdout-only). Non-blocking for trading.
   (d) DONE 2026-05-16: patched `docs/runtime/next-session-go-live-plan.md` for the 3 audit-caught path errors (`data/lane_allocation.json` → `docs/runtime/lane_allocation.json`; `logs/session.log` → `logs/live/live_<ts>.log`; stale commit anchor `5dd1a822` → `8c7786cb`).
   (e) **Monday coverage strategy = A (single long-running session)** per baton plan `C:\Users\joshd\.claude\plans\get-going-on-this-whimsical-rain.md` § 0.4. Launch `python scripts/run_live_session.py --instrument MNQ --profile topstep_50k_mnq_auto --live` before 23:25 BNE Mon, leave running through ~03:40 BNE Tue. Covers all 3 windows in one process. Paste the one-liner from `docs/runtime/next-session-go-live-plan.md` § One-shot Monday evidence-capture at T+3min and on any anomaly.
6. **Phase 1 entry condition (post-Monday).** Phase 0 grounding (this weekend) is COMPLETE: lanes verified, ProjectX spec extract at `resources/projectx_api_spec_2026_05_16.md` written, TopStep rules confirmed, evidence-capture one-liner appended. **Before starting Phase 1 implementation: `/clear` first**, then read in order — (1) `C:\Users\joshd\.claude\plans\get-going-on-this-whimsical-rain.md`, (2) this HANDOFF.md, (3) `docs/runtime/sessions/<Monday-date>-live-debut-followup.md` (Monday evidence), (4) `resources/projectx_api_spec_2026_05_16.md`, (5) `resources/prop-firm-official-rules.md` § TopStep. Then write `docs/runtime/stages/phase1-live-pipeline-hardening.md` and begin Phase 1.1 (D2 file logging — HIGH, the lead item).

## This Session (2026-05-16 PM v2) — START_BOT preflight reliability fix

- **Tool:** Claude Code (Opus 4.7)
- **Status:** UNCOMMITTED, READY TO COMMIT. 272/272 pass on touched surface. Drift unchanged at 6 pre-existing (none mine).
- **Stage:** `docs/runtime/stages/start-bot-reliability-minimal.md` — IMPLEMENTATION, 5-file scope-lock.
- **Files modified (5):**
  - `scripts/run_live_session.py` — replaced stubbed `results["brackets"]=True`/`results["fill_poller"]=True` with `_probe_brackets(components)` / `_probe_fill_poller(components)`. Mirrors `SessionOrchestrator._verify_brackets` / `_verify_fill_poller` line-by-line (account_id=0 sentinel, NotImplementedError = only FAIL signal). `_check_notifications` threads `ctx.components` and surfaces `brackets:PASS/FAIL · fill_poller:PASS/FAIL` in inline summary.
  - `trading_app/live/session_orchestrator.py` — narrowed three `except Exception` blocks (lines 361-378 ORB caps; 383-392 max_risk_per_trade; 399-417 lane_allocation regime gate) to explicit class tuples; preserved profile-account `raise`. Block 1 post-audit expanded to include `FileNotFoundError, OSError, json.JSONDecodeError` for future-proofing against `load_allocation_lanes` refactors.
  - `tests/test_scripts/test_run_live_session_preflight.py` — +12 tests (probe paths + summary visibility + no-hardcoded-stubs source grep).
  - `tests/test_trading_app/test_session_orchestrator.py` — +7 tests in `TestSafeguardExceptNarrowing` using load-block-replay pattern (matches existing `test_per_aperture_load_path_end_to_end_with_real_profile`). Profile/non-profile KeyError + malformed JSON + missing strategy_id + KeyboardInterrupt/SystemExit propagation. Added `import json` at top.
  - `tests/test_trading_app/test_bot_dashboard.py` — single test updated for new `components` kwarg contract. DB-free guarantee preserved.
- **Audit:** `evidence-auditor` (independent context, per `.claude/rules/adversarial-audit-gate.md`) verdict CONDITIONAL → CLOSED. Critical finding (Block 1 missing JSON/OS classes) addressed; other claims (probe semantics, router `__init__` safety at account_id=0, no hidden callers, `raise` preservation, test quality) passed.
- **Suggested commit:** `fix(preflight): real bracket/fill-poller probes + narrow safeguard excepts` — judgment classification, audit already ran.
- **Phase 2 (dashboard polish) was gated on this audit verdict. Green to start after commit + `/clear`.**
- **Nuggets noted (NOT actioned, drift-risk avoidance):**
  1. Drift check: probe ↔ verifier parity diff.
  2. Drift check: Block 1 except tuple ⊇ `get_lane_registry` transitive raise classes.
  3. Operator-visibility: `WARNINGS (…)` still counts as `passed=True`; bump to `False` for profile accounts only.
  4. **Trading-edge (`resources/`-grounded, hypothesis-only):**
     - Fill-poller as live slippage telemetry (Harris 2002 Ch 14 §14.2) — `SessionStats.fill_polls_*` counters exist, unsurfaced.
     - `_regime_paused` is read-once at `__init__` — Carver Ch 11 treats allocation as continuous signal; periodic re-read on mtime.
     - `_orb_caps` symmetric long/short — Fitschen Ch 3 + Yordanov NQ ORB suggest directional asymmetry; one-shot P90 scan.
     - Broker-reachability discrimination (Aronson EBTA) — `query_order_status(0)` failure-class surface to dashboard.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
