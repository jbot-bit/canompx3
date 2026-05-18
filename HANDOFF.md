# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session (2026-05-18 PM — dashboard Start-Signal preflight mode threading)
- **Tool:** Claude Code (Opus 4.7)
- **Date:** 2026-05-18 (Sun BNE — Monday-eve)
- **Commits pushed this session:** `45c1ffb4` skills frontmatter migration (was unpushed locally — rode this push), `bd229c67` fix(dashboard): thread mode into Start preflight so signal-only path is not gated by live telemetry maturity. Tip is now `bd229c67`. Origin clean (0/0) at session end.
- **Files changed:** `trading_app/live/bot_dashboard.py` (single-file scope, +12/-5), `tests/test_trading_app/test_bot_dashboard.py` (+90 lines, 3 new tests), `docs/runtime/stages/2026-05-18-dashboard-start-signal-preflight-mode.md` (new stage file, committed alongside).
- **Session summary:** `/next` resumed the open IMPLEMENTATION stage. `_run_preflight_subprocess(profile)` now takes `mode="live"|"signal"` and appends `--signal-only` when mode=="signal"; `_prepare_profile_for_start` threads `mode` through; `action_start` (which already has `mode`) passes it; `action_preflight` (ad-hoc dashboard button) keeps live-mode default — **intentional asymmetry**. **Why:** `_check_telemetry_maturity` in `scripts/run_live_session.py:369-378` auto-passes when `ctx.signal_only=True` and fail-closes otherwise, so Start Signal was being blocked by the very gate signal-only mode is meant to clear. 3 new tests cover live-mode omission, signal-mode insertion (and arg ordering after `--preflight`), and helper-to-subprocess propagation. Pre-existing 2 `_derive_operator_state` failures + Check 101 drift verified on stashed baseline — not regressions of this commit.
- **Stage NOT closed:** `docs/runtime/stages/2026-05-18-dashboard-start-signal-preflight-mode.md` remains open. Criteria 1-4 met with positive test coverage; criteria 5-6 show no regression vs baseline; criterion 7 (operator hits `POST /api/action/start?mode=signal&profile=topstep_50k_mnq_auto`, confirms non-blocked status + `logs/live/live_signals_2026-05-18.jsonl` appears within 30s) requires a live dashboard run.

## Last Session
- **Tool:** Claude Code (Opus 4.7)
- **Date:** 2026-05-18 (Sun BNE — Monday-eve)
- **Commits:** `7877f47b` refresh lane_allocation.json (4→3 MNQ lanes, fresh rebalance), `ae3cee57` close stage file.
- **Files changed:** `docs/runtime/lane_allocation.json` regenerated via canonical allocator script.
- **Session summary:** User invoked `/next` asking for "more than 4 lanes for Monday". Investigation rejected three misframes (add sessions to profile, batch-chordia 715 paused lanes, lower MIN_TRAILING_N=20). Ran fresh canonical rebalance (rebalance_date 2026-05-14 → 2026-05-18). **Lane count went DOWN: 4 → 3.** Removed: ORB_VOL_2K, VWAP_MID_RR1.0_O15, COST_LT12, OVNRNG_25. Added: OVNRNG_100 (ExpR 0.2159), VWAP_MID_RR1.5_O15 (ExpR 0.2416), COST_LT12 (kept). **Criterion 8 OOS gate (drift check #149) caught a silent failure on `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25` that the stale 2026-05-14 file was masking** — would have routed real capital at an unsafe lane Monday. All 3 deployed lanes are PASS_CHORDIA/PASS_PROTOCOL_A, status_reason="Session HOT". 133/133 drift checks pass.
- **Honest answer to "more lanes":** the allocator's truthful number today is 3, not 4. Going to 4+ requires either fresh chordia replays on selected paused MNQ lanes (slow doctrine path, hours per pre-reg) or gate relaxations (capital-class doctrine violation, refused). Per-lane EV × correctness > lane count.

## Prior Session (2026-05-17 Codex — preventive allowlist)
- **Commit:** `e37fce01` — chore(profile): preventive allowlist expansion (NYSE_CLOSE + LONDON_METALS) for topstep_50k_mnq_auto
- **Files changed:** `trading_app/prop_profiles.py` (active MNQ profile session allowlist + notes metadata)
- **Session summary:** Preventive allowlist housekeeping — expanded `topstep_50k_mnq_auto.allowed_sessions` to include `NYSE_CLOSE` and `LONDON_METALS` so that future Chordia/regime/doctrine unlocks in those sessions will not be silently vetoed by the profile allowlist. Verified: zero MNQ NYSE_CLOSE/LONDON_METALS entries currently in `docs/runtime/lane_allocation.json::lanes[]`. Net new tradeable strategies that day: 0.

- **Tool:** Claude Code (Opus 4.7)
- **Date:** 2026-05-17 late evening
- **Tip:** c0fb8a19 (audit deployment-coverage rebalance refresh 2026-05-17, annual_r rerank)
- **Prior unpushed → pushed this session:** ff1f13ee (hysteresis aperture DD bug + canonical paused-set parser + fail-closed precondition on corrupt JSON) and 7624656b (work_queue render-handoff --write requires --force; pulse warning no longer recommends footgun). Both code-reviewed (Grade A-) before push.
- **Live preflight result (real broker APIs):** `python scripts/run_live_session.py --instrument MNQ --profile topstep_50k_mnq_auto --preflight --signal-only` → 7/7 PASS. Token acquired, 4 lanes loaded, daily features fresh (atr_20=323.675, vel=Stable, dow=6), contract resolved `CON.F.US.MNQ.M26` (MNQM6 = June 2026 front month), bracket+fill-poller probes PASS, TradeJournal opens. Step 7 SKIPPED (signal-only mode bypasses copy-trading account resolution — needs non-signal-only run before clicking Start Live).
- **Capital-class hardening landed (ff1f13ee):** session_orchestrator now delegates paused+stale parsing to `prop_profiles.load_paused_strategy_ids` (single drift surface vs lanes[] parser); profile_* accounts hard-fail on missing OR corrupt `lane_allocation.json` instead of silently routing blocked strategies live; hysteresis session_key in lane_allocator includes orb_minutes (was charging dd_used with wrong-aperture lane_dd across O5/O15/O30 swaps).
- **No DB mutation. No allocator file mutation.** Code + tests + push only.

## Next Session — Start here (2026-05-19, Monday BNE)

**Current truth (verified end of 2026-05-18 session):**
- `docs/runtime/lane_allocation.json`: rebalance_date `2026-05-18`, **3 MNQ deployed lanes** (OVNRNG_100, VWAP_MID_RR1.5_O15, COST_LT12), 833 paused, 8 displaced.
- Monday capital decision: **STILL RED (BLOCK_LAUNCH_COPY_SET_UNVERIFIED)** until Option A/B/C below. The lane refresh did NOT change the broker mismatch; it only changed which lanes would route IF launched.

### Highest-EV pending operator action (do FIRST)
- **Option A — Provision the second TopstepX account** (still the cleanest unblock; full detail at "Monday-morning decision" block further down).
  - Log into TopstepX → activate a second funded account under the same API credentials.
  - Re-run `python scripts/run_live_session.py --instrument MNQ --profile topstep_50k_mnq_auto --preflight` (non-signal-only). Expect `OK (copies=2, 2 accounts discovered)`.
  - Then launch dashboard → Start Live on `topstep_50k_mnq_auto`. Confirm `Copy trading: primary=<id>, shadows=[<id>]` log line.
  - Result: GREEN, three lanes route Monday 23:25 BNE onward.
- **Option B fallback** if Option A not provisioned by 23:00 BNE: edit `trading_app/prop_profiles.py:481` `copies=2` → `copies=1` (stage file required, production code edit). YELLOW launch on single account, lose copy-trading regression coverage until Option A later.
- **Option C** (launch as-is, copies=2 with 1 account): not recommended — silent degrade, institutional-rigor § 4 + § 6 violation.

### What to NOT do
- Mutate `docs/runtime/lane_allocation.json` further (just refreshed today; next refresh on operator demand or after Monday session).
- Try to "get to 4 lanes" by relaxing gates. The 3-lane number is what the canonical allocator + Chordia + C8 + correlation pruning produced; chase quality not count.
- Re-litigate MGC LONDON_METALS (verdict frozen — see "Next Steps — Active" item 1 further down).
- Re-litigate the 78 ROUTABLE_DORMANT deployment-coverage decision before first live day.

### Open carry-overs (not actioned today)
- **Open IMPLEMENTATION stage — `docs/runtime/stages/2026-05-18-dashboard-start-signal-preflight-mode.md`.** Commit `bd229c67` landed criteria 1-4 (mode-aware `_run_preflight_subprocess`, threaded `_prepare_profile_for_start`, `action_start` propagation, `action_preflight` live-mode default preserved) + 3 covering tests. **Criterion 7 outstanding:** restart dashboard, `POST /api/action/start?mode=signal&profile=topstep_50k_mnq_auto`, confirm non-blocked status, verify `logs/live/live_signals_2026-05-18.jsonl` appears within 30s. Delete stage file after operator verification.
- **Amendment 3.0 loader collision blocking NYSE_CLOSE prereg authoring** — `docs/audit/hypotheses/2026-05-13-mnq-nyse-close-mode-a-k1-revalidation.yaml` fails to load at `trading_app/hypothesis_loader.py:291` (theory_citation × Amendment 3.0). All 10 NYSE_CLOSE lanes remain PARKED. Decision-ledger: `mnq-nyse-close-k1-prereg-blocked-by-loader-2026-05-17`.
- **Dashboard `/api/bars-recent?instrument=MNQ` returns `"bars":[]`** — uncharacterized since 2026-05-16 debut. Needs ≥1 live run evidence (tick log + 3-min-later curl + last 5 aggregator log lines) before fix stage justified.
- **HWM tracker `hwm_dollars=0.0` on account 21944866** — shell exists, never populated. Defer until ≥1 real Monday session; revisit only if still 0.0 after broker activity.
- **`logs/live/live_<ts>.log` not written under `--live`** — output stdout-only on 2026-05-16 debut. Needs ≥1 more run to characterize.

### Hygiene
- Untracked draft `docs/audit/hypotheses/drafts/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.draft.yaml` is from a parallel session — leave alone unless owner identifies.
- `.coverage` shows `M` every session — test-runtime artifact, contains absolute paths, do NOT stage.

## Next Session — Preflight Gate Outcome (2026-05-17 ~21:19 BNE)
- **Step A result:** PASS 7/7. `python scripts/run_live_session.py --instrument MNQ --profile topstep_50k_mnq_auto --preflight` clean. Token acquired, 4 lanes loaded, contract `CON.F.US.MNQ.M26`, bracket+fill+notifications PASS, journal opens. **Step 7 executed (not SKIPPED) — exercised `_select_primary_and_shadow_accounts` against real broker without RuntimeError. a0b3c24b + bb0619d2 regression surface CLEAN.**
- **Step A anomaly (capital-class):** Step 7 returned `OK (copies=2, 1 accounts discovered)`. Profile `topstep_50k_mnq_auto.copies = 2` (`prop_profiles.py:481`); broker discovered 1 active account (id=21944866, `EXPRESS-V2-451890-53179846`). bb0619d2 fix degrades gracefully — no error — but `_select_primary_and_shadow_accounts(n_copies=2, accounts=[1 acct])` returns `(primary, shadows=[])`. At runtime `session_orchestrator.py:593` `if shadow_account_ids:` is FALSY, so single-account router is wired (line 607-608) and the `Copy trading: primary=..., shadows=[...]` log line at line 601-606 is NEVER emitted.
- **Step B result:** NOT EXECUTED — evidence rule satisfied from Step A alone. Operator-supplied rule: `if copies=2 AND discovered_accounts<2 → BLOCK_LAUNCH_COPY_SET_UNVERIFIED`. Running dashboard cannot produce required `shadows=[...]` evidence on current broker state.
- **Capital decision for Monday Strategy-A: RED (BLOCK_LAUNCH_COPY_SET_UNVERIFIED)** until ONE of the three options below is actioned. Code state itself is healthy (Step A proves it); the block is on the broker/profile mismatch, not on the fix regression.

### Monday-morning decision (pick ONE before 23:25 BNE Strategy-A launch)

**Option A — Provision the second TopstepX account (cleanest):**
  - Log into TopstepX → activate a second funded account under the same API credentials.
  - Re-run Step A. Expect `OK (copies=2, 2 accounts discovered)`.
  - Then run Step B (dashboard Start Live on `topstep_50k_mnq_auto`); confirm log line `Copy trading: primary=<id>, shadows=[<id>]` emitted from `session_orchestrator.py:601-606`.
  - Result: GREEN.

**Option B — Drop profile to `copies=1` (make profile match reality):**
  - Edit `trading_app/prop_profiles.py:481` → `copies=1`. Stage file required (production code edit).
  - Re-run Step A. Step 7 will SKIP per `run_live_session.py:348` `prof.copies <= 1` gate.
  - Loses copy-trading regression coverage but unblocks Monday on truthful single-account spec.
  - Result: YELLOW (single-account launched, copy-trading path uncovered until Option A is provisioned later).

**Option C — Explicit override, launch as-is (highest risk):**
  - Accept that `copies=2` in profile + `copies=1` at runtime is a silent degrade.
  - Write override justification here under "decision-ledger.md" and proceed.
  - Reviewer/auditor reading `prop_profiles.py:481` will see `copies=2` and form a wrong mental model.
  - **Not recommended** — institutional-rigor § 4 (canonical sources must be truthful) + § 6 (no silent failures).

### Carry-over (unchanged from line 22)
- `/api/bars-recent?instrument=MNQ` returning `"bars":[]` still uncharacterized; capture evidence during first live session per Phase 0 D3.

## Prior Session (2026-05-17 evening — Check 107 SHA cleanup)
- **Commit:** feat(check107) SHA migration manifest + sibling integrity check; Check 107 orphan-SHA 11 → 0 with zero DB mutation.
- **Detail (compressed):** Git-archaeology audit of 11 orphans → all mapped to Amendment 3.3 (`8ab4fe13`) `theory_grant: false` stamp; manifest at `docs/audit/check_107_sha_migrations.yaml` with introducing_commit / migration_commit / current_sha per entry; sibling check `check_phase_4_sha_migration_manifest_integrity` guards against fabricated entries. 207/207 tests pass. Full detail in `docs/audit/results/2026-05-17-check-107-orphan-sha-audit.md`.

## Older Session (2026-05-17 PM — ATR_P70 chordia unlock)
- **Tool:** Claude Code (Opus 4.7)
- **Commit:** a080967b — research(chordia): ATR_P70 unlock UNVERIFIED_INSUFFICIENT_POWER (PASS_CHORDIA / OOS power tier STATISTICALLY_USELESS)
- **Prior:** c6c190a3 (chore(gitignore): ignore tmp/ scratch directory)
- **Files changed:** 4 files in a080967b — 1 draft yaml (Amendment 3.3 `theory_grant: false` stamp) + 1 result MD + 1 result CSV + 1 stage closeout (`docs/runtime/stages/atr-p70-chordia-unlock-prereg-loop.md`).
- **Session summary:** Resumed pre-existing TRIVIAL stage `atr-p70-chordia-unlock-prereg-loop`. Prereg-loop had already executed prior to session start — verified acceptance via output inspection (result MD with `PASS_CHORDIA`, IS N=578 / ExpR=0.1731 / t=4.62 vs strict 3.79; OOS pooled sign-match N=47). Role-decision overridden to `UNVERIFIED_INSUFFICIENT_POWER` per backtesting-methodology RULE 3.3 — OOS power 15.2% pooled / 10.6% long / 9.2% short (all STATISTICALLY_USELESS tier; numbers match result MD body verbatim — earlier 12.8/11.1/9.9 draft was a transposition, corrected on amend). Long-side OOS sign flip (-0.041 vs IS +0.178) noise-consistent at this power, NOT refutational. Same discipline as P50 sibling (091a03e9) and 88a03d19 VWAP_MID_ALIGNED O30. No allocator, experimental_strategies, validated_setups, or chordia_audit_log.yaml mutation.
- **Drift noise:** 11 pre-existing Check 107 (Phase 4 SHA integrity) orphan-SHA violations in `experimental_strategies` — confirmed via `git stash` baseline that count is unchanged with/without P70 stage diff. Zero new violations introduced by this commit.
- **Hygiene:** `.coverage` shows as `M` in working tree on every session — it must NOT be staged (test-runtime artifact, contains absolute paths). Verify `git status` before `git add -A` style commits.

## Next Session — Active
- **Check 107 orphan-SHA cleanup CLOSED (2026-05-17 evening).** Audit MD `docs/audit/results/2026-05-17-check-107-orphan-sha-audit.md` + migration manifest `docs/audit/check_107_sha_migrations.yaml` shipped; Check 107 now reports 0 orphans with no DB mutation. Sibling check `check_phase_4_sha_migration_manifest_integrity` guards against fabricated manifest entries (cited commits must exist, migration_commit must touch the file, introducing_commit blob SHA-256 must equal orphan_sha). Future hypothesis-YAML migrations (any subsequent in-place edit) generate new orphans — append manifest entries with git evidence rather than re-running this audit.
- **VWAP_MID_ALIGNED_O30 status:** pre-reg authored + audited (PR #291); final verdict UNVERIFIED_INSUFFICIENT_POWER (OOS N=46, power 7.8%); bullpen-only, no capital deployment. See decision-ledger entry `o30-pass-chordia-audit-not-deployed-2026-05-14` (`docs/runtime/decision-ledger.md:66`).

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
2. **Highest-EV next is MNQ.** Live: **3 deployed MNQ lanes** per `docs/runtime/lane_allocation.json` (rebalance_date 2026-05-18, refreshed end of 2026-05-18 session: OVNRNG_100, VWAP_MID_RR1.5_O15, COST_LT12). Previously 4 lanes (verified 2026-05-16); C8 OOS gate caught silent failure on OVNRNG_25 during fresh rebalance. ~~Concrete candidate: rank-3 AUDIT_GAP_ONLY VWAP_MID_ALIGNED_O30 pre-reg authoring per Chordia v2 readouts.~~ **STALE 2026-05-18:** VWAP_MID_ALIGNED_O30 already authored + audited 2026-05-13 → final verdict UNVERIFIED_INSUFFICIENT_POWER (PR #291, decision-ledger `o30-pass-chordia-audit-not-deployed-2026-05-14`, result MD `2026-05-13-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.md`). Next concrete candidate: TBD on next session start — check Chordia v2 readout for rank-4+ AUDIT_GAP_ONLY survivor OR queue fast-lane batch (see item 6). (Prior "MEMORY 3 vs canonical 2 — reconcile" sub-item RESOLVED 2026-05-16: both surfaces now agree at 4 post-Chordia-K=20 rebalance per `memory/live_lanes_2026_05_14_four_deployed_post_chordia_k20.md`.) **2026-05-17 NYSE_CLOSE branch BLOCKED:** locked `docs/audit/hypotheses/2026-05-13-mnq-nyse-close-mode-a-k1-revalidation.yaml` fails to load at `trading_app/hypothesis_loader.py:291` (theory_citation × Amendment 3.0 collision); cohort-park rule keeps all 10 NYSE_CLOSE lanes PARKED until doctrine amendment lands. Decision-ledger entry: `mnq-nyse-close-k1-prereg-blocked-by-loader-2026-05-17`.
3. **Pre-existing carry-over (still open):** Track D MNQ COMEX_SETTLE Gate 0 runner design (Databento top-of-book table + bounded runner for DESIGN_ONLY prereg); **deployment-coverage decision on 78 ROUTABLE_DORMANT strategies — REFRESHED 2026-05-17**, fresh snapshot at `docs/audit/results/2026-05-17-deployment-coverage-orphans.md` (counts unchanged: 78 DORMANT / 0 ORPHAN / 809.4 R blocked-capital; ROUTABLE_ACTIVE annual_r −103.3 R after refresh of 296 stale `last_trade_day` rows). **Activation-vs-PARK decision DEFERRED to next session** per user stance "refresh first, decide after fresh numbers". No `prop_profiles.py` or `lane_allocation.json` mutation this session. Prior 2026-05-12 snapshot retained for evidence trail.
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

## This Session (2026-05-17 Codex)
- User request: "get it sorted for Claude to audit" on the NYSE_CLOSE K=1 quality blocker.
- Aligned tests with Amendment 3.3 semantics: `testing_mode: individual` is now valid without per-hypothesis `theory_citation` when `metadata.theory_grant: false` is explicit. Updated stale legacy test in `tests/test_trading_app/test_hypothesis_loader.py` that still enforced the pre-Amendment-3.3 rule.
- Verification: `pytest -q tests/test_trading_app/test_hypothesis_loader.py` -> 69 passed.
- No trading logic mutation, no DB mutation, no lane/profile mutation.

- Follow-up hardening: added regression test `test_real_k1_nyse_close_prereg_loads_no_theory_pathway_b` to pin the real locked prereg (`docs/audit/hypotheses/2026-05-13-mnq-nyse-close-mode-a-k1-revalidation.yaml`) to Amendment 3.3 semantics (`testing_mode=individual`, `has_theory=False`) so audits cannot regress to stale pre-3.3 assumptions.
- Extended verification: `pytest -q tests/test_trading_app/test_hypothesis_loader.py tests/test_llm_hypothesis_proposer.py` -> 111 passed.
