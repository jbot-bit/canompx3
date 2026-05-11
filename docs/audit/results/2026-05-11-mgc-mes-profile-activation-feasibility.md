---
date: 2026-05-11
mode: read-only feasibility orientation
status: COMPLETE
verdict: (b) MEDIUM — TopStep MGC/MES path is small; cross-broker path is medium-to-large
deliverable: decision document only — zero code edits
---

# MGC/MES Profile Activation — Feasibility Orientation

## Scope

Assess whether MGC/MES profile activation is a small code PR, a broader broker
integration project, or blocked by stale profile/account state. This document
is read-only feasibility orientation only; it does not authorize activation,
live sizing, profile flips, or broker changes.

## Bottom line

Two distinct paths exist, with very different effort:

1. **TopStep multi-instrument expansion (`topstep_50k_type_a`)** — broker-side is solved (projectx contract_resolver already handles MGC/MNQ/MES). The blocker is **account-side, not code-side**: a TopStep Type-A account class must exist and be live; AND the TopStep 5-XFA aggregate cap (`pre_session_check.check_topstep_xfa_aggregate_cap`, `trading_app/pre_session_check.py:430-460`) requires sum(copies)≤5, and `topstep_50k_mnq_auto` already burns 2 copies — leaving only 3 of 5 XFAs available for any new active TopStep profile. Activation work itself = small. **Sub-verdict (a) Small PR — pending user-side account decision.**

2. **Cross-broker MGC/MES (`bulenox_50k`, `tradeify_50k_type_b`, `self_funded_tradovate`)** — blocked by broker-auth state, not by code:
   - Tradeify profiles: "Tradovate auth broken" (notes in `prop_profiles.py:695,738`, archived plan `docs/plans/archive/2026-03/2026-03-31-max-extraction-plan.md:152,205`).
   - Bulenox: "Activate after Rithmic API conformance + paper trading validation" (`prop_profiles.py:780`).
   - Self-funded Tradovate: "Activate after opening Tradovate personal account + API test" (`prop_profiles.py:822`).
   **Sub-verdict (b) Medium — broker integration debt blocks at least 3 of the 4 profiles.**

The third option — `topstep_50k` (MGC TOKYO_OPEN-only) — is a special case: it is the SHADOW lane and is explicitly conditional pending P99 null clearance + N=250 temporal diversity. That is a research-gate problem, not an activation problem.

---

## Section 1 — `AccountProfile` schema (`trading_app/prop_profiles.py:80-160`)

| Field | Type | Required | Default | Consumer |
|---|---|---|---|---|
| `profile_id` | str | Y | — | dispatch key in `ACCOUNT_PROFILES` |
| `firm` | str | Y | — | `get_account_tier((firm, account_size))`, F-6 cap check, payout policy router |
| `account_size` | int | Y | — | `ACCOUNT_TIERS` lookup → `max_dd`, contracts limits |
| `copies` | int | N | 1 | XFA cap (F-6); `_select_primary_and_shadow_accounts` in `run_live_session.py:305` |
| `stop_multiplier` | float | N | 0.75 | `validate_dd_budget` worst-case, execution engine stop calc |
| `max_slots` | int | N | 6 | cognitive cap — selectors / book builder |
| **`active`** | bool | N | True | **Gates listed in §2 below.** |
| `allowed_sessions` | frozenset[str]\|None | N | None | `rebalance_lanes.py:113` filter, dispatch routing |
| `allowed_instruments` | frozenset[str]\|None | N | None | `rebalance_lanes.py:111` filter, dispatch routing |
| `daily_lanes` | tuple[DailyLaneSpec,…] | N | () | `effective_daily_lanes()` first, then `load_allocation_lanes()` fallback |
| `payout_policy_id` | str\|None | N | None | payout mechanics modeling; None = unmodeled |
| `max_risk_per_trade` | float\|None | N | None | dollar cap per trade; None = no limit |
| `is_express_funded` | bool | N | True | F-5 HWM tracker — XFA starts $0, TC starts at account_size |
| `is_live_funded` | bool | N | False | F-3 LFA DLL semantics — RESERVED, NOT WIRED |
| `notes` | str | N | "" | docs |
| `execution_symbol_map` | Mapping[str,str]\|None | N | None | Stage 1 NQ-mini symbol translation; identity if None |
| `execution_qty_divisor` | Mapping[str,int]\|None | N | None | paired with above; `__post_init__` validates both-or-neither |

The schema validator (`__post_init__`, lines 123-159) enforces only the symbol-map pairing invariant. No field controls broker selection — that comes from the `BROKER` env var (`broker_factory.get_broker_name`, default `projectx`) or `--broker` CLI flag, not from `profile.firm`.

**Implication:** `firm` is informational/payout-routing only. Broker dispatch and `firm` are orthogonal. A user could in principle run `--profile bulenox_50k --broker projectx` — but the broker would not have a Bulenox account underneath; auth would either succeed against TopStep or fail entirely. This is a defense-in-depth gap, not an activation requirement.

---

## Section 2 — Every consumer of `profile.active`

Grep across `trading_app/` and `scripts/`. The table below covers all runtime-path consumers; an additional cluster of scripts-tier reporting tools (`scripts/tools/backtest_allocator.py:495`, `build_optimal_profiles.py:446`, `generate_trade_sheet.py:504,1171`, `score_lanes.py:224`, `generate_profile_lanes.py:59`) also branch on `.active` but are LOW-severity reporting filters — flipping the flag silently includes the profile in their output, never blocks activation. **The list below is not exhaustive for the reporting tier; it is exhaustive for runtime gates.**

| File:line | What flipping `.active=True` enables | Severity |
|---|---|---|
| `trading_app/prop_profiles.py:891` (`get_active_profile_ids`) | Includes profile in default enumeration | LOW |
| `trading_app/prop_profiles.py:911` (`resolve_profile_id` when `active_only=True`) | Raises `ValueError` if False, returns id if True | MED |
| `trading_app/pre_session_check.py:443` (`check_topstep_xfa_aggregate_cap`) | Adds `copies` to TopStep 5-XFA total | **BLOCKER** — must keep ≤5 |
| `trading_app/pre_session_check.py:66` and `:219` (default profile-resolution) | First-True-wins primary profile picker | MED |
| `trading_app/paper_trade_logger.py` (lines 64,66,217,219 per grep) | Routes paper-trade rows to a profile | LOW |
| `trading_app/prop_portfolio.py:671` (`build_book` enumerator) | Includes in cross-profile book builder | LOW |
| `trading_app/lane_allocator.py` | NO direct `.active` consumption — gating happens at callers | — |
| `scripts/tools/rebalance_lanes.py:83,92` | `--all-profiles` filter; first-active default | MED |
| `scripts/run_live_session.py` | **NO `.active` check on `--profile X` path** — reads `ACCOUNT_PROFILES[args.profile]` directly | (see Stage 3-B below) |
| `pipeline/check_drift.py:5611-5648` | Validates that every `daily_lane` in every ACTIVE profile maps to a `validated_setups` row with `status='active'` and `deployment_scope='deployable'` | **BLOCKER** — must pass before commit |
| `trading_app/live/bot_dashboard.py`, `bot_dashboard.html` | Surface display | LOW |
| `trading_app/live/multi_runner.py`, `live/session_orchestrator.py`, `live/performance_monitor.py` | runtime — must verify per-profile (not enumerated here; out of scope for orientation; flagged in §4 follow-up) | MED |
| `trading_app/live_config.py`, `lane_ctl.py`, `portfolio.py`, `paper_trader.py` | also reference `ACCOUNT_PROFILES`; `.active` semantics not exhaustively traced | LOW-MED |

**Two hard gates fire on `active=True`:**
1. `check_topstep_xfa_aggregate_cap`: sum of TopStep `copies` must remain ≤5. Today: `topstep_50k_mnq_auto.copies=2`, so 3 slots left.
2. Drift check at `check_drift.py:5611`: every lane in `daily_lanes` (or `lane_allocation.json` fallback) must exist in `validated_setups` as `status='active' AND deployment_scope='deployable'`. If a stale ghost lane is in the profile, drift fails and the commit cannot land.

---

## Section 3 — 8-profile inventory

Reading `ACCOUNT_PROFILES` (`trading_app/prop_profiles.py:407-860`):

| profile_id | firm | size | active | instruments | sessions | daily_lanes | payout_policy_id | copies | Notes blocker |
|---|---|---|---|---|---|---|---|---|---|
| `tradeify_50k` | tradeify | 50K | False | {MNQ} | 5 sessions | 3 MNQ lanes (allocator-rebuilt 2026-04-19) | tradeify_select_funded | 5 | "Tradovate API bot ready" (no MGC/MES) |
| `topstep_50k` | topstep | 50K | False | {MGC} | {TOKYO_OPEN} | 1 lane (conditional shadow) | topstep_express_standard | 5 | Per-session P99 not cleared; N=125; shadow only |
| `topstep_50k_mnq_auto` | topstep | 50K | **True** | {MNQ} | 7 sessions | () dynamic via `lane_allocation.json` | topstep_express_standard | 2 | **LIVE** |
| `topstep_50k_mes_auto` | topstep | 50K | False | {MES} | {CME_PRECLOSE} | 1 MES lane (ORB_G8) | topstep_express_standard | 1 | "User activates when TopStep Express account is ready" |
| `topstep_50k_type_a` | topstep | 50K | False | **{MNQ,MGC,MES}** | 8 sessions inc LONDON_METALS | 4 MNQ lanes (no MGC/MES populated) | topstep_express_standard | 5 | "Activate after proving loop on topstep_50k_mnq_auto" |
| `topstep_100k_type_a` | topstep | 100K | False | {MNQ,MGC,MES} | 8 sessions | 4 MNQ lanes | topstep_express_standard | 5 | "Activate when upgrading from 50K tier" |
| `tradeify_50k_type_b` | tradeify | 50K | False | {MNQ,MGC,MES} | 8 sessions | 5 MNQ lanes | tradeify_select_funded | 5 | "Blocked: Tradovate auth broken" |
| `tradeify_100k_type_b` | tradeify | 100K | False | {MNQ,MGC,MES} | 8 sessions | 5 MNQ lanes | tradeify_select_funded | 5 | "Blocked: Tradovate auth broken" |
| `bulenox_50k` | bulenox | 50K | False | {MNQ,MGC} | 5 sessions | 4 MNQ lanes | None | 3 | "Activate after Rithmic API conformance + paper trading validation" |
| `self_funded_tradovate` | self_funded | 30K | False | {MNQ,MGC,MES} | 8 sessions | 6 MNQ lanes | self_funded | 1 | "Activate after opening Tradovate personal account + API test" |

**Class observation:** every multi-instrument profile is **already a fully-populated scaffold**, NOT a placeholder. `daily_lanes`, `allowed_*`, `payout_policy_id`, `copies`, `max_slots` are all set with real values (rebuilt 2026-04-19 from the allocator-backed shelf). The catch: every populated `daily_lanes` is **MNQ-only** even on `{MNQ,MGC,MES}` profiles. The MGC/MES surface widening would require lane authoring (validated MGC LONDON_METALS rows and MES CME_PRECLOSE rows exist in `validated_setups` per memory `recent_findings`; they would need to be added to the profile's `daily_lanes` tuple, then pass drift check #5611).

---

## Section 4 — Diff: `topstep_50k_mnq_auto` (active) vs `topstep_50k_type_a` (nearest MGC/MES candidate)

Comparison target chosen: `topstep_50k_type_a`. Justification: same firm (`topstep`), same account_size (50K), same payout_policy_id, same broker (projectx via TopStep), instruments superset = {MNQ,MGC,MES}, sessions superset includes LONDON_METALS where MGC lives. `bulenox_50k` rejected — different firm and broker (rithmic) introduces broker-auth confound.

| Field | `topstep_50k_mnq_auto` | `topstep_50k_type_a` | Delta | Risk to flip |
|---|---|---|---|---|
| `firm` | topstep | topstep | same | none |
| `account_size` | 50_000 | 50_000 | same | none |
| `copies` | 2 | **5** | +3 | **CAP** — would push TopStep total to 7 > 5; check_topstep_xfa_aggregate_cap BLOCKS |
| `stop_multiplier` | 0.75 | 0.75 | same | none |
| `max_slots` | 7 | 16 | +9 | benign |
| `active` | True | False | flip | gate flip |
| `allowed_sessions` | 7 sessions (no LONDON_METALS) | 8 sessions (+LONDON_METALS) | +LONDON_METALS, -SINGAPORE_OPEN, -EUROPE_FLOW | session catalog change |
| `allowed_instruments` | {MNQ} | {MNQ,MGC,MES} | +MGC, +MES | **the entire point of activation** |
| `daily_lanes` | `()` dynamic via JSON | 4 hardcoded MNQ lanes | mode change | JSON profile_id mismatch (see below) |
| `payout_policy_id` | topstep_express_standard | topstep_express_standard | same | none |

**Critical issues uncovered by the diff:**

A. **5-XFA cap.** Both profiles set `copies` and BOTH fire the TopStep cap check. With mnq_auto at 2 and type_a at 5, simultaneous activation = 7 > 5 → drift BLOCK. Resolution requires lowering `copies` on one or both to keep sum ≤5. This is a **user-decision-required** parameter, not pure code work.

B. **Lane allocation JSON is single-profile-scoped.** `docs/runtime/lane_allocation.json` carries `"profile_id": "topstep_50k_mnq_auto"`. `load_allocation_lanes()` (lines 1068-1136) fails-closed on profile mismatch (line 1102-1103). `topstep_50k_type_a.daily_lanes` IS populated (4 MNQ lanes from 2026-04-19 rebuild) so it would not need the JSON — BUT if user wanted dynamic allocator-managed lanes for type_a, they'd need either (i) a second JSON file at a custom path, OR (ii) refactor `load_allocation_lanes` to support a profile→path map, OR (iii) refactor the allocator to write per-profile files. Today: only path (i) works without code change, via `allocation_path` arg — but no caller passes that arg, so per-profile JSON is **not wired**.

C. **`daily_lanes` populated but MNQ-only.** Even though `allowed_instruments={MNQ,MGC,MES}`, the 4 hardcoded lanes are all MNQ. Activation would still produce 0 MGC/MES exposure unless MGC/MES lanes are authored into the tuple. The MGC LONDON_METALS rows (13 validated per `recent_findings`) and MES CME_PRECLOSE rows (48 validated) are NOT in the profile.

D. **Sessions delta.** type_a adds LONDON_METALS (where MGC lives) and drops SINGAPORE_OPEN/EUROPE_FLOW. This is consistent with the MGC widening goal but means if the user is currently running mnq_auto's SINGAPORE_OPEN/EUROPE_FLOW lanes, switching to type_a alone would lose them. Concurrent operation requires the 5-XFA cap fix above.

E. **`is_express_funded` default.** Both profiles inherit True (XFA). F-5 HWM tracker semantics carry — no change.

F. **`is_live_funded` not wired.** Both default False. F-3 LFA DLL is reserved/unwired. Not relevant to activation.

---

## Section 5 — Broker dependency

(Stage 3-A.)

| Profile firm | Broker | Integration state | MGC/MES support |
|---|---|---|---|
| topstep_* | projectx (TopStepX/ProjectX API) | LIVE (currently executing `topstep_50k_mnq_auto`) | **YES** — `trading_app/live/projectx/contract_resolver.py:14-17` includes MGC, MNQ, MES |
| tradeify_* | tradovate | **BROKEN** — "Tradovate auth broken (password rejected)" per `docs/plans/archive/2026-03/2026-03-31-max-extraction-plan.md:152,205`; profiles annotated `# Blocked: Tradovate auth broken` at `prop_profiles.py:695,738` | Integration exists (`live/tradovate/`) but blocked at auth |
| bulenox_* | rithmic | "Activate after Rithmic API conformance + paper trading validation" per `prop_profiles.py:780`; integration exists (`live/rithmic/`) | **YES** — `trading_app/live/rithmic/contracts.py:26-30` `INSTRUMENT_ROOTS` explicitly maps MES, MNQ, MGC (docstring line 25: "No translation needed") |
| self_funded_tradovate | tradovate (per profile firm) | same as Tradeify — broken auth | Same as Tradeify |

**Broker selection is decoupled from `profile.firm`.** `broker_factory.get_broker_name()` reads `BROKER` env var or `--broker` flag (`run_live_session.py:431`). The factory dispatches to one of `projectx`/`tradovate`/`rithmic`. There is NO assertion that the chosen broker matches `profile.firm`. This is a defense-in-depth gap: operator could in principle mismatch broker and profile.

**Implication for activation:**
- `topstep_50k_type_a` activation has **zero broker-side blocker** — projectx already handles MGC, MNQ, MES.
- `tradeify_*` activation requires fixing Tradovate auth first.
- `bulenox_50k` activation requires Rithmic conformance work + paper-trade validation cycle.
- `self_funded_tradovate` requires both (a) opening a real Tradovate personal account AND (b) the Tradovate auth fix.

---

## Section 6 — Paper-trade staging

(Stage 3-B.)

Two distinct paper-trade surfaces exist:

A. **Historical replay** (`trading_app/paper_trader.py`) — takes `--instrument` only, no `--profile`. Feeds historical bars through ExecutionEngine + RiskManager. Used for IS/OOS replay. **Cannot stage a profile.**

B. **Live signal-only / demo mode** (`scripts/run_live_session.py`):
   - `--signal-only`: connects to broker for bar feed only, places NO orders. Safe.
   - `--demo`: places orders on broker's DEMO/paper account, no real money.
   - `--profile X`: reads `ACCOUNT_PROFILES[args.profile]` **directly with no `.active` check** (line 498). User can pass `--profile topstep_50k_type_a --demo` today without flipping `active=True`.

**This means paper-trade staging WITHOUT activation is possible.** A multi-instrument profile can be exercised end-to-end (auth, contract resolution, signal generation, order placement to broker demo) without tripping the `pre_session_check.check_topstep_xfa_aggregate_cap` gate or the drift check at line 5611 (both gate on `profile.active`).

**Caveats:**
- The drift check at `check_drift.py:5611` only enumerates active profiles, so a stale-lane bug in `topstep_50k_type_a.daily_lanes` would NOT be caught at commit time. Operator must validate lanes manually before paper-trade.
- The `paper_trade_logger.py` lane mapping is profile-scoped (lines 64-66, 217-219 enumerate active profiles); paper trades from an inactive profile in `--demo` mode may not route to the expected profile bucket. Not exhaustively traced — flagged for follow-up if path (a) is chosen.

---

## Section 7 — Prior history

(Stage 3-C.)

Commits (`git log --all --oneline | grep -iE "mgc.*profile|mes.*profile|activate|bulenox|tradeify|type_a|type_b"`):

| Commit | Summary | Relevance |
|---|---|---|
| `f8b583d4` | fix(prop_profiles): correct MES profile metadata per Bloomey review | MES profile post-creation cleanup |
| `dad2894c` | feat(prop_profiles): MES CME_PRECLOSE deployment profile (`topstep_50k_mes_auto`) | **prior MES profile creation** — never flipped to active |
| `c0be4c50` | feat: full auto-scaling profiles — TYPE-A/TYPE-B for TopStep + Tradeify at 50K/100K | **the inactive scaffolds were authored together in one commit** |
| `1376764b` | plan: multi-broker deployment — TopStep+Tradovate+Bulenox+IBKR | the original multi-broker plan that produced these scaffolds |
| `2a4fd850` | feat: Rithmic broker adapter for Bulenox/Elite | broker integration for `bulenox_50k` exists |
| `915af820` | feat(prop_profiles): rewrite 7 stale inactive profile lane sets | 2026-04-19 lane rebuild — sets are NOT stale |
| `fbb6890e` | feat: uncomment $100K Apex upgrade profile (active=False, ready to activate) | "ready to activate" pattern is precedent for this work |

**No prior session has executed activation.** Closest precedent: MES profile was created (dad2894c) and then refined (f8b583d4) but never flipped. TYPE-A/TYPE-B were scaffolded (c0be4c50) and rebuilt 2026-04-19 but never activated.

**No `HANDOFF.md` mention** of multi-profile activation as next-session work (grep returned nothing).

**`docs/runtime/debt-ledger.md`** mentions paper-trade canonical-source + MES cost-realism debt; no profile-activation debt entry. `cost-realism-slippage-pilot` Trigger A (auditor's `2026-04-25-mes-event-tail-slippage-debt-scope.md`) fires on "any MES lane enters the live allocator" — this is a **research/cost gate**, not an activation block, but it would attach to MES activation work.

---

## Section 8 — Verdict

### Per-path verdict

**(a) Small PR — `topstep_50k_type_a` MGC+MES expansion (subject to user-side decision):**

If user has (or will have) a TopStep Type-A-equivalent set of XFA accounts AND is willing to reduce `topstep_50k_mnq_auto.copies` to keep TopStep total ≤5, then activation is roughly:
1. Author MGC LONDON_METALS lane(s) and MES CME_PRECLOSE lane(s) into `topstep_50k_type_a.daily_lanes` (sourced from validated_setups; MGC=13 candidate rows, MES=48 candidate rows per `recent_findings`).
2. Lower `copies` on one or both TopStep profiles to keep XFA total ≤5.
3. Flip `active=True`.
4. Pass drift check #5611 (lanes-validated alignment).
5. Paper-trade via `--profile topstep_50k_type_a --demo` for a confirmation cycle.
6. Drift check + tests + commit.

Estimated effort: a few hours to one day. **Path is unblocked by code; user decision on accounts and copies-split is the gating input.**

> **Single-blocking user decision for the Small PR path:** `topstep_50k_type_a.copies=5` cannot coexist with `topstep_50k_mnq_auto.copies=2` without breaching the TopStep 5-XFA cap (sum=7>5). The PR cannot land without lowering one or both. Common splits: (i) `mnq_auto.copies=2, type_a.copies=3`; (ii) `mnq_auto.copies=1, type_a.copies=4`; (iii) deactivate `mnq_auto` and run `type_a.copies=5` (requires migrating mnq_auto's deployed lanes into type_a or accepting lane loss). This is a hard arithmetic gate enforced by `pre_session_check.check_topstep_xfa_aggregate_cap` on every session start.

**(b) Medium project — cross-broker activation (`tradeify_*`, `bulenox_50k`, `self_funded_tradovate`):**

Each is gated by broker-side work that is not just "flip the flag":
- Tradeify: fix Tradovate auth. Effort unknown but explicitly listed as a Phase-2 blocker in archive plan.
- Bulenox: complete Rithmic API conformance + paper-trade validation cycle.
- Self-funded: requires user opening a real Tradovate personal account + API test, in addition to the Tradovate auth fix.

Estimated effort: 2-4 days each, plus user-side account setup.

**(c) Large project — N/A.** No broker integration is missing entirely (all three have adapters). No risk-manager or infra rewrite is required by activation itself.

### Overall verdict: **(b) MEDIUM** — driven by the cross-broker paths. The TopStep Type-A path is (a) but it is conditional on user-side account state that this orientation cannot resolve.

### Audit correction (post-evidence-auditor pass)

Evidence-auditor returned CONDITIONAL with three findings, all addressed in-doc:

1. **Consumer list non-exhaustive for reporting tier.** Updated §2 prefix to acknowledge `scripts/tools/backtest_allocator.py:495`, `build_optimal_profiles.py:446`, `generate_trade_sheet.py:504,1171`, `score_lanes.py:224`, `generate_profile_lanes.py:59` as LOW-severity `.active` consumers that flip-includes the profile in reporting outputs (no activation impact).
2. **Rithmic MGC support unverified.** Now verified: `trading_app/live/rithmic/contracts.py:26-30` explicitly maps MGC, MNQ, MES. Bulenox path remains (b) Medium gated by API conformance, NOT by missing instrument support.
3. **Copies-split user decision buried.** Now surfaced as a callout box under the Small PR path verdict.

### What this orientation does NOT decide

- Which profile to activate (depends on user's TopStep XFA count + Type-A account status + risk appetite).
- Whether to lower `topstep_50k_mnq_auto.copies` to fit a Type-A activation.
- Whether MGC LONDON_METALS or MES CME_PRECLOSE validated rows are the right lanes to author into Type-A (research-grade question — needs current `validated_setups` query + chordia-audit status).
- Estimated dollar EV — explicitly out of scope.
- Whether Tradovate-auth and Rithmic-conformance work is worth pursuing now (depends on multi-firm scaling priorities — see `topstep_scaling_corrected_apr15.md` in memory).

## Outputs

This orientation produced one decision artifact:

- `docs/audit/results/2026-05-11-mgc-mes-profile-activation-feasibility.md`

No code, config, database, profile, broker, or runtime state was changed. The
evidence base is static source/doc inspection plus referenced prior findings,
not a fresh broker-auth or live-session run.

## Limitations

This is not a deployment-readiness review. It does not prove that any TopStep,
Tradeify, Bulenox, or self-funded account exists, has current credentials, or
can place orders today. It also does not re-run per-lane statistical validation
or author MGC/MES `daily_lanes`; those remain required before any activation
PR can be treated as deployable.

### Recommended next plan (depending on user input)

1. **If user has Type-A TopStep accounts ready** → small plan: cut a worktree, author MGC/MES lanes into `topstep_50k_type_a.daily_lanes`, adjust copies, flip active, paper-trade, drift, PR. Single-stage.
2. **If user wants to clear cross-broker paths first** → multi-stage plan: (i) Tradovate auth fix, (ii) Bulenox Rithmic conformance, (iii) paper-trade validation cycles, (iv) flip active flags. 2-4 stages.
3. **If neither is ready** → park; revisit when chordia-audit unlocks the MGC/MES validated rows for promotion AND broker prerequisites are met.

---

## Appendix — file references (for follow-up)

- Canonical profile model: `trading_app/prop_profiles.py:34-160` (dataclasses), `:243-400` (firm/tier specs), `:407-860` (profile inventory), `:883-927` (resolvers), `:1068-1136` (allocator JSON loader)
- TopStep 5-XFA cap: `trading_app/pre_session_check.py:430-460`
- Lanes-validated drift check: `pipeline/check_drift.py:5611-5654`
- Broker factory: `trading_app/live/broker_factory.py:11-108`
- Live entry point: `scripts/run_live_session.py:387-738`
- ProjectX contract resolution (proves MGC/MES support): `trading_app/live/projectx/contract_resolver.py:14-17`
- Tradovate-auth blocker source: `docs/plans/archive/2026-03/2026-03-31-max-extraction-plan.md:152,205`
- MES event-tail Trigger A doc: `docs/plans/2026-04-25-mes-event-tail-slippage-debt-scope.md` (attaches to MES activation)
