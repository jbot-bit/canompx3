# Capital-Risk Review Loop — Ledger

**Purpose:** Persistent memory for the recurring `/code-review` capital-risk loop (cron `4f71a27f`, every 20m).
Each iteration MUST read this file FIRST, skip anything already resolved/surfaced/not-a-bug below, and APPEND its
results. This prevents re-auditing the same surfaces and re-surfacing the same findings every 20 minutes.

## How each iteration uses this ledger
1. Read this whole file before auditing.
2. Skip files/findings marked `RESOLVED`, `SURFACED-OPEN` (already waiting on operator), or `NOT-A-BUG`.
3. Audit a NEW or CHANGED surface, or go deeper on a previously time-boxed one.
4. Append a new `## Iteration N` block. Never rewrite prior blocks (append-only, like other runtime ledgers).
5. If git HEAD moved since last iteration, note the new commits — they may resolve open findings.

## Known-landed capital fixes (do NOT re-litigate as bugs)
- account-routing fail-closed — `18fdfa17`
- naked-position kill+flatten — session_orchestrator.py ~2837-2854
- C11 6-file content fingerprint + D-3 sizing guard — `fab6292d`
- preflight-on-live-arm unconditional gate — `755cdd7e`
- live drift-gate exact-match ignore + rename parse — `99caaa4e`
- reaper dead-holder corpse-lock guard — `3e9aec96`
- bar-persister CRITICAL flush log — `43d58129`
- Finding-4 bars-present/features-absent drift check — `ed29aac7`

## Rotation queue (audit NEW ground each iteration — do NOT gate on git diff)
The loop is NOT diff-gated anymore (idle session ships no commits → diff-gating skips forever,
useless). Each iteration audits the next UNREVIEWED module(s) below for capital-loss risk, then
marks them [x]. When exhausted, restart or switch lens (perf, error-handling, test coverage).

REVIEWED clean (Iter 1-2): session_orchestrator, account_survival, execution_engine,
run_live_session, prop_profiles, cost_model, portfolio (sizing).

UNREVIEWED — capital-adjacent, priority order:
- [x] broker_connections.py + broker_base.py — real-broker order routing/auth (Iter 5)
- [x] copy_order_router.py — copy-trade fan-out (MULTIPLIES exposure ×N accounts) (Iter 5)
- [x] position_tracker.py — position truth vs broker (Iter 5)
- [x] webhook_server.py + http_client.py — external order-trigger surface (Iter 6)
- [ ] instance_lock.py + multi_runner.py — double-instance = double orders
- [ ] account_hwm_tracker.py + sr_monitor.py — HWM/MLL halt truth
- [ ] lane_allocator.py + prop_portfolio.py — allocation/cap enforcement
- [ ] pre_session_check.py + telemetry_maturity.py — preflight gates
- [ ] alert_engine.py + trade_journal.py — failure-alert + audit-trail integrity
- [ ] bar_persister.py + bar_aggregator.py + bar_ring.py — data-into-engine integrity

## Open findings registry (one line each; status: SURFACED-OPEN / RESOLVED / NOT-A-BUG)
<!-- iterations append here -->
- SURFACED-OPEN(MEDIUM) | webhook_server.py:403,408 vs :219-243,252-261 dedup/position-limit TOCTOU | dedup cache (line 408) AND `_OPEN_POSITIONS` increment (line 403) are written AFTER `await _place_order` (line 373); guard checks (`_check_dedup` :336, `_check_position_limit` :344) run synchronously BEFORE the await reading not-yet-updated state. Two near-simultaneous identical entry alerts each pass dedup (cache empty) + position cap (counter 0) → BOTH submit, 2 open positions despite MAX_OPEN_POSITIONS=1 and the dedup window. PROVEN by new char-test `test_concurrent_duplicate_entries_both_submit_toctou` (submit_calls==2, _OPEN_POSITIONS==2). Bounded by rate-limit ≤3/60s (rate-append IS pre-await, line 275); demo-default; NOT wired into session_orchestrator live path (webhook is a separate Tradovate/feedless entry path). NOT silently fixed: moving dedup-reserve/position-increment pre-await changes order-trigger timing (Tier B). OPERATOR DECISION: (a) accept as bounded (rate-limit backstop + TV serial-fire normal case), or (b) reserve dedup key + increment position optimistically before await, rolling back on submit failure. | Iter 6
- NOT-A-BUG | webhook_server.py:154-155,323,171 inbound auth | NO unauthenticated→order path. Lifespan RAISES RuntimeError if WEBHOOK_SECRET empty (server refuses to start, :154-155); /trade does `hmac.compare_digest(req.secret, WEBHOOK_SECRET)` (:323) → 403 on any mismatch; req.secret defaults "" (:171) so a no-secret caller gets 403. Bound to 127.0.0.1 only (run_webhook_server.py:89), external exposure via named cloudflared tunnel only; --live launcher requires secret ≥16 chars + typed CONFIRM (run_webhook_server.py:54-72). Pinned by test_webhook_rejects_wrong_secret + test_webhook_lifespan_blocks_empty_secret | Iter 6
- NOT-A-BUG | webhook_server.py:164-201 payload validation | Pydantic TradeRequest fail-closes on bad input: direction∉{long,short}→422, action∉{entry,exit}→422, entry_model∉{E1,E2}→422; instrument allowlist (:328 vs ACTIVE_ORB_INSTRUMENTS)→400; E2 missing entry_price→ValueError→400 (:285,375); qty cap MAX_ORDER_QTY checked on POST-division execution_qty (:352, the value that reaches the router)→400. No injection surface (typed JSON→dataclass spec, no SQL/shell). Malformed payload fails CLOSED (reject), never fails open | Iter 6
- NOT-A-BUG | http_client.py:280-294,409-414 status/protocol handling | non-2xx NOT treated as success: 4xx(≠401/429)→BrokerPermanentError raise (:283-289); 5xx→retry then BrokerTransientError (:333-340); 401→single refresh then BrokerAuthError (:296-316); 429→Retry-After then BrokerRateLimitExhausted (:318-326); non-JSON OR {"success":false} body on a 200→BrokerProtocolError, NO retry (:400-414). record_success() only fires AFTER parse-layer validation (:372,390), so a 200+success:false counts as FAILURE not success. record_failure on every failure class incl deadline-exceeded (:246). No swallowed exception masks a failed placement | Iter 6
- NOT-A-BUG | http_client.py:84-100,233-256 retry double-send risk | ORDER_POLICY retries (max_attempts=4) but order-placement idempotency is enforced at the CALL SITE (order_router idempotency key, documented :97-99) not the client; retries here are on connect/timeout/5xx where the broker did NOT ack — a transient with no response cannot have placed an order. Per-call deadline_s budget (:233,243-247) bounds total wall-clock so retries can't exceed caller SLA. TLS/cert: uses default requests.Session() (:205) — verify=True by default, NOT disabled anywhere in file | Iter 6
- NOT-A-BUG | copy_order_router.py:98-121 submit() fan-out | primary submits FIRST; ProjectX submit() returns status="submitted" on success or RAISES on failure (order_router.py:209-214,238) — never returns a non-skip "unknown" status. Primary failure raises → caught upstream (kill+flatten), shadows never reached. Primary success → "submitted" correctly copies to shadows. Inverted _SKIP_STATUSES is a defensive belt for failure-as-status brokers; unreachable mis-copy on live ProjectX path | Iter 5
- NOT-A-BUG | copy_order_router.py:107-121 shadow failure does NOT raise in submit() | by design: primary already filled, must return its result so orchestrator manages position. Marks router DEGRADED → orchestrator proactively kills+flattens every 10 bars (session_orchestrator.py:1943-1953) AND next submit() raises ShadowDivergenceError (line 80-85). Two independent backstops; divergence cannot produce a new asymmetric entry. Net of a failed shadow = shadow flat + primary flattened on halt = all flat (safe) | Iter 5
- NOT-A-BUG | copy_order_router.py:58-67,123-198 raw qty copied across accounts | single execution_qty (from primary event.contracts) built into ONE spec, copied verbatim to shadows. Intentional copy-trade semantics (same-tier Topstep accounts); live max_contracts clamped to 1 micro (portfolio.py:877) so per-account cap divergence is not reachable today. Design assumption, not a cap-violation bug | Iter 5
- NOT-A-BUG | broker_connections.py:236-247 connect_all_enabled swallows per-conn failure | logs warning + continues; NOT fail-open into trading — this is the connection-store UI/preflight path, not an order-arm gate. Per-connection connect() (line 217-234) DOES fail-closed (re-raises, sets status="error", clears auth). Live arming gates (preflight chain, 755cdd7e) enforce readiness separately | Iter 5
- NOT-A-BUG | position_tracker.py:141-148 / 131-138 / 176-181 / 64-71 | pure in-memory lifecycle FSM (NOT broker-truth source): late fill after exit cannot resurrect (R2-H3 CRITICAL reject), duplicate entry fill ignored, exit-sent rejected if entry unfilled (R2-H2), double-entry blocked. All transitions fail-closed; broker reconciliation lives in session_orchestrator (_reconcile_positions_on_reconnect / EOD), verified Iter 1 | Iter 5
- NOT-A-BUG | broker_base.py:178-238 bracket-leg verify contract | (None,None) default is the CORRECT answer for non-queryable-leg brokers; callers MUST gate on has_queryable_bracket_legs() (default True conservative) and supports_sequential_bracket_ids() (default False conservative). CopyOrderRouter delegates all three to primary (copy_order_router.py:204-226) so the active ProjectX path inherits real verification, not the no-op default | Iter 5
- NOT-A-BUG | session_orchestrator.py:2859 `_submit_bracket` separate-submit path | unreachable in live (all 3 brokers native-bracket → return early); live entries use merged-bracket path with full naked-position kill/flatten | Iter 1
- NOT-A-BUG | account_survival.py:424 hardcoded `contracts_per_trade=1` | intentional per self-funded-sizing-doctrine; D-3 gate fails closed if live sizes above it | Iter 1
- NOT-A-BUG | execution_engine.py:259-302 `_compute_contracts` clamp-to-max | vol-scaler can only reduce-then-reject (<1→0) or clamp to max_contracts; live profile path hardcodes max_contracts=1 (portfolio.py:877) → 1-micro fail-closed end-to-end | Iter 2
- NOT-A-BUG | execution_engine.py:982-983 / 1203 / 1378-1379 HALF_SIZE `max(1, int(...))` floor | size_multiplier=0.5 at 1 contract floors back to 1 → HALF_SIZE no-op; documented; errs conservative (never increases exposure) | Iter 2
- NOT-A-BUG(note) | session_orchestrator.py:2911-2924 EXIT divisor-raise `return` w/o flatten | asymmetric vs entry path (which emergency-flattens on divisor-raise), BUT dormant: divisor=1 for all active profiles, and any position that ENTERED had divisible qty so its same-qty EXIT is also divisible → unreachable for entered positions. Defensive-hardening candidate only, NOT a live bug | Iter 2
- NOT-A-BUG | run_live_session.py:1364-1403 mandatory pre-launch preflight gate | unconditional in-process _run_preflight on every --demo/--live (CLI+dashboard), sys.exit(1) on FAIL, no skip flag (755cdd7e); signal-only self-SKIPs | Iter 2
- NOT-A-BUG | run_live_session.py:599-693 telemetry-maturity Express-funded waiver | WAIVER IS CORRECT BY DOCTRINE (telemetry-maturity-waiver.md); verified fail-CLOSED for real-capital: None/unknown profile or is_express_funded=False → FAIL | Iter 2
- NOT-A-BUG | cost_model.py:97-251 COST_SPECS active instruments | MGC/MNQ/MES all carry nonzero commission_rt + spread_doubled + slippage; no zero-cost / missing-cost active instrument; get_cost_spec fail-closed ValueError on unknown | Iter 2

---

## Iteration 1 (2026-06-07)

**Surfaces audited:** (1) `trading_app/live/session_orchestrator.py` — kill switch,
emergency flatten, bracket-leg verification/reconciliation, reconnect resync, entry
guards. (2) `trading_app/account_survival.py` — DD-projection Monte Carlo, survival
sizing, D-3 sizing-parity gate, look-ahead check.

**Deferred to next iteration:** `trading_app/execution_engine.py` (order submission /
fill-vs-submit / sizing application — note: file is at `trading_app/execution_engine.py`,
NOT `trading_app/live/`), `scripts/run_live_session.py` (preflight chain / arming gates),
`trading_app/prop_profiles.py` (is_express_funded / lane caps), `pipeline/cost_model.py`.

**Findings table:**
| severity | file:line | one-liner | disposition |
|---|---|---|---|
| info | session_orchestrator.py:3042-3146 `_emergency_flatten` | cancels bracket legs before exit; marks FLAT only on confirmed fill (else PENDING_EXIT for EOD broker-truth reconcile); 3-retry then MANUAL CLOSE alert | NOT-A-BUG (hardened) |
| info | session_orchestrator.py:3183-3239 `_kill_switch_positions_confirmed_flat` | fail-closed: generic query_open Exception → NOT confirmed + alert; broker-open or local-active → NOT flat | NOT-A-BUG (correct) |
| info | session_orchestrator.py:3241-3334 `_reconcile_positions_on_reconnect` | untracked broker position on reconnect → kill+flatten (fail-closed); generic Exception → alert, no silent clean resume | NOT-A-BUG (correct, matches Row08 #2) |
| info | session_orchestrator.py:2192-2276 `_submit_bracket` | all 3 naked sub-paths (no risk_pts / spec None / submit raises) → kill+flatten; present-but-zero risk_pts no longer takes median fallback | NOT-A-BUG (F4 hardened) |
| info | session_orchestrator.py:2776-2856 entry bracket-leg verify | missing SL/TP or unavailable fallback → naked_position → kill+flatten; merged-bracket safety gate refuses entry if not merged (2692) | NOT-A-BUG (correct) |
| info | session_orchestrator.py:2438-2561 ENTRY guards | C1 kill-switch guard, orphan containment, ORB cap, max-risk-per-trade, RiskManager halt (L1), HWM/MLL halt (L2), allocator block — all fail-closed before broker submit | NOT-A-BUG (comprehensive) |
| info | account_survival.py:836-869 `_assert_single_micro_sizing` | D-3 parity guard fails C11 closed if any live lane max_contracts>1 or builder raises | NOT-A-BUG (fail-closed, matches fab6292d) |
| info | account_survival.py:389-459 `_load_lane_trade_paths` | uses end_date=as_of_date (point-in-time, no look-ahead); contracts_per_trade=1 per doctrine | NOT-A-BUG (no look-ahead) |
| info | account_survival.py:677-803 `simulate_survival` | Monte Carlo samples daily scenarios with replacement — no future leakage; FIXED starting equity (conservative, overstates DD) | NOT-A-BUG (no look-ahead) |

**FIXED-safe:** none (no real findings — every capital path on the two surfaces is
already hardened by the known-landed fixes; all guards verified fail-closed by trace).

**SURFACED-OPEN:** none new. (Pre-existing D-3 seam — survival gate models 1-micro while
engine would size from equity once clamp lifts — is already tracked in memory as a
design-ready stage `project_d3_seam_stage1_design_audited_2026_06_07`; the live gate
correctly fails closed today via `_assert_single_micro_sizing`, so it is NOT an active
capital-loss bug. Not re-surfaced.)

**Net:** files touched = 1 (this ledger only — append + 2 NOT-A-BUG registry entries; zero
production-code edits). drift = not re-run (no code edits → state unchanged from main
agent's clean baseline; drift check itself timed out >180s on git-subprocess fan-out, an
env/timing issue not a code failure). tests = none run (no edits to verify). **Clean
iteration — zero real capital-risk findings on the two highest-priority surfaces.**

---

## Iteration 2 (2026-06-07)

**Surfaces audited:** (1) `trading_app/execution_engine.py` — full read: ORB/IB state
machine, `_compute_contracts` sizing application, all 3 entry-model fill paths
(E1/E2/E3), risk-manager derisk, size-multiplier application, exit/scratch P&L.
(2) `scripts/run_live_session.py` — preflight chain, mandatory pre-launch arming gate,
`_check_telemetry_maturity` (Express-funded waiver), `_check_survival_report`,
`_check_live_readiness_report`, `_check_project_pulse_for_live`, account-selection.
(3) `trading_app/prop_profiles.py` — `is_express_funded` classifier, `max_contracts`
caps, `resolve_execution_order` divisor fail-closed, all ACCOUNT_PROFILES sizing.
(4) `pipeline/cost_model.py` — COST_SPECS slippage/commission completeness for active
instruments + session-slippage multipliers. **All 4 surfaces covered — none deferred.**

**Cross-checked downstream (BOTH directions):** `trading_app/portfolio.py`
`build_profile_portfolio` (max_contracts=1 hardcode), `compute_position_size_vol_scaled`,
`session_orchestrator.py` engine instantiation + order-qty submit/exit paths.

**Findings table:**
| severity | file:line | one-liner | disposition |
|---|---|---|---|
| info | execution_engine.py:259-302 | `_compute_contracts` clamps to max_contracts; live profile path = 1 micro (portfolio.py:877) | NOT-A-BUG (fail-closed) |
| info | execution_engine.py:982-983,1203,1378 | HALF_SIZE `max(1,int())` floor = no-op at 1ct; conservative | NOT-A-BUG (documented) |
| info | execution_engine.py:1496-1504 | ambiguous bar (hit_target AND hit_stop) → conservative LOSS at stop | NOT-A-BUG (correct) |
| info | execution_engine.py:719-728,1059-1078 | unknown filter_type / unknown entry_model → fail-closed REJECT, no arm | NOT-A-BUG (correct) |
| info | run_live_session.py:1364-1403 | mandatory unconditional pre-launch preflight gate, no skip flag (755cdd7e) | NOT-A-BUG (hardened) |
| info | run_live_session.py:599-693 | telemetry waiver Express-funded-only; real-capital fail-CLOSED | NOT-A-BUG (doctrine-correct) |
| info | cost_model.py:97-251 | active MGC/MNQ/MES all nonzero cost; fail-closed on unknown | NOT-A-BUG (no cost illusion) |
| low(note) | session_orchestrator.py:2911-2924 | EXIT divisor-raise `return`s without flatten (asymmetric vs entry emergency-flatten) | NOT-A-BUG (dormant — unreachable for entered positions; see below) |

**FIXED-safe:** none — no real capital-loss findings on any of the 4 surfaces. Every
sizing / arming / cost path traced fail-closed.

**SURFACED-OPEN:** none requiring operator decision.

**Dormant asymmetry (logged, not a live bug):** On the EXIT path
(session_orchestrator.py:2911), a non-divisible execution qty raises ValueError →
logs CRITICAL + notifies + writes EXECUTION_QTY_REJECTED + `return`s, WITHOUT
flattening. The ENTRY path (`_submit_bracket`:2214) handles the same raise inside its
try/except → emergency-flatten. The asymmetry would matter ONLY if a position could
exist at the broker while its exit qty is indivisible. TRACE: (a) all active
ACCOUNT_PROFILES have `execution_qty_divisor=None` → `resolve_execution_order` returns
early at prop_profiles.py:218 (divisor=1, never raises); (b) entry and exit use the same
`event.contracts` source, so any position that ENTERED had a divisible qty and its
same-qty exit is also divisible. VERDICT: unreachable for entered positions today.
Defensive-hardening candidate (make EXIT-block also flatten) if NQ-mini substitution
(execution_qty_divisor) is ever activated on a live profile — not a current capital bug.

**Net:** files touched = 1 (this ledger only — Iteration 2 block + 7 NOT-A-BUG registry
entries; zero production-code edits). drift = not re-run (no code edits → unchanged from
clean baseline; drift fan-out times out >180s = env, not code). tests = none run (no
edits). **Clean iteration — all 4 deferred surfaces audited, zero real capital-risk
findings; one dormant asymmetry logged for future NQ-mini activation.**

## Iteration 3 (2026-06-07) — NO-NEW-CODE FAST PASS

**Ground truth:** HEAD unchanged since Iterations 1-2 (still 1 docs-only commit ahead of
origin/main: 97161aad). Working tree clean except HANDOFF.md + the two untracked loop
ledgers. `git rev-list --left-right --count origin/main...HEAD` = 0/1. NO capital code
shipped since the last deep audit.

**Action:** Per loop-memory design + subagent-budget rule, did NOT re-dispatch a full
deep-audit subagent — re-reading the identical unchanged 6-file surface that Iterations
1-2 already cleared (zero findings) would burn ~150-180K tokens for zero new information.
That is the exact anti-pattern the budget rule forbids. Iteration 3 = recorded no-op.

**Findings:** none (no new code to find them in). All prior dispositions stand.

**Net:** files touched = 1 (this ledger). drift = N/A (no code change). tests = N/A.
**Rule for future iterations:** deep-audit ONLY when `origin/main...HEAD` shows NEW capital
code (trading_app/live/, execution_engine.py, run_live_session.py, account_survival.py,
prop_profiles.py, cost_model.py) or the working tree dirties one of them. Otherwise record
a fast no-op like this one. The full priority surface is already audited clean.

---

## Iteration 4 (2026-06-07) — NO-NEW-CODE FAST PASS

**Ground truth:** HEAD still 97161aad (unchanged since Iter 1-3; 1 docs-only commit ahead
of origin/main). `git diff --name-only HEAD` = HANDOFF.md only. `git diff --cached` empty.
0/1 left-right vs origin/main. No capital code shipped.

**Commands run:** `git diff --name-only HEAD` → HANDOFF.md ; `git diff --cached --name-only` → (empty)

**Action:** No-op per the Iter-3 rule (deep-audit only on NEW capital code). No subagent
dispatched — the 6-file priority surface is already audited clean (Iter 1-2). Re-reading
unchanged code would burn ~150K tokens for zero new findings (subagent-budget anti-pattern).

**Findings:** none. All prior dispositions stand. **Grade carry-forward: A** (no new code to
regrade).

**Net:** files touched = 1 (this ledger). drift = N/A. tests = N/A.

---

## Iteration 5 (2026-06-07) — BROKER-ROUTING + COPY-FAN-OUT CLUSTER (new ground)

**Ground truth:** HEAD unchanged (97161aad, 1 docs-only ahead of origin/main; working tree
clean except HANDOFF.md + untracked ledgers). Per the rotation-queue rule, audited the
next UNREVIEWED capital-adjacent surfaces (NOT diff-gated — idle session, no new code, but
the prior 6-file surface is exhausted-clean so this is genuinely new ground).

**Surfaces audited (full read + downstream trace):**
1. `trading_app/live/copy_order_router.py` (273 ln) — HIGHEST exposure (×N fan-out).
2. `trading_app/live/broker_base.py` (318 ln) — the BrokerRouter/Auth/Positions ABCs.
3. `trading_app/live/broker_connections.py` (294 ln) — credential store + auth lifecycle.
4. `trading_app/live/position_tracker.py` (248 ln) — in-memory position FSM.

**Downstream/upstream traced (both directions):** `session_orchestrator.py:656-672`
(CopyOrderRouter construction), `:1943-1953` (proactive divergence kill+flatten),
`:2233-2649` (qty→spec→submit entry path), `projectx/order_router.py:195-238` (submit
success="submitted" / failure RAISES semantics), `projectx/contract_resolver.py:74` +
`tradovate/contracts.py:26-41` (resolve_all_account_ids / >1-account fail-closed,
landed 18fdfa17).

**Findings table:**
| severity | file:line | one-liner | disposition |
|---|---|---|---|
| info | copy_order_router.py:69-121 submit() | fan-out: primary-first, ProjectX failure RAISES (never non-skip status) → shadows unreached on primary fail; shadow fail marks DEGRADED (2 backstops) | NOT-A-BUG (fail-closed) |
| info | copy_order_router.py:58-67 raw qty copy | single primary qty copied to shadows = intentional copy-trade; 1-micro clamp makes per-cap divergence unreachable | NOT-A-BUG (design assumption) |
| info | copy_order_router.py:127-169 cancel / bracket-cleanup | primary fail-closed (raises); shadow best-effort marks DEGRADED → next submit raises + orchestrator halts | NOT-A-BUG (correct) |
| info | session_orchestrator.py:1943-1953 | proactive is_degraded() check every 10 bars → kill switch + emergency_flatten (primary) → all-flat | NOT-A-BUG (fail-closed) |
| info | broker_connections.py:217-247 | per-conn connect() fail-closed (re-raise, status=error, auth cleared); connect_all_enabled swallow is preflight-UI path, not an arm gate | NOT-A-BUG (correct) |
| info | position_tracker.py:64-71,131-148,176-181 | in-memory FSM; late-fill-no-resurrect (R2-H3), dup-fill ignore, exit-before-fill reject, double-entry block — all fail-closed; not the broker-truth source | NOT-A-BUG (correct) |
| info | broker_base.py:178-238 | (None,None)/True/False bracket-verify defaults are conservative; CopyOrderRouter delegates all 3 to primary so live path inherits real verification | NOT-A-BUG (correct) |

**FIXED-safe:** none — no provable capital-loss bug on any of the 4 surfaces. Every
fan-out / routing / auth / position-FSM path traced fail-closed with a line citation.

**SURFACED-OPEN:** none requiring an operator decision. (The raw-qty-copy is a documented
design assumption that is dormant under the 1-micro clamp; it would need re-review IF the
clamp lifts AND copy-trade is run across DIFFERENT-tier accounts simultaneously — logged
above as a design assumption, not an active bug.)

**Test coverage check:** `tests/test_trading_app/test_copy_order_router.py` already pins the
F-2b invariants (degradation, multi-shadow partial failure, raise-propagation, skip-status
matrix, all delegations) — 33/33 PASS (ran this iteration). No untested capital invariant
gap warranted a new test, so none added (per the Tier-B "tests only when they pin a real
gap" posture — fabricating a redundant test would be theater).

**Net:** files touched = 1 (this ledger only — Iter 5 block + 6 NOT-A-BUG registry entries +
3 queue [x]; ZERO production-code edits). drift = not re-run (no code edits → state unchanged
from clean baseline; drift fan-out times out >180s = env, not code). tests = 33 pass
(copy_order_router suite, verifying the existing F-2b pins are green). **Clean iteration —
the highest-exposure surface (×N copy fan-out) plus broker routing/auth and the position FSM
all audited, zero real capital-risk findings. The copy-router fail-closed design (primary-
first + raise-on-primary-fail + dual divergence backstops) is sound.**

---

## Iteration 6 (2026-06-07) — WEBHOOK + HTTP-CLIENT CLUSTER (external order-trigger surface)

**Ground truth:** HEAD 97161aad (2/1 vs origin/main — 1 docs-only ahead, 2 behind; no NEW
capital code). Audited the next rotation-queue cluster: the EXTERNAL order-trigger surface.

**Surfaces audited (full read + caller/launcher trace both directions):**
1. `trading_app/live/webhook_server.py` (410 ln) — TradingView→Tradovate webhook /trade.
2. `trading_app/live/http_client.py` (437 ln) — canonical broker REST retry client.
**Traced:** `scripts/run_webhook_server.py` (launcher: 127.0.0.1 bind, --live secret≥16 +
CONFIRM gate), `session_orchestrator.py:3946-3952` (webhook = separate feedless-broker
entry path, NOT the feed-driven live engine), `tests/test_trading_app/test_webhook_server.py`
(16 tests after this iter).

**HIGHEST-CONCERN verdict (unauthenticated→order):** REFUTED with citations. Lifespan
RAISES if WEBHOOK_SECRET empty (server won't start, :154-155); /trade HMAC-compares secret
→ 403 on any mismatch (:323); req.secret defaults "" so a no-secret caller is rejected.
Bound to localhost; external only via named cloudflared tunnel. No bypass path to
`_place_order`. NOT a critical.

**Findings table:**
| severity | file:line | one-liner | disposition |
|---|---|---|---|
| MEDIUM | webhook_server.py:403,408 vs 336/344 | dedup + position-limit increments are POST-await; 2 concurrent identical entries both submit (TOCTOU) | SURFACED-OPEN (proven by new char-test) |
| info | webhook_server.py:154-155,323,171 | no unauth→order path; lifespan+HMAC fail-closed; localhost bind | NOT-A-BUG (auth correct) |
| info | webhook_server.py:164-201,328,352 | Pydantic + allowlist + post-division qty-cap; malformed → reject (fail-CLOSED) | NOT-A-BUG (validation correct) |
| info | http_client.py:280-294,409-414 | non-2xx never = success; 200+success:false → BrokerProtocolError no-retry; record_success only post-parse | NOT-A-BUG (status handling correct) |
| info | http_client.py:84-100,233-256,205 | retries gated by deadline budget; placement idempotency at call site; TLS verify=True (default, not disabled) | NOT-A-BUG (no double-send / no TLS bypass) |

**FIXED:** none (the one real finding is SURFACED-OPEN — fixing it changes order-trigger
timing on a capital path = Tier B, operator decision required, NOT silently edited).

**Test ADDED (Tier-B permitted — pins a real capital invariant gap):**
`test_concurrent_duplicate_entries_both_submit_toctou` in test_webhook_server.py. Proves the
TOCTOU: two concurrent identical entry alerts → `submit_calls==2` and `_OPEN_POSITIONS["MGC"]==2`
despite MAX_OPEN_POSITIONS=1 and the dedup window. Pins CURRENT (unsafe) behavior so a future
pre-await dedup-reserve fix flips it RED and forces a conscious behavior change. (Also surfaced
a latent test-isolation leak: module-global WEBHOOK_PROFILE_ID persists across `_load_ws`
re-imports — handled defensively by resetting it in the new test; not a production bug.)

**SURFACED-OPEN (operator decision):** The dedup/position-limit TOCTOU. WHY operator-gated:
the gap is real and proven, but the only fix that closes it (reserve dedup key + increment
position counter BEFORE `await _place_order`, roll back on failure) alters WHEN the order
trigger commits its dedup/cap state — a behavior change on the live order-trigger path that
Tier B forbids editing silently. Mitigations already present: rate-limit (≤3/60s, increment
IS pre-await) caps the blast to 3 orders; TV serial-fire is the normal case; demo-default;
webhook path is not the session_orchestrator feed-driven engine. Options for operator:
(a) accept as bounded, or (b) approve the pre-await reserve+rollback fix.

**Net:** files touched = 2 (this ledger + test_webhook_server.py — 1 new test; ZERO
production-code edits). tests = 16 pass (full webhook suite incl. the new char-test).
drift = NOT re-run (no production-code edit → state unchanged from clean baseline; drift
fan-out times out >180s = env/timing, not code — per the Iter-1..5 standing note). **One
real MEDIUM TOCTOU surfaced + proven; auth/validation/HTTP-status/retry/TLS all traced
fail-closed with line citations.**

---
