# MAX-EV Extraction Campaign — Sequenced Execution Plan

**Created:** 2026-04-19
**Owner:** canompx3
**Status:** LOCKED — awaiting Phase 0 execution
**Authority:** `.claude/rules/institutional-rigor.md`, `.claude/rules/branch-discipline.md`, `.claude/rules/backtesting-methodology.md`, `docs/institutional/pre_registered_criteria.md`, `trading_app/holdout_policy.py`

**Derives from:** 2026-04-19 MAX-EV honest extraction audit + ordering corrections. Supersedes the raw audit's 10-priority list; the raw audit is retained as orientation only.

---

## Ordering corrections the audit missed

**1. Mode A refresh precedes SR-monitor diagnosis, not the other way round.** The 2 "false blocker" lanes (NYSE_OPEN COST_LT12 RR1.0, TOKYO_OPEN COST_LT12 RR1.5) show canonical OOS > IS on `orb_outcomes`. SR monitor alarms on a different stream. The most likely divergence source is that SR monitor compares OOS actuals against the Mode-B-inflated stored `expectancy_r` baseline — so OOS looks relatively worse even when canonical Mode A shows the lane is fine. If that hypothesis is right, Mode A refresh auto-resolves 2 of 4 SR alarms with zero code change to `sr_monitor.py`. Diagnose against the corrected baseline, not the contaminated one.

**2. Cost-screen reclassification is a doctrine amendment before it's a code change.** COST_LT12 at 98.7% fire-rate and ORB_G5 at 95.3% violate RULE 8.1 — agreed. But the fix is not "reclassify in a registry." The fix is: (a) amend `docs/institutional/pre_registered_criteria.md` to make fire-rate ≥ 95% a hard promotion blocker (currently it's a flag-not-block), (b) write a NO-GO / reclassification pre-reg under the new criterion, (c) THEN update any registry/allocator code to match. Skipping the doctrine step is the same band-aid failure mode `.claude/rules/institutional-rigor.md` forbids.

---

## Sequenced execution plan

### Phase 0 — Branch hygiene (do this first, always)

Before any PR is opened this session:

```
git fetch origin
git status
git log --oneline origin/main..main
```

If local `main` is 14 commits ahead of `origin/main`, push before branching anything new. `.claude/rules/branch-discipline.md` + the 2026-04-18 a88505cd incident: do not branch new PRs from a local `main` that's ahead of `origin/main` — every new PR diff will bleed those 14 commits. Either push them or branch new work from `origin/main` explicitly.

Also flush the 10 dirty working-tree files — either commit to a named branch, stash, or revert. Don't carry uncommitted state into the campaign.

### Phase 1 — Zero-risk unblocks (parallel, no dependencies)

Execute immediately, no design gate needed:

**1.1 CI workflow ui/ fix.** Merge the one-line diff in PR #9 body — or cherry-pick to a named branch off `origin/main` if PR #9 is blocked on the ruff sweep. This unblocks every future PR from green. Zero blast radius outside CI.

**1.2 SINGAPORE_OPEN ATR_P50_O15 filter-type re-verification.** The audit's own self-check flagged this as unverified — `filter_type` was derived from `strategy_id` suffix, which has mis-mapped before. Re-query `validated_setups` for the exact `filter_type` string on that lane; re-run the canonical SR-alarm diagnostic with the corrected mapping. Deliverable: 2-line confirmation or a corrected alarm status.

**1.3 Stage-file hygiene.** The `htf-path-a-build.md` leftover was PR #7's fix. Confirm `docs/runtime/stages/` has zero stale files from prior campaigns; if any linger, delete. Hook design for leak prevention is deferred per audit Priority 5 — fine, but the manual sweep is free.

### Phase 2 — Canonical-truth refresh (gates Phase 3)

**2.1 Mode A re-validation of all 38 active `validated_setups`.**

Prerequisites:
- Script already exists: `research/mode_a_revalidation_active_setups.py`.
- `docs/institutional/pre_registered_criteria.md` already locks `HOLDOUT_SACRED_FROM = 2026-01-01`.
- `trading_app.holdout_policy.enforce_holdout_date()` is the enforcement path.

Execute: re-run the script against `orb_outcomes` as of current canonical cutoff. For each of the 38 lanes, compute Mode A ExpR, N, t, Sharpe, year-stability. Write results to `docs/audit/results/2026-04-19-mode-a-revalidation-refresh.md` with one row per lane: `strategy_id | stored_expR_modeB | canonical_expR_modeA | delta | drift_direction | promotion_status_under_criteria`.

Do NOT update `validated_setups.expectancy_r` in the DB as part of this phase. The long-term fix is making the allocator recompute at run-time; overwriting stored values just creates a different staleness class. Ban stored ExpR for allocation in code (Phase 3.2), leave the stored column as historical record.

**2.2 Fire-rate >95% audit across all active `validated_setups`.** Query fire-rate per lane on pre-2026 canonical. List every lane > 95%. Cross-reference against lanes > 98% (near-pass-through, trivially cost-screen). Write to `docs/audit/results/2026-04-19-fire-rate-audit.md` with ARITHMETIC_ONLY flag per RULE 8.2 (`|wr_spread| < 3%` AND `|ΔIS| > 0.10`).

**2.3 SINGAPORE_OPEN O15 vs O30 Jaccard decomposition.** The audit's missed-angle #5. Canonical per-day fire masks for both apertures on the SGP lane; pairwise Jaccard. If overlap > 0.7, treat as single effective lane (2026-04-19 HTF Path A / rel_vol cross-instrument lesson class). This may materially change the current "SGP has two active lanes" count in the deployable view.

**2.4 Cross-session momentum SGP→EUROPE_FLOW portfolio re-evaluation.** The audit called this a "self-revised KILL → genuine edge blocked by correlation gate" but left it off the priority list. That's a portfolio-optimization decision, not a research decision — evaluate options B/C/D against current live book with Mode-A-corrected ExpR for both sides. Not urgent, but shouldn't drop off the queue.

### Phase 3 — Diagnoses and code changes (gated by Phase 2)

**3.1 SR-monitor stream source audit.** Only after Phase 2.1 refresh completes. For the 2 remaining false-alarm candidates (NYSE_OPEN COST_LT12 RR1.0, TOKYO_OPEN COST_LT12 RR1.5) — if Mode A refresh didn't auto-resolve them, trace what `sr_monitor.py` reads:
- What stream? (`paper_trades`, `canonical_forward`, other)
- What baseline? (stored `expectancy_r`, recomputed Mode A, hardcoded)
- What fire-day filter logic? (`filter_signal` delegation or inline)

2026-04-19 OVNRNG_100 lesson applies: if `sr_monitor.py` inline-re-implements any filter, that's a silent drift source. Check before diagnosing symptoms.

Deliverable: `docs/audit/results/2026-04-19-sr-monitor-stream-audit.md`. If the divergence root cause is Mode B baseline, document and cross-reference Phase 3.2.

**3.2 Allocator: ban stored `expectancy_r` for allocation decisions.** Design-proposal-gate item — do NOT write code until we've agreed on approach. Candidate designs:

- **(A) Runtime recompute.** Every allocator run, recompute Mode A ExpR for each candidate lane from canonical `orb_outcomes`. Most faithful; highest runtime cost; cleanest invariant.
- **(B) Nightly refresh job.** Write a nightly batch that refreshes a `validated_setups_live_view` with Mode A ExpR; allocator reads the view. Lower runtime cost; introduces a staleness window (typically <24h).
- **(C) On-demand refresh + versioned stored.** Stamp `expectancy_r` with `computed_at` and `computed_under_holdout`; allocator refuses any row with `computed_under_holdout != 'mode_a'` or `computed_at < cutoff`. Mid-option.

I'd lean (B) for simplicity + (C)'s staleness guard grafted on. Tell me your preference.

### Phase 4 — Design-proposal-gate items (no code without sign-off)

**4.1 PP-167 per-(session, instrument) cap schema change.** Audit surfaced three options; write a proper design proposal at `docs/plans/2026-04-19-pp167-per-session-instrument-cap.md` covering:
- (a) Schema change: `DailyLaneSpec.max_orb_size_pts` becomes `Dict[(session, instrument), int]`. Faithful, higher blast radius.
- (b) `min()` reducer at caller. Cheapest, loses per-instrument information.
- (c) Composite cap object. Middle ground.

Blast radius map: every consumer of `DailyLaneSpec.max_orb_size_pts`. Grep comprehensively; I want the full list before picking (a).

**4.2 Cost-screen doctrine amendment.** Proposed amendment to `docs/institutional/pre_registered_criteria.md`:

- New criterion #13: `fire_rate ∈ [5%, 95%]` as a hard promotion blocker, not advisory. Existing RULE 8.1 is a flag; elevate to block.
- New criterion #14: `|wr_spread| < 3%` AND `|ΔIS| > 0.10` triggers ARITHMETIC_ONLY classification; lane routes to a separate `cost_screens` registry with allocation rules distinct from filter lanes (separate correlation gate class, separate capacity bucket, separate monitoring).
- Literature citation: extract from `docs/institutional/literature/` — Bailey-Lopez de Prado on backtest overfitting + Harvey-Liu multiple testing penalty justify stricter fire-rate gate for "filter" claims. If no verbatim extract covers this explicitly, register it as training-memory-grounded and apply Chordia t ≥ 3.79 for any borderline cost-screen claimed as a filter.

After amendment is merged, write the reclassification pre-reg. After pre-reg, code changes to registry and allocator.

### Phase 5 — Implementation drag cleanup

**5.1 PR #9 ruff sweep.** Coordinate rebases per the blast-radius table. Merge when all 4 worktrees reach pause point. Phase 0 must be complete first so the rebase base is clean.

**5.2 Stage-file leak prevention hook.** Low priority per audit. Defer to next sprint; write a design note only.

### Phase 6 — Missed angles the audit flagged but didn't prioritize

Queue for a follow-up pre-reg cycle (not this campaign):

- **Direction-asymmetry stratification of the rel_vol_HIGH_Q3 5 "universal" lanes.** Audit missed-angle #7. Mechanism_priors.md-aligned, cheap pre-reg, K=5 (or K=10 with LONG/SHORT split). If asymmetry exists, the "universal confluence" claim needs amending.
- **MGC 2026 OOS sanity count.** Audit missed-angle #8. Not a promotion pre-reg — informational monthly diagnostic. Auto-schedule, don't defer indefinitely.
- **HTF Path A shadow ledger.** Define the N ≥ 30 trigger explicitly in the shadow ledger doc; put a calendar reminder or check in `HANDOFF.md`.
- **3rd orthogonal signal search for rel_vol × X composite (beyond garch).** Audit missed-angle #6. Mechanism_priors.md Stage 2+ territory per Carver Ch 9-10 forecast combination. Pre-reg required, MinBTL budget applies.

---

## Audit gaps I'm adding

1. **Cascading effects not mapped.** Phase 2.1 Mode A refresh changes what the allocator sees for all 38 lanes — not just the 4 SR-alarmed ones. The entire `deployable_validated_setups` view needs re-evaluation under corrected ExpR. Correlation-gate recomputation may drop/add lanes from the live book. Surface this before Phase 3.2 ships.

2. **Filter-delegation audit not expanded to `sr_monitor.py`.** 2026-04-19 OVNRNG_100 lesson only got a partial sweep. Any code path that computes filter fires — `sr_monitor.py`, any research script, any allocator code — needs to delegate to `research.filter_utils.filter_signal`. Grep `exclude_sessions`, `filter_type ==`, and any inline filter-fire computation. Write results at `docs/audit/results/2026-04-19-filter-delegation-sweep-expanded.md`.

3. **"Book closed" verdicts are orientation-only, not canonical.** Audit says H2 garch synergy is book-closed, adversarial fade is book-closed — trusting prior audits at "orientation" level. Canonical re-audit is ~2 hours each per the self-check. If any become load-bearing for a new decision, re-verify before citing.

4. **The `filter_type` suffix derivation pattern is fragile.** The SINGAPORE_OPEN mis-mapping is one instance. Any code that parses `strategy_id` for `filter_type` is a mis-mapping hazard. Check if `validated_setups.filter_type` is the canonical column; if yes, enforce reading from there everywhere.

5. **Amendment audit trail.** If Phase 4.2 doctrine amendments land, `docs/plans/2026-04-07-holdout-policy-decision.md` is the structural template. Follow that shape for the cost-screen amendment — amendment number, rationale, literature citation, affected criteria, rollback plan.

---

## Verification protocol — applies to every phase

Before declaring any phase complete:

- `python pipeline/check_drift.py` passes (output shown).
- Relevant tests pass (output shown).
- Dead-code sweep on any new symbols (`grep -r`).
- Self-review per `.claude/rules/institutional-rigor.md` §1.
- For any filter computation in new/touched code: `grep -n filter_signal` confirms delegation.
- For any ORB window computation: `grep -n orb_utc_window` confirms canonical source.
- For any holdout-sensitive code: `grep -n HOLDOUT_SACRED_FROM` confirms canonical source.
- No inline `exclude_sessions`, `date(2026, 1, 1)`, hardcoded fire-rate thresholds, or re-implemented session time strings.

---

## Priority ordering (confirmed)

1. Phase 0 branch hygiene (unblocker, 5 min)
2. Phase 1.1 CI ui/ fix (unblocker, 5 min)
3. Phase 1.2 SINGAPORE_OPEN filter-type re-verification (2-line confirmation)
4. Phase 2.1 Mode A re-validation (truth refresh, gates Phase 3)
5. Phase 2.2 Fire-rate audit (evidence for Phase 4.2 amendment)
6. Phase 2.3 SGP O15/O30 Jaccard (lane-count correction)
7. Phase 3.1 SR-monitor stream audit (only after Phase 2.1)
8. Phase 4.1 PP-167 design proposal (design gate)
9. Phase 4.2 cost-screen doctrine amendment (design gate)
10. Phase 3.2 allocator recompute (code change, requires 4.1/4.2 and Phase 2)
11. Phase 5.1 PR #9 merge (implementation drag)
12. Phase 6 follow-up pre-regs (next sprint)

Execute phase-by-phase with verification between each. No batching. No "and also fix X" while inside a phase.

No code touches until Phases 4.1 and 4.2 design proposals are signed off. Phases 0–2 are read-only or documentation; safe to proceed on "go."

---

## Session-grounded precedents (these corrections and findings happened this campaign)

- **PR #7:** stale `htf-path-a-build.md` stage file deleted on `origin/main`. Post-hoc verified: 12/12 HTF tests pass, `verify_htf_fire_rate.py` all 6 cells in band. Justified retroactively.
- **PR #9:** 398-file ruff format sweep. Drift check identical on both branches (3/102/6). 670/671 pipeline tests pass on reformatted tree (1 fail = worktree data-dir absence, not format). Safe; coordination pending.
- **C11/C12 control-state refresh** executed (`refresh_control_state.py --profile topstep_50k_mnq_auto`). Surfaced 4 SR-ALARM blockages previously masked by fingerprint mismatch.
- **Shadow YAML `committed_sha`** filled on `research/htf-path-a-design` (`2b6f1959`) — `a1332f12` recorded per §2a placeholder-fill protocol.
- **Stash `776c1ef7`** dropped — content verified on main at `79ab4c47`.
- **Canonical SR-ALARM gap analysis** (stored baseline vs canonical OOS, filter_signal delegated):
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`: stored 0.112, canonical OOS +0.006 → gap -0.106 → REAL DECAY
  - `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`: stored 0.087, canonical OOS +0.136 → gap +0.049 → FALSE ALARM candidate
  - `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`: stored 0.109, canonical OOS +0.062 → gap -0.047 → MILD DECAY (pending Phase 1.2 filter-type re-verify)
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`: stored 0.129, canonical OOS +0.153 → gap +0.024 → NEUTRAL / FALSE ALARM candidate
- **MEMORY 2026-04-19 "Mode A drift 38/38 lanes" claim is NOT universal.** Spot-check: `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` stored 0.118 ≈ canonical Mode A IS 0.119 (zero drift); `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` stored 0.112 vs canonical 0.092 (22% drift, not 55%). Full per-lane sweep required per Phase 2.1.
- **NEW CRITICAL FINDING — X_MES_ATR60 canonical filter fires 0 rows** for `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` (stored N=673). Filter delegation integrity bug. Phase 3.1 / audit gap #2 addresses.

---

## Out-of-session coordination state

- Local `main` 14 commits ahead of `origin/main` (user WIP — `bc967d48` through `5e1ac409`).
- Local `main` 10 dirty files in working tree (`.codex/*`, `CODEX.md`, `HANDOFF.md`, `MEMORY.md`, `research/lib/__init__.py`, `tests/test_trading_app/test_prop_profiles.py`, `trading_app/prop_profiles.py`).
- Worktrees:
  - `canompx3-f5` on `followup/a2b-2-shape-e-eq9-population-fix` (ahead 16 of `origin/main`)
  - `canompx3-htf` on `research/htf-path-a-design` (ahead 1: `2b6f1959`; behind 64 of `origin/main`)
  - `canompx3-phase-d` on `phase-d-volume-pilot-d0` (ahead 4 of `origin/main`)
  - `canompx3/.worktrees/mnq-nyse-close-long-direction-locked-v1` (user-created during session)

Phase 0 resolves this state before any branching.

---

**End of original plan. See re-audit appendix below before executing.**

---

# APPENDIX — Re-audit corrections (2026-04-19, second pass)

Triggered by: (a) user reminder "always grounded in /resources literature, do not pretend to read, professional" (b) "revise, rethink, reaudit, improve plan" (c) "reground, state has changed, no tunnel vision". This appendix records what the original plan got wrong, what I actually verified by reading, and what branch-points I missed.

## A. Claims retracted (were wrong)

**A.1 X_MES_ATR60 "filter fires 0 rows canonically" — RETRACTED.**
`X_MES_ATR60` IS registered in `trading_app.config.ALL_FILTERS` (verified: 91 filters total; `X_MES_ATR60`, `X_MES_ATR70`, `X_MGC_ATR70` present). The "0 fires" result in my spot-check was because `CrossAssetATRFilter` requires `cross_atr_{source}_pct` columns **injected at runtime**, not loaded from `daily_features`. My query loaded only `daily_features.*` → filter returns False for every row.
Evidence: `research/mode_a_revalidation_active_setups.py` lines 178-179 comment: *"CrossAssetATRFilter requires cross_atr_{source}_pct which is NOT in daily_features schema — it is injected at discovery/fitness time"*. This is documented, not a bug. My finding was a query error, not a canonical integrity failure.

**A.2 "Mode A drift 38/38 lanes" — still not universal per spot-checks, but:**
- `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100`: stored 0.118, canonical Mode A IS 0.119 → **zero drift**
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`: stored 0.112, canonical Mode A IS 0.092 → **22% drift**
Per-lane Phase 2.1 run is still needed; the MEMORY claim of "38/38 material drift" is not supported by the 2 canonical spot-checks done.

## B. Claims verified against canonical sources

**B.1 Chordia t ≥ 3.79 — VERIFIED against extract.**
`docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` line 20 verbatim: *"MHT threshold for alpha t-statistic (t_α) is **3.79** while that for FM coefficient t-statistic (t_λ) is **3.12**"*. At q=0.05 + 5% FDP tolerance with correlation control (FDP-StepM method). Project already codifies this as criterion #4 in `pre_registered_criteria.md`: "t ≥ 3.00 (w/ theory) or 3.79 (w/o)". No amendment needed — already live.

**B.2 `research/mode_a_revalidation_active_setups.py` is canonical-integrity verified.**
Docstring + code (lines 1-179 read): reads `validated_setups.filter_type` (canonical column, not suffix-derived); delegates to `research.filter_utils.filter_signal`; filters by direction from `execution_spec`; applies strict `trading_day < HOLDOUT_SACRED_FROM`; flag tolerances ΔN/stored_N > 0.10, |ΔExpR| > 0.03, ΔSharpe > 0.20; read-only (writes NOTHING to validated_setups); output: `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`. Safe to run as Phase 2.1.

**B.3 Fire-rate 95% rule is PROJECT-ENGINEERING, not extract-literature.**
`.claude/rules/backtesting-methodology.md` RULE 8.1 is the source. No extract in `docs/institutional/literature/` directly supports a 95% fire-rate threshold. Chordia extract (verified) supports t-stat thresholds and MHT methods — not fire rates. Phase 4.2 amendment must label the fire-rate rule honestly as "project-local engineering rule, consistent with Chordia FDR framework but not explicit in extracted literature." Not a defect; just honest provenance.

## C. Plan gaps (original plan was loose)

**C.1 Phase 1.2 (SINGAPORE_OPEN filter re-verify) is redundant with Phase 2.1.**
Mode A script re-validates every active lane including SINGAPORE_OPEN, reading `filter_type` from the canonical column. Drop Phase 1.2 as a standalone step; covered by 2.1.

**C.2 Output filename mismatch.**
Plan said `docs/audit/results/2026-04-19-mode-a-revalidation-refresh.md`. Actual script writes `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`. Use the script's path.

**C.3 Phase 4.2 should AMEND existing criteria, not invent #13/#14.**
`pre_registered_criteria.md` already has 12 criteria and an amendment format from 2026-04-07 (Amendment 2.7 holdout policy). Phase 4.2 should follow that template as Amendment 2.8 adding fire-rate threshold + ARITHMETIC_ONLY classification. Precedent for grandfathering existing lanes exists in `research-truth-protocol.md` Phase 0 section.

**C.4 Phase 2.2 fire-rate audit must invoke T1 per `quant-audit-protocol.md`.**
Any lane flagged `fire_rate > 95%` must run the Win Rate Monotonicity test (T1): does WR vary >5% across quintiles of the filter variable, or is WR flat with only payoff rising? WR-flat + payoff-rising = ARITHMETIC_ONLY (cost screen); WR-monotonic = SIGNAL. The fire-rate flag alone is insufficient to classify.

**C.5 Phase 2.2 must scope to validated_setups (validated-universe-rule).**
`research-truth-protocol.md` § Validated Universe Rule: "NEVER run research queries against the full unfiltered orb_outcomes." Fire-rate audit joins validated_setups → orb_outcomes with the exact filter applied per-lane. Not a raw scan.

**C.6 Mode A drift ≠ SR-gap analysis.**
Plan conflated two distinct comparisons:
- **Mode A drift:** stored IS ExpR (possibly Mode B grandfathered) vs canonical Mode A IS ExpR. Answers: "is the stored baseline inflated?"
- **SR-gap:** stored ExpR (baseline) vs canonical OOS ExpR post-2026-01-01. Answers: "is current OOS performing above or below the stored baseline?"
Both are valid; different questions. Phase 2.1 answers Mode A drift. Phase 3.1 SR-monitor audit is SR-gap relevant. Don't use one to claim the other.

**C.7 Worktree state is DYNAMIC.**
At plan-write time: 4 worktrees. Now: **5 worktrees** — user created `canompx3/.worktrees/campaign-2026-04-19-phase-2` on `research/campaign-2026-04-19-phase-2` (parallel campaign work). `canompx3-f5` has moved to `fix/check-37-honor-duckdb-path`. Phase 5.1 (PR #9 rebase coordination) must query `git worktree list` live, not trust plan-time snapshot. Don't step on in-flight work in user's parallel worktree.

**C.8 Chordia extract DOES NOT support fire-rate threshold.**
My original Phase 4.2 paragraph said "Bailey-LdP on backtest overfitting + Harvey-Liu multiple testing penalty justify stricter fire-rate gate" — training memory, no extract verification. I have now read only the Chordia extract; it is about t-statistic thresholds and MHT method choice, not fire rates. Harvey-Liu and Bailey-LdP extracts are unread. Until read, Phase 4.2 literature claims stay labeled UNSUPPORTED.

## D. Branch-points missed (anti-tunnel-vision)

The original plan assumed the truth-refresh campaign will proceed linearly. Three outcomes of Phase 2 that would redirect the plan:

**D.1 If Phase 2.1 Mode A refresh shows MINIMAL drift across 38 lanes** (consistent with EUROPE_FLOW OVNRNG_100 spot-check), then Phase 3.2 "ban stored expectancy_r" is unnecessary. The allocator is NOT being fed inflated baselines. Plan should branch: if drift-median < 0.02 R across lanes → skip Phase 3.2; if drift-median > 0.05 R → execute Phase 3.2 (B)+(C) design.

**D.2 If Phase 2.2 fire-rate audit shows >20 of 38 lanes > 95%**, the book is mostly cost-screens, not filter edges. A doctrine amendment alone doesn't address the portfolio-level implication. Plan should branch: add a portfolio-phase-out step where cost-screens route through a separate capacity bucket (per Phase 4.2 proposal's criterion #14) vs. retirement entirely.

**D.3 SR-ALARM "false blocker" frame has an alternative interpretation.**
My audit assumed: "stored baseline inflated → canonical OOS > stored → ALARM is false." Alternative: "canonical OOS N is 54-72 (small), single-month outliers dominate (e.g., TOKYO_OPEN Jan +0.359 is outlier), and the SR path-dynamic correctly detects within-stream drift even when aggregate OOS > baseline." SR is path-dependent; aggregate gap analysis is not a sufficient proxy. Phase 3.1 must do the path-walk, not just the gap.

## E. Coordination with user's parallel worktree

User created `research/campaign-2026-04-19-phase-2` during this session. Before executing anything in Phase 2, I must:
1. Inspect what commits that worktree has (if any).
2. Confirm which Phase 2 items they are working vs. which are open for me.
3. Not run Phase 2.1 Mode A script if user is already running it — it writes a deterministic output file and two parallel runs are a race condition.

## F. Consolidated revised priority (supersedes original)

0. **Phase 0 — branch hygiene + coordination check.** Check user's `campaign-2026-04-19-phase-2` worktree state. Decide disposition of 14 unpushed commits + 7 dirty files (count reduced from 10 during session). `research/lib/__init__.py`, `tests/test_trading_app/test_prop_profiles.py`, `trading_app/prop_profiles.py` may have already been committed.

1. **Phase 1.1 — CI ui/ workflow fix.** (User-push; my OAuth token lacks workflow scope.)

2. **Phase 1.3 — stage-file hygiene sweep.** (Trivial read-only; confirm `docs/runtime/stages/` has only `.gitkeep`.)

3. **Phase 2.1 — Mode A re-validation.** Script is integrity-verified. Coordinate with user worktree first. Output at `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`.

4. **Branch point D.1: evaluate Phase 2.1 drift distribution.** If drift-median < 0.02 R across 38 lanes, skip Phase 3.2. Else proceed.

5. **Phase 2.2 — fire-rate audit scoped to validated_setups, with T1 monotonicity per flagged lane.** Label flags SIGNAL / ARITHMETIC_ONLY / UNVERIFIED. Output at `docs/audit/results/2026-04-19-fire-rate-audit.md`.

6. **Branch point D.2: evaluate Phase 2.2 fire-rate distribution.** If >20 of 38 > 95%, add portfolio-phase-out step.

7. **Phase 2.3 — SGP O15/O30 Jaccard.** Unchanged.

8. **Phase 3.1 — SR-monitor stream source audit.** Must do the PATH WALK (per-alarm SR statistic reconstruction on the actual stream), not just aggregate gap. D.3 branch-point.

9. **Phase 4.1 — PP-167 design proposal.** Precondition: grep comprehensively for `DailyLaneSpec.max_orb_size_pts` consumers. Then design.

10. **Phase 4.2 — cost-screen doctrine amendment.** Precondition: READ `bailey_et_al_2013_pseudo_mathematics.md`, `bailey_lopez_de_prado_2014_deflated_sharpe.md`, `harvey_liu_2015_backtesting.md` extracts. Cite only what the extracts actually say. Label fire-rate rule as project-engineering with Chordia-framework consistency if extracts don't support a 95% threshold directly.

11. **Phase 3.2 — allocator recompute.** Gated on Phase 2.1 drift-distribution outcome. Skip if drift minimal.

12. **Phase 5.1 — PR #9 merge.** Live `git worktree list` check before coordination.

## G. Conditions for "done" on this campaign

Not "all 12 phases executed" — rather:
- Every load-bearing literature claim cites an extract file path + verbatim passage (or is labeled UNSUPPORTED).
- Every canonical stat was recomputed from `bars_1m`/`daily_features`/`orb_outcomes` within-session, not cited from MEMORY.
- Every code change follows design-proposal-gate + verification protocol.
- Every branch-point (D.1, D.2, D.3) was evaluated against evidence, not assumed.
- No new canonical integrity findings outstanding without either a fix or a documented accept-decline decision.

**End of appendix. Awaiting user's coordination signal for Phase 2 start + disposition of their parallel worktree work.**
