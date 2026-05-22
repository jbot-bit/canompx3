# ORB State Controller — Overlay Design Brainstorm (DESIGN/AUDIT ONLY)

**Mode:** DESIGN. No code, no preregs, no edits beyond this file.
**Date:** 2026-05-22
**Risk tier:** high (touches capital-allocation surface). Holdout Mode A respected throughout.

---

## Context

User asks for a brainstorm-and-rank of "ORB State Controller" overlays — Markov / transition / state-machine ideas that **improve already-validated ORB lanes** via sizing modifier, veto/pause, priority ranking, lane-health transitions, account-pressure throttle, cross-asset concordance, or winner/loser-speed state. The work is explicitly NOT new entry-strategy discovery; it is layer atop the validated lane book.

The motivating problem: the live allocator (`trading_app/lane_allocator.py`) already emits a per-lane `status` (`DEPLOY`/`PROVISIONAL`/`PAUSE`/`RESUME`/`STALE`) at monthly rebalance cadence. But the *triggers* are coarse (trailing windows, Chordia verdict, C8 OOS, Shiryaev–Roberts drift alarm). Two refinements are plausibly EV-positive WITHOUT new edge discovery: (a) **sizing modifier** that respects Carver's vol-target framework, and (b) **finer-grained lane-state transitions** that react inside the monthly window. Everything else in the requested menu has either failed before, collides with NO-GO entries, or duplicates infrastructure that already exists.

---

## Adversarial audit corrections (applied 2026-05-22)

Independent evidence-auditor pass found 9 FAILs against the first draft. Each is addressed below; this block exists so a future reader can see what was fixed and why.

1. **SR overlay #3 reopen-claim partially unsupported by extract.** The Pepelyshev-Polunchenko 2015 extract (`docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`) grounds the SR math + multi-cyclic optimality but does NOT contain a verbatim passage contrasting SR vs trailing-arithmetic. The "structurally different statistic" argument is *author synthesis* of the literature, not a literal extract claim. **Corrective:** the reopen audit (overlay 3 prerequisite) MUST treat the differentiation as an open empirical question, not a literature given.
2. **"Breakeven trail stops DEAD" NO-GO row missing from collision table.** Added below as row 7. Relevant to overlay #7 (winner/loser-speed) as exit-modifier collision.
3. **N<30 power floor not addressed in overlay 1.** Adding: lanes with fewer than 30 trades in the `rolling_sharpe_60` window default to size-multiplier = 1.0 (no modification). Per `feedback_n_unique_trading_days_floor_clustered_se.md`.
4. **K_effective arithmetic not computed.** Carver Stage-2 parked prereg K=3 + new overlay K_budget=5 → K_effective = 8 against MinBTL = 300 clean MNQ trials. Stated explicitly below.
5. **"10–20% Sharpe lift" claim unsourced.** Re-labeled as *author-estimated from Carver mechanism logic, not backtested* per `institutional-rigor.md` § 7.
6. **SR alarm + FAILED_RATIO double-counting unaddressed.** Adding: overlay 3 must explicitly check `chordia_audit_log.yaml.verdict` and skip SR-veto application on lanes where Chordia PAUSE is already active.
7. **Contract-floor clamping unaddressed for overlay 4.** Adding: throttle floor = 1 contract (broker minimum); when proportional scale would yield <1, lane skips for the day (binary not fractional).
8. **Topstep DD geometry gap.** Carver Stage-2 prereg `account_constants` block (lines 29-31, 54-56) caps vol target at 25% on $2.5K-trailing-DD prop accounts. The brainstorm must surface this for overlay 1.
9. **lane_allocation.json shared-state coordination protocol unaddressed.** Adding: any controller writing to `lane_allocation.json` MUST follow `multi-terminal-shared-file-hygiene.md` three-check protocol; the operational-surface prereg must declare scope_lock on this file.

Two PASS-with-caveats from the auditor also worth surfacing:

- **Slot-headroom claim verified.** `topstep_50k_mnq_auto` has 3 deployed lanes vs `max_slots=7` (`prop_profiles.py:492`); 4 empty slots confirm overlay #5 is "marginal under current headroom."
- **Vol-regime NO-GO wording accurate** but the plan omitted the actual research tool (`research_vol_regime_switching.py`). Added in collision table below.

---

## STATE FIRST — institutional grounding

**Authority map read.** `CLAUDE.md` § Document Authority + `docs/governance/document_authority.md`.
**Source-of-truth chain.** Canonical layers `bars_1m` / `daily_features` / `orb_outcomes`; live state in `lane_allocation.json` + `chordia_audit_log.yaml`; allocator surface in `trading_app/lane_allocator.py`; profile caps in `trading_app/prop_profiles.py`. Sizing canonical: Carver Table 25 (extract at `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`).
**Holdout Mode A respected.** `HOLDOUT_SACRED_FROM = 2026-01-01` per `trading_app/holdout_policy.py` (NB: file lives under `trading_app/`, not `pipeline/` — confirmed by glob 2026-05-22). Every prereg derived from this brainstorm will declare `--holdout-date 2026-01-01` and Criterion 8 (`OOS ExpR ≥ 0.40 × IS ExpR`, `dir_match=TRUE` with `power ≥ 0.80`).
**Tunnel-vision risk checked.** This brainstorm consumes no slot capital and re-uses Carver / Shiryaev–Roberts mechanisms already grounded in `docs/institutional/literature/`. Risk: re-litigating the April 2026 "individual-lane pause" verdict. Mitigated by requiring differentiation from that NO-GO row.

### NO-GO collisions (must differentiate or refuse)

| NO-GO row (STRATEGY_BLUEPRINT.md §5) | Why it kills the naïve version | What differentiation REOPENs |
|---|---|---|
| Vol-regime adaptive parameter switching (Mar 2026, `research_vol_regime_switching.py`) | `H0 not rejected`; static parameters indistinguishable from adaptive | "Fundamentally new regime detection method" |
| Breakeven trail stops (DEAD) | −0.12 to −0.17 Sharpe vs flat-stop baseline | None known — Sharpe degradation observed across all variants |
| Regime-conditional discovery rolling-window (Apr 2026) | Friction drag explains the regime; rolling admits ~50% FP per Carver | Microstructure evidence (non-cost) of true regime change |
| Individual-strategy pause/resume (lane_allocator.py removed Apr 2026) | Backtest: top-5 individual pause = −799R/yr vs +630R/yr session-level regime gate | Different state variable (NOT trailing trade outcomes) |
| Vol-spike avoidance (Apr 2026, REVERSED) | High vol → MFE expands p<0.0003; MAE identical. Throttle-on-high-vol = wrong sign | Nothing — direction is robustly confirmed |
| Break-speed / break-delay filter (Mar 2026) | 7.2M trades, 0 BH-FDR survivors at O15; O5/O30 direction FLIP | New mechanism explaining the flip |
| Pre-velocity × atr_vel regime interaction (Apr 2026) | Cross-section K=168 cumulative, 0 BH-FDR | Continuous-quintile interaction with different mechanism citation |

### Repo / DB truth gaps

1. **No per-lane state machine at intra-month cadence today.** `LaneScore.status` is recomputed monthly; live lane_allocation.json holds the snapshot. There is no online transition layer.
2. **`session_regime` field exists (HOT/COLD/FLAT) on `LaneScore` but is operator-classified, not computed from canonical layers.** A controller that reads it inherits operator labels.
3. **Shiryaev–Roberts monitor (`trading_app/sr_monitor.py` + `trading_app/live/sr_monitor.py`) writes `sr_status ∈ {ALARM, CONTINUE, NO_DATA}` per lane** — partially overlaps with proposed "lane-health transition" controller. New controller must not re-encode SR math (per institutional-rigor.md § 4 "delegate to canonical sources"). NB: original plan cited `pipeline/sr_monitor.py`; correct path is under `trading_app/` — confirmed by glob 2026-05-22.
4. **Carver vol-target table (Table 25) is parked in `2026-05-01-carver-stage2-vol-targeted-sizing.yaml` (status: `DESIGN_LOCKED_AWAITING_GO_AND_CHORDIA_PASS`, K_budget=3).** Any sizing-modifier overlay proposed here must either (a) supersede that prereg with explicit revoke note, or (b) be framed as a complementary state-conditional layer ON TOP OF the Carver baseline. Cannot ignore it.
5. **`daily_features` lookahead gates differ by ORB-start time (RULE 1.2).** Cross-asset concordance overlays must declare valid-domain for the partner instrument's session window.

---

## The seven proposed overlay families — full evaluation

Evaluated below per the user's requested rubric. Decisions are conservative because (a) live slot capital is finite, (b) we already burned MinBTL trials on regime / state work in 2026 Q1, and (c) the only Stage-2 sophistication tier with literature backing (Carver R3 sizing) is already in a parked prereg waiting for Chordia clearance.

### Overlay 1 — Carver-table sizing modifier (R3 POSITION-SIZE)

- **Edge claim.** Trades on lanes whose realized 60-day Sharpe sits in the upper half of the lane's IS Sharpe distribution warrant ~1.2× size; lower half warrants ~0.8×. Aggregate effect: half-Kelly-anchored vol-targeting around the lane's mean.
- **Mechanism.** Carver 2015 p.143-146 Table 25 (extract verified). Half-Kelly explicitly recommended for positive-skew systems; per-lane realized Sharpe is the natural input.
- **State variables (canonical, pre-trade knowable).** `rolling_sharpe_60` from `strategy_fitness.FitnessScore` (60-day window strictly ending at `trading_day − 1`), `realized_atr_20` for vol-target denominator, IS Sharpe percentile from `validated_setups`.
- **N<30 power floor.** Lanes whose `rolling_sharpe_60` window contains < 30 trades default to `size_multiplier = 1.0` (no modification). Per `feedback_n_unique_trading_days_floor_clustered_se.md`: clustered-SE estimator degrades on skewed cluster distributions; cannot trust the Sharpe percentile below the floor.
- **Pre-trade knowable?** YES — all inputs lag the entry decision by ≥1 calendar day. ARITHMETIC_ONLY flag: clear.
- **NO-GO collision risk.** **MEDIUM.** The existing parked prereg `2026-05-01-carver-stage2-vol-targeted-sizing.yaml` already covers this; this entry must be framed as the *operational* surface (the controller table + state-machine UI), with that prereg as the *statistical-gate* surface. We do not re-test what Carver Table 25 already authorizes.
- **Topstep DD geometry cap (BINDING).** Per Carver Table 23 + Carver p.149 + the parked prereg's `account_constants` block (lines 29-31, 54-56): on $2.5K-trailing-DD prop accounts ($50K capital), max vol target = 25%. The operational controller MUST clamp size_multiplier so realized monthly vol ≤ 25% × capital; otherwise worst-daily-loss-month exceeds DD. Self-funded accounts uncapped except by Carver half-Kelly recommendation.
- **K cost.** ~5 trials at most (one per deployed-lane cluster × 2 Sharpe tiers + 1 holdout robustness check). **K_effective = 3 (parked Carver prereg) + 5 (this overlay) = 8** against Bailey 2013 MinBTL = 300 clean MNQ horizon. K_effective is what the next prereg must declare per `feedback_k_effective_prior_trial_accounting_minbtl.md`.
- **Test design.** Pathway B K=1 (institutional, not pooled): retrospective replay of the lane book under sizing modifier vs flat sizing, compare *policy_ev_per_opportunity_r* per `conditional-edge-framework.md` § 4. Per-lane breakdown required (`pooled-finding-rule.md` flip-rate cap 25%).
- **Failure condition.** Δ Sharpe ≤ 0.05 (Harvey–Liu haircut on top of IS) OR maxDD increases vs flat-size baseline OR ≥25% of lanes flip sign vs pooled direction (heterogeneity_ack required).
- **Portfolio EV rationale.** Carver Table 23 ties vol target to prop-firm trailing-DD geometry; this is the lowest-friction sophistication step above binary filters per `mechanism_priors.md` Stage 2 roadmap. *[Author-estimated, not backtested: lift on the order of 10–20% Sharpe via redistribution across currently-FIT vs currently-DECAY lanes — labeled per `institutional-rigor.md` § 7. The actual lift number is what the parked Carver prereg measures; do not pre-commit to a number.]*
- **Decision: CONTINUE.** This is the highest-EV overlay precisely because the literature gate already exists and the parked prereg can be unblocked.

### Overlay 2 — Drift-tier veto / pause (R1 lane-health transition)

- **Edge claim.** Lanes whose Shiryaev–Roberts statistic crosses an upper alarm tier are higher-than-baseline likely to be in a real (not noise) regime break. Pausing them until SR returns to CONTINUE prevents capital loss during a degrading regime.
- **Mechanism.** Pepelyshev-Polunchenko 2015 CUSUM/SR theory; `trading_app/sr_monitor.py` already implements the running statistic per lane.
- **State variables (canonical).** `LaneScore.sr_status ∈ {ALARM, CONTINUE, NO_DATA}` + `sr_statistic` numeric. Per-lane R-multiple stream from `orb_outcomes`.
- **Pre-trade knowable?** YES — SR is computed off prior-day-and-earlier outcomes.
- **NO-GO collision risk.** **HIGH (REOPEN gate).** Collides directly with the April-2026 "individual-strategy pause/resume DEAD" row. Differentiation: that NO-GO killed pause/resume triggered by **trailing-window arithmetic** (e.g., 30-day rolling ExpR). SR is a structurally different statistic (cumulative likelihood-ratio test, not trailing average). Pepelyshev-Polunchenko 2015 grounds the SR math + multi-cyclic optimality; the **"structurally different from trailing arithmetic" differentiation is author synthesis, not a verbatim extract claim** (caught by auditor 2026-05-22). The reopen audit MUST treat the differentiation as an empirical question, not a literature given. **REOPEN criterion plausibly met IF audit shows alarm-to-recovery gap differs materially from the buried trailing-window test.**
- **Chordia overlap guard (anti-double-counting).** SR-veto MUST check `chordia_audit_log.yaml.verdict` and `LaneScore.status` first. If the lane is already in PAUSE under Chordia FAILED_RATIO or already DISPLACED, SR-veto is a no-op (it's already paused). The audit must report `n_lanes_already_paused_when_sr_alarm_fired` and exclude those from the lift calculation — otherwise SR appears to "rescue" capital that was already not at risk.
- **K cost.** ~10 trials (alarm threshold × 2 directions × 5 lane clusters). Within MinBTL.
- **Test design.** Counterfactual replay: what did each ALARM lane do in the 30 days after alarm vs the 30 days before, vs same lane in non-ALARM 30-day windows. Per-lane (not pooled-only) breakdown. Apply Carver-style haircut to the IS lift. Compute clustered-SE at `trading_day` cluster level (`feedback_clustered_se_trading_day_pooled_finding_guard.md`).
- **Quantitative kill threshold (PARK).** Adversarial audit PARKs the overlay if ANY of: (a) median alarm-to-recovery gap < 7 trading days (alarms are noise, not regime breaks); (b) ≥25% of ALARM lanes flip sign relative to pooled lift direction; (c) ≥50% of ALARM lanes were already in Chordia PAUSE at alarm time (lift double-counted); (d) clustered-t < 3.00 (sub-Chordia bar) on the alarm-veto lift after Harvey-Liu haircut.
- **Failure condition.** Any of the four kill-threshold conditions above. Reported BEFORE any prereg lands.
- **Portfolio EV rationale.** SR's role today is informational only (writes status, no veto). Wiring it into the allocator gate would close a known fail-open path. EV: modest, conservative; payoff is downside reduction, not upside.
- **Decision: CONTINUE — but only after an adversarial reopen audit explicitly compares SR alarms vs the trailing-arithmetic gate that the April NO-GO killed.** Park if differentiation cannot be demonstrated.

### Overlay 3 — Cross-instrument concordance gate (R7 confluence)

- **Edge claim.** When MNQ and MES both fire same-direction ORB breaks in the same overlapping session window (e.g., US_DATA_1000), the joint trade is higher-quality than either alone. Veto trades where the partner instrument's same-session ORB break is opposite-sign.
- **Mechanism.** Liquidity-displacement co-trading at index level (Harris 2002 Ch 4 § 4.5.2); both micros tracking the same underlying index basket.
- **State variables (canonical).** Partner-instrument `orb_outcomes.break_dir` from `daily_features` (`orb_{SESSION}_break_dir`) — confirmed safe per RULE 6.1.
- **Pre-trade knowable?** YES if `orb_{SESSION}_break_dir` is computed on bar-close-after-ORB-window (canonical). Must NOT use `break_ts`/`break_delay_min` (RULE 6.3 E2 lookahead).
- **NO-GO collision risk.** **MEDIUM.** The April-2026 `2026-04-11-mnq-cross-asset.yaml` prereg parked the K=6 family without conclusive verdict. Re-litigation requires showing this overlay is NOT a reformulation of that family — frame as *direction concordance* (binary same-sign) rather than the ATR-relative scale that the parked prereg tested.
- **K cost.** ~12 trials (3 RR × 2 apertures × 2 directions). Within MinBTL but the parked-prereg cumulative K must be added (`feedback_k_effective_prior_trial_accounting_minbtl.md`).
- **Test design.** Pass 1 unfiltered vs Pass 2 filtered (RULE 2). Cross-RR family gate REQUIRED (per `regime-and-rr-handling-framework.md` § 2: target RR + ≥1 sibling RR passes T0+T1+T2+T6).
- **Failure condition.** Sibling-RR fails core tests OR direction concordance correlates >0.70 with already-deployed VWAP gates (RULE 7 tautology).
- **Portfolio EV rationale.** Plausibly real (literature-grounded co-trading mechanism), but EV is bounded — concordance veto reduces N, not lift. Survives only if remaining selected trades show clean Δ ExpR.
- **Decision: PARK PENDING CARVER UNBLOCK.** Higher-priority work (overlay 1) blocks the slot for K-budget. Re-evaluate after Carver Stage-2 completes.

### Overlay 4 — Account-pressure throttle (R8 portfolio allocator)

- **Edge claim.** When realized account drawdown approaches the prop-firm trailing-DD floor, downscale all lane sizes proportionally so worst-daily-loss-month stays under the DD limit.
- **Mechanism.** Carver Table 23 + p.149 ("vol target > 15% requires daily recalculation"). This is risk-management math, not edge discovery.
- **State variables.** `account_hwm_tracker.py` outputs + per-lane vol target. NOT a Markov state — a deterministic linear scale.
- **Pre-trade knowable?** YES — account state is end-of-prior-day.
- **NO-GO collision risk.** None. This is engineering, not research.
- **K cost.** Zero — there is no hypothesis being tested. It is a guardrail derived from Carver Table 23.
- **Contract-floor clamp (BINDING engineering constraint).** Broker minimum tradeable contract = 1 (MNQ/MES/MGC). Proportional throttle yielding `size_multiplier < 1.0 contracts` MUST round to either 1 (continue) or 0 (skip lane for the day) — never fractional. Decision rule: if `throttle_target ≥ 0.5 contracts` → execute 1 contract; else skip lane. The skip must be logged so the operator can distinguish a throttle-skip from a normal-no-fire day.
- **Shared-state coordination (BINDING).** This overlay writes to `lane_allocation.json` (size_multiplier per lane). That file is under `multi-terminal-shared-file-hygiene.md` three-check protocol via `shared-state-commit-guard.py`. Implementation MUST: (a) declare scope_lock on `lane_allocation.json` in the stage file, (b) run the three coordination checks (sibling-commit drift, peer scope_lock, sibling-worktree heat) before mutating, (c) re-run on every commit that stages the file.
- **Test design.** Engineering verification (replay against a synthetic DD-stress sequence), not statistical.
- **Failure condition.** Replay shows the throttle either fires too late (hits DD floor) or too aggressively (chokes capital deployment), OR the contract-floor clamp produces silent skips that the operator cannot reconstruct from logs.
- **Portfolio EV rationale.** Pure capital-protection; EV is the avoided ruin event. Required infrastructure before sizing-modifier (overlay 1) can deploy at scale.
- **Decision: CONTINUE as engineering work, NOT as a research overlay.** Implements alongside or BEFORE overlay 1. Does not consume K-budget.

### Overlay 5 — Priority-ranking rotation (R8 portfolio allocator weight)

- **Edge claim.** When `max_slots` < `len(active_lanes)`, rotate slot priority by FitnessScore tier (FIT > WATCH > DECAY > STALE) rather than the current static daily order.
- **Mechanism.** No new mechanism — uses existing fitness signal. This is allocator-policy refinement.
- **State variables.** `FitnessScore.fitness_status` + `rolling_sharpe`. Canonical.
- **Pre-trade knowable?** YES.
- **NO-GO collision risk.** **LOW** but flirts with the "individual-strategy pause/resume" NO-GO. Differentiation: rotation is a *priority* change, not a pause; DECAY-tier lanes still fire when slots are available.
- **K cost.** Minimal — this is policy code, but the *effect* on portfolio Sharpe still requires retrospective verification (counts as ≤3 trials).
- **Test design.** Replay live lane history with FitnessScore-ordered priority vs current static order. Per-lane policy_ev_per_opportunity_r.
- **Failure condition.** FitnessScore tiers are too lagged (e.g., a lane goes DECAY but trade outcomes were already deteriorating long before) → rotation is reactive, not predictive.
- **Portfolio EV rationale.** Marginal at best in current slot-headroom regime (most accounts not slot-pressured). Becomes more valuable only when `len(active_lanes) >> max_slots`.
- **Decision: PARK.** Save K-budget for higher-EV ideas; revisit if accounts become slot-pressured.

### Overlay 6 — Cross-asset concordance for direction sizing (R3 + R7 hybrid)

- **Edge claim.** Combine overlay 1 (Carver size table) with overlay 3 (cross-instrument direction): full size when partner agrees, half size when ambiguous, skip when partner opposes.
- Decision: **PARK.** Composite-of-two ideas inherits the K-cost of both; cannot test before its component overlays clear. Re-evaluate after overlays 1 & 3 clear.

### Overlay 7 — Winner-speed / loser-speed Markov state (state-machine proper)

- **Edge claim.** Trades whose first 1-minute bar moves further in the direction of the entry (winner-speed state) are higher-expectancy than trades that start slow.
- **Mechanism.** Stop-cascade momentum continuation (Harris 2002 Ch 4 § 4.5.2; Chan 2013 Ch 7).
- **State variables.** First-bar return after entry.
- **Pre-trade knowable?** **NO.** First-bar return is *post-entry*, so this signal can only modify *exits* or *holding decisions*, not entry sizing. Re-classifying it as an exit signal collides with the **breakeven-trail-stops DEAD** NO-GO and the **break-speed/break-delay filter DEAD** NO-GO.
- **NO-GO collision risk.** **HIGH.** The April-2026 break-speed/break-delay test (7.2M trades, BH K=96, 0 survivors at O15, O5/O30 direction flip) covers the entry-time version of this hypothesis. The exit-time version is shielded only if it differs structurally — and the same Simpson's-paradox aperture problem likely applies.
- **K cost.** Would be high (≥30 trials across RR × aperture × confirm_bars).
- **Test design.** Would require a separate exit-rule prereg with literature mechanism explaining the aperture-flip phenomenon. Currently no such mechanism exists in `docs/institutional/literature/`.
- **Failure condition.** Same as the prior break-delay test: aperture-flip kills the mechanism claim.
- **Portfolio EV rationale.** Low — re-investigating a recently-buried verdict.
- **Decision: KILL.** No mechanism differentiation from the April-2026 break-speed NO-GO. Reopen only if Harris stop-cascade extract is supplemented with new literature explaining the O5/O30 direction flip.

---

## Ranked shortlist (max 5)

| Rank | Overlay | Type | K cost | Decision | Why this rank |
|---|---|---|---|---|---|
| **1** | **Carver-table sizing modifier (R3)** | Sizing | K_eff = 3 (parked) + 5 (new) = 8 | CONTINUE | Highest-EV: literature already grounded, parked prereg already drafted, mechanism = Carver Table 25, no NO-GO collision. Lift number is what the parked prereg measures — not pre-committed. 25% vol-target cap binds on Topstep prop. |
| **2** | **Account-pressure throttle (R8 engineering)** | Guardrail | 0 (engineering) | CONTINUE | Required infrastructure for overlay 1 to deploy safely. Carver Table 23 explicit prop-firm DD math. Zero K-cost; pure capital-protection. Implement alongside or BEFORE overlay 1. |
| **3** | **Drift-tier veto via Shiryaev–Roberts (R1 lane-health)** | Veto | ~10 | CONTINUE pending reopen audit | Plausibly differentiable from the April-2026 individual-lane-pause NO-GO because SR is a structurally different statistic (cumulative LR test, not trailing arithmetic). Adversarial audit required FIRST. |
| **4** | **Cross-instrument concordance gate (R7)** | Veto | ~12 | PARK | Real mechanism, but lower expected lift than 1–3 and inherits the parked `2026-04-11-mnq-cross-asset.yaml` K-budget. Re-evaluate after overlay 1 lands. |
| **5** | **Priority-ranking rotation (R8)** | Allocator | ~3 | PARK | Marginal under current slot headroom; resurrect only when accounts become slot-pressured. |

**KILLED (do not re-litigate):**
- Winner-speed / loser-speed Markov state — collides with break-speed NO-GO, no new mechanism.
- Generic regime-conditional or vol-spike throttle — collides with Vol-spike-REVERSED and Vol-regime-DEAD NO-GO rows.

---

## Highest-EV next prompt (to pre-register overlay 1)

```
TASK: pre-register the Carver Stage-2 sizing-modifier OPERATIONAL surface, atop the existing parked prereg.

MODE: pre-registration only. Do NOT run the backtest. Do NOT touch live allocator.

CONTEXT:
- The statistical-gate prereg already exists: docs/audit/hypotheses/2026-05-01-carver-stage2-vol-targeted-sizing.yaml (DESIGN_LOCKED_AWAITING_GO_AND_CHORDIA_PASS, K_budget=3, Pathway B).
- That prereg covers the statistical question: does Carver Table 25 sizing beat flat sizing on K=3 lanes whichever pass Chordia.
- This new prereg covers the OPERATIONAL question: how does the lane_allocator surface read FitnessScore.rolling_sharpe + IS Sharpe percentile + Carver Table 25 to emit a per-lane size_multiplier into lane_allocation.json, end-to-end, without changing entry logic.

DELIVERABLES (one file, draft only):
1. docs/audit/hypotheses/drafts/2026-05-22-carver-sizing-controller-operational.draft.yaml
   - testing_mode: family
   - Pathway: B (institutional, K=1 per lane)
   - K_budget: 5 (≤3 sized lanes + 2 robustness)
   - K_effective: 8 (= 3 parked Carver Stage-2 + 5 new) — declare explicitly per feedback_k_effective_prior_trial_accounting_minbtl.md
   - n_max_at_horizon ≤ 300 (Bailey 2013 MinBTL clean)
   - literature_ref: carver_2015_volatility_targeting_position_sizing.md, carver_2015_ch11_portfolios.md, carver_2015_ch12_speed_and_size.md
   - canonical_source_delegation: cost_specs, holdout_constant, account_profile, lane_allocator entry-points
   - Criterion 8: OOS ExpR ≥ 0.40 × IS ExpR, dir_match=TRUE with power≥0.80
   - kill criteria: ΔSharpe ≤ 0.05 (Harvey–Liu haircut applied), maxDD increases, ≥25% lane sign-flip (heterogeneity_ack required if hit)
   - per-cell breakdown mandatory (pooled-finding-rule)
   - **N<30 floor:** lanes with <30 trades in rolling_sharpe_60 window default to size_multiplier=1.0 (no modification)
   - **Topstep DD cap (BINDING):** size_multiplier clamped so realized monthly vol ≤ 25% × capital on $2.5K-trailing-DD profiles (Carver Table 23 + p.149 + parked prereg account_constants)
   - **Contract floor (BINDING):** size_multiplier × baseline contract count rounds to ≥1 (continue) or 0 (skip-and-log); never fractional
   - **scope_lock declaration:** lane_allocation.json is under multi-terminal-shared-file-hygiene.md three-check protocol; declare scope_lock on the file in the stage definition

BEFORE WRITING:
- Run scripts/tools/estimate_k_budget.py --hypothesis <draft-path> (REQUIRED before file lands)
- Run /nogo "carver sizing" and /nogo "vol target" — confirm no buried KILL verdict applies
- Compute K_effective = 8 against MinBTL ceiling of 300 (clean MNQ) and write the arithmetic into the yaml

DO NOT IMPLEMENT THE CONTROLLER. Pre-reg only. Code-path design lives in a follow-up plan.
```

---

## Verification

This is a design plan, not implementation. Verification = the next prompt produces a `.draft.yaml` that passes `scripts/tools/estimate_k_budget.py` and clears `/nogo` checks before any code is touched.

---

## FINAL DISPOSITION (2026-05-22 second interrupt — SHELVED)

> User pushback #2: *"do we park this whole idea and work on the live app trade shit"*
>
> **Answer: yes. Shelved.** This brainstorm is preserved as design exhaust, not action plan.

**Why shelved (verified live, not narrated):**

| Truth source | Number | What it means |
|---|---|---|
| `chordia_audit_log.yaml.audits` verdict histogram | 14 PASS_CHORDIA + 2 PASS_PROTOCOL_A + 6 PARK + 7 FAIL_BOTH + 7 FAIL_CHORDIA = 36 audited | 16 cleared-edge lanes exist in the audit log |
| `lane_allocation.json.lanes` (deployed) | 3 | Only 3 are deployed to `topstep_50k_mnq_auto` (max_slots=7, verified `prop_profiles.py`) |
| Gap | **13 cleared-edge lanes not deployed** | Bottleneck is allocator plumbing / correlation gate / manual rebalance cadence, NOT edge scarcity |

The state-controller overlays solve cross-sectional sizing + lane-health transitions. Both get *more* valuable as N_deployed grows. With the choke point at allocator throughput rather than edge supply, **building the controller now is option value, not realized value.** Live-app work compounds (more throughput → larger deployed book → controller becomes worthwhile); doing the controller first does not (no controller benefit until the book grows anyway).

**The ONE carve-out — Overlay 4 (account-pressure throttle).** Engineering not research; zero K-cost; required before N_deployed grows past 3 so a synchronized bad day does not breach Topstep $2.5K trailing-DD. Bring this forward as part of live-app hardening, NOT as a research overlay. ~half-day implementation.

**Resurrection trigger.** Re-open this brainstorm when *any* of:
- `len(lane_allocation.json.lanes) ≥ 5` (cross-section materializes)
- Live-app carry-overs are closed (`logs/live/*.log` FileHandler gap, dashboard `/api/bars-recent`, `.env` parse warnings, OS-level singleton on live runner — see [[feedback_narrow_exception_for_fail_open_observability]])
- A specific Chordia-cleared lane is blocked on lack of size differentiation (would need overlay 1)

Until then: **shelved without prejudice**. Do not re-litigate; this document is the lineage citation if the topic returns.

---

## Tunnel-vision / worthwhileness check (added 2026-05-22 post-interrupt-1)

> SUPERSEDED by FINAL DISPOSITION above. Kept for audit trail / lineage. The "PARK overlay 1 until N≥5" framing below was an over-correction — the true picture (verified after interrupt #2) is that N=3 is a transient choke, not steady state, AND that even with the time-phased reframe, the user's pivot to live-app work is the higher-EV move.



User raised exactly the right question at task-completion time: *is this work even worthwhile, or is it tunnel-vision?* Re-running the [[feedback_two_track_decision_rule]] decision discipline with live data, not narrative:

**Live allocator state, 2026-05-21 rebalance** (verified via `docs/runtime/lane_allocation.json`):

| Bucket | Count | What it means |
|---|---|---|
| `lanes` (DEPLOY) | **3** | Three MNQ lanes currently sized at 1.0 contract on `topstep_50k_mnq_auto` |
| `paused` | **832** | Paused — Chordia FAILED_RATIO, C8 OOS fail, SR alarm, or below threshold |
| `displaced` | **9** | Bumped by correlation gate; cleared on edge but redundant to a deployed lane |
| `stale` | **0** | None |
| `all_scores_count` | **844** | Total lane universe |

**What this means for each overlay's portfolio-EV claim:**

1. **Overlay 1 (Carver sizing) — EV materially smaller than the brainstorm suggests.** Carver Table 25 rebalances size *across multiple lanes by realized Sharpe percentile.* With only 3 deployed lanes, the redistribution surface is tiny — and "upper half vs lower half of IS Sharpe distribution" is barely a tertile split. The mechanism still applies (each lane's vol target × Topstep DD cap), but the *cross-sectional lift* the brainstorm motivated assumes a fuller book. **Honest reframe:** overlay 1's value at N=3 deployed is **vol-target-per-lane** (overlay 4's work, essentially) rather than **cross-sectional redistribution**. The "10–20% Sharpe lift" author-estimate is even less defensible at N=3.

2. **Overlay 2 (SR drift veto) — same problem, smaller surface.** SR alarms on 3 deployed lanes give 3 binary events to test. Fewer than 30 alarm samples is impossible; clustered-SE collapses. **Honest reframe:** SR veto is a population-level monitor (informational), not a per-lane veto, until the deployed book is materially larger.

3. **Overlay 4 (account-pressure throttle) — UNCHANGED.** This is engineering, not statistics. Carver Table 23 applies whether you have 3 or 30 lanes. Still the highest-confidence work in the shortlist.

4. **Overlays 3, 5, 6, 7 — already PARKed/KILLed. No change.**

**Higher-EV alternatives the brainstorm did NOT consider:**

- **Grow the deployed book from the 9 displaced lanes.** Per [[feedback_max_profit_grow_chordia_inventory_not_force_slots]] and [[feedback_high_r_inventory_comes_from_chordia_not_raw_expr]]: displaced ≠ rejected; they're cleared-edge candidates blocked only by correlation. Cycling one displaced lane into deployment (when the incumbent it correlates with rotates out) is a direct +1R/trade gain at zero K-cost. **This is the actual highest-EV move right now.**
- **Run a fresh Chordia batch on PD_*×E1** (per [[feedback_pd_filter_family_e2_deployment_unsafe]] — the family is E2-blocked but E1-eligible). Adding 1–2 cleared-edge lanes to a 3-lane book delivers ~33–67% portfolio expansion before any sizing sophistication is worth its K-cost.
- **Verify the Carver Stage-2 parked prereg even has 3 qualifying lanes** (the prereg's K_budget=3 is "whichever pass Chordia" — if fewer than 3 deployed lanes have Chordia PASS today, the prereg can't run anyway).

**Verdict on this brainstorm's overall worthwhileness:**

- **Overlay 4 is still worthwhile and unblocked.** Carver Table 23 DD throttle is required infrastructure before any sizing work and has zero K-cost.
- **Overlay 1 is worthwhile ONLY conditional on growing the deployed book first.** At N=3 lanes, cross-sectional Carver sizing is over-engineered. Park overlay 1's prereg until N_deployed ≥ 5 (a 2× growth target satisfied by promoting 2 of the 9 displaced lanes).
- **Overlay 2 stays PARKed pending reopen audit AND a larger deployed book.**
- **The highest-EV next move is NOT pre-registering the Carver operational prereg as the brainstorm proposed.** It is checking which of the 9 displaced lanes are deploy-candidates (Chordia cleared, correlation-rotation eligible) and running a `rebalance_lanes.py` cycle on those.

**Updated next-prompt recommendation (replaces the one at the end of § Highest-EV next prompt):**

```
TASK: Inspect docs/runtime/lane_allocation.json's `displaced` list (9 candidates as
of 2026-05-21 rebalance). For each: report strategy_key, Chordia verdict from
docs/runtime/chordia_audit_log.yaml, the lane it was displaced by, and the
correlation coefficient. Rank by deploy-readiness: cleared edge + low correlation
with current 3-deployed = top candidates for next rebalance cycle.

DO NOT promote anything. Read-only audit; recommend one PROMOTE candidate for
the next manual rebalance, with full lineage citation.
```

This is what "worthwhile" looks like under the two-track rule: grow the book before building the controller that sizes the book.

---

## Canonical-path corrections applied at write time (2026-05-22)

The plan as authored cited two file paths that don't exist at those locations in the current tree. Corrected inline above:

| Plan-as-authored | Actual canonical path (verified via Glob 2026-05-22) | Where corrected |
|---|---|---|
| `pipeline/holdout_policy.py` | `trading_app/holdout_policy.py` | § STATE FIRST → Holdout Mode A |
| `pipeline/sr_monitor.py` | `trading_app/sr_monitor.py` (+ `trading_app/live/sr_monitor.py`) | § Repo / DB truth gaps item 3; Overlay 2 mechanism line |

No semantic change — both files are canonical SR/holdout surfaces, just under `trading_app/` rather than `pipeline/`. Flagged here so the overlay-1 prereg cites the correct `canonical_source_delegation` paths.
