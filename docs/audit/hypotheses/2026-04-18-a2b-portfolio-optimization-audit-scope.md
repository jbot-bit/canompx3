# A2b Stage-1 — Portfolio-Optimization Audit Scope

**Date:** 2026-04-18
**Status:** SCOPE_LOCKED (this doc is the Stage-1 deliverable)
**Authority:** adversarial re-open at `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md`
**Design iterations before lock:** 11
**Literature-coverage verified against:** `docs/institutional/literature/` extracts

**This doc is NOT a backtest, NOT a pre-reg for a discovery, NOT a recommendation to change the allocator.** It is a literature-grounded RANKING of which allocator-design axes have the highest expected EV delta if changed, and it proposes the top-3 to pursue as follow-up sub-audits A2b-1/2/3 — each of which will itself get its own plan-iterate-design cycle.

---

## 1. Scope

Portfolio-level allocator-design axes only. Per-lane discovery concerns (RR selection, stop-loss placement, cost modelling) are out of scope.

**Question answered by this doc:** *which allocator-design axes are worth auditing next, ranked by literature-grounded expected EV × tractability × bug-fix priority?*

## 2. Pre-committed top-3 selection criteria (locked before ranking)

A candidate axis enters the top-3 only if ALL four are true:

1. **Literature-grounded** — exact extract cited in `docs/institutional/literature/`, verified content (not just title-match)
2. **Expected EV delta MED or HIGH** — qualitative, reasoned against literature, not just intuition
3. **Implementation cost LOW or MED** — not a multi-month rebuild; must fit a single sub-audit cycle
4. **Low-to-medium blast radius** — configurable without touching discovery layer (no per-lane re-validation required)

Axes that fail any one criterion are excluded from top-3. Literature gaps and extract-uncertain candidates go to the follow-up queue, not the top-3.

## 3. Axis enumeration (16 axes, literature-coverage verified)

Categories: **[LIT ✓]** = extract verified in Iter 10/11; **[LIT ✗]** = no extract in our library; **[LIT ?]** = claimed but verification failed or incomplete.

| # | Axis | Current allocator impl | Literature alternative | Lit status | EV δ | Impl cost | Blast | Passes top-3? |
|---|---|---|---|---|:---:|:---:|:---:|:---:|
| 1 | **Ranking: DSR** | `_effective_annual_r` = annual_R × 3mo-decay | DSR formula per Bailey-LdP 2014 Eq 2 with γ₃/γ₄/T/V[SR]/N corrections | **[LIT ✓]** Bailey-LdP 2014 | HIGH | MED | LOW | **YES** |
| 1b | **Ranking: HL haircut** | same | Haircut SR per Harvey-Liu 2015 Eq 5 (analytical, simpler than DSR) | **[LIT ✓]** Harvey-Liu 2015 | HIGH | LOW | LOW | **YES** (merged into axis 1 for sub-audit scope) |
| 2 | Correlation threshold | rho<0.70 (Pearson) | Markowitz 1952 rho<~0.3 for material risk reduction | **[LIT ✗]** | MED | LOW | LOW | NO — lit gap |
| 3 | Covariance estimation | Pairwise Pearson, overlap days | Ledoit-Wolf 2004 constant-correlation shrinkage | **[LIT ✗]** | MED | HIGH | MED | NO — lit gap + cost |
| 4 | Selection algorithm | Greedy rho-gated | LdP 2020 HRP (Hierarchical Risk Parity) | **[LIT ?]** extract is Ch 1 intro only; HRP in Ch 4-7 not extracted | HIGH | HIGH | HIGH | NO — lit extract insufficient |
| 5 | Window choice | Trailing 12mo | Carver 2015 Ch 9 (longer for Sharpe); EWMA variants | **[LIT ✓]** Carver (partial — vol targeting framework) | MED | LOW | LOW | candidate — lower EV than 1/6/8 |
| 6 | **Sizing: Half-Kelly** | Flat 1 contract × copies | Carver 2015 Ch 9 Table 25: SR→vol-target→Kelly | **[LIT ✓]** Carver Ch 9 explicit table | HIGH | MED | MED | **YES** |
| 7 | Hysteresis | 1.2× replacement threshold | Leland 1999 turnover / transaction cost optimization | **[LIT ✗]** | LOW-MED | LOW | LOW | NO — lit gap + low EV |
| 8 | **Regime gating: filtered-per-lane** | Unfiltered 6mo session ExpR <= 0 → PAUSE (VERIFIED BUG from adversarial audit — filtered lanes on "cold" sessions mis-gated) | Pepelyshev-Polunchenko 2015 SR monitor per-lane (already used for C12) + Chan 2008 Ch 7 regime-switching taxonomy (GARCH, HMM) | **[LIT ✓✓]** Pepelyshev-Polunchenko + Chan Ch 7 | HIGH (bug-fix) | MED | MED | **YES — bug-fix priority** |
| 9 | Max slots | 7 (from scarcity audit) | Meucci 2010 effective # bets | **[LIT ✗]** | LOW | LOW | LOW | NO — lit gap |
| 10 | DD estimation | P90 ORB × stop_mult × point_val | Artzner-Delbaen-Eber-Heath 1999 coherent risk / CVaR | **[LIT ✗]** | MED | MED | MED | NO — lit gap |
| 11 | Rebalance frequency | Monthly | Leland 1999 (turnover cost); regime-event-driven alts | **[LIT ✗]** | LOW | LOW | LOW | NO — lit gap |
| 12 | MinBTL max-strategies cap | None explicit (uses max_slots=7) | Bailey 2013 Theorem 1: MinBTL ≈ 2·ln(N)/E[max_N]²; 5yr data → max 45 independent trials | **[LIT ✓]** Bailey 2013 | LOW-MED | LOW | LOW | candidate — protective rather than additive EV |
| 13 | FST portfolio benchmark | None | LdP-Bailey 2018 Theorem 1: E[max SR under null] ≈ (1-γ)Z⁻¹[1-1/K] + γZ⁻¹[1-1/(Ke)]; K=100 → SR≈2.3 | **[LIT ✓]** LdP-Bailey 2018 | MED (diagnostic) | LOW | LOW | candidate — diagnostic, not EV-generating |
| 14 | CPCV validation | None; 12mo trailing only | LdP 2020 CPCV framework | **[LIT ?]** extract is Ch 1 only | HIGH | HIGH | HIGH | NO — lit extract insufficient |
| 15 | Forecast combination sizing | None (binary DEPLOY/PAUSE) | Carver 2015 Ch 10 forecast combiner | **[LIT ?]** Carver extract focuses Ch 9 vol targeting; Ch 10 combiner content uncertain in extract | HIGH (if confirmed) | HIGH | HIGH | NO — extract-uncertain + cost |
| 16 | Sharpe haircut on ranking | None | Harvey-Liu 2015 Eq 5: haircut Sharpe under multi-testing (subset of axis 1b) | **[LIT ✓]** Harvey-Liu 2015 | HIGH (overlaps 1) | LOW | LOW | MERGED into axis 1 sub-audit |

## 4. Literature coverage gap — 6 axes blocked

Six axes cannot be pursued as top-3 without expanding `docs/institutional/literature/`:

- **Markowitz 1952** (correlation threshold grounding)
- **Ledoit-Wolf 2004** (shrinkage covariance)
- **Meucci 2010** (effective number of bets)
- **Artzner-Delbaen-Eber-Heath 1999** (coherent risk measures)
- **Leland 1999** (turnover / rebalance cost)
- **LdP 2020 Chapters 4-7** (HRP, CPCV — the book is in `resources/` but only Ch 1 extracted)

Action: **DEFER these to a separate "literature expansion" workstream.** Not this audit's scope.

## 5. Top-3 locked — bug-fix-prioritized ordering

### A2b-1 (first): **Regime gating — filtered per-lane, Pepelyshev-Polunchenko + Chan regime taxonomy**
- **Why first:** fixes a VERIFIED bug surfaced in the 2026-04-18 adversarial audit. The current `_compute_session_regime` in `lane_allocator.py:371-398` uses UNFILTERED session-level pnl_r, mis-classifying lanes whose filter produces positive edge on "cold" unfiltered sessions. Fixing a known defect BEFORE pursuing theory-grounded EV optimizations is standard institutional rigor discipline.
- **Literature:** Pepelyshev-Polunchenko 2015 (SR monitor formulae Eq 11, 13; already wired for C12 reviews); Chan 2008 Ch 7 (regime taxonomy: inflation/recession, high/low vol, mean-rev/trending; GARCH + HMM frameworks).
- **Proposed sub-audit design** (locked pre-implementation in A2b-1 Stage-1):
  - Patch `_compute_session_regime` to accept optional `filter_key`; when provided, apply `filter.matches_df()` before averaging pnl_r
  - Regression test: pre-patch vs post-patch allocator output on 2026-04-18 rebalance — enumerate which (if any) PAUSED/excluded lanes change status
  - Drift check addition: orphan-column-style check that regime gate consistent across filtered variants of same session
- **Expected EV δ:** +5 to +15 R/yr immediate (if CME_PRECLOSE MNQ lane unlocks) + systemic future-proofing
- **Implementation cost:** MED (1-2 days: patch + tests + regression + drift check)
- **Blast radius:** MED (touches canonical allocator logic)

### A2b-2 (second): **DSR ranking — replace `_effective_annual_r` with Deflated Sharpe**
- **Why second:** highest-EV theory-grounded improvement with literature-verified formula. Bailey-LdP 2014 Eq 2 directly implementable. Current ranking has zero multi-testing correction.
- **Literature:** Bailey-LdP 2014 (Deflated Sharpe formula with γ₃/γ₄/T/V[SR]/N corrections) + Harvey-Liu 2015 (simpler haircut as alternative, for simpler implementation)
- **Proposed sub-audit design** (locked pre-implementation in A2b-2 Stage-1):
  - Compute DSR (or haircut Sharpe) for all 38 validated lanes
  - Re-rank lanes; identify changes in top-7 selection
  - Forward-backtest on IS (pre-2026-01-01) comparing `_effective_annual_r` ranking vs DSR ranking — which portfolio has higher IS Sharpe?
  - Mode A holdout protected: no OOS consumed; re-ranking uses already-scored lanes
- **Expected EV δ:** HIGH — rankings can shift materially; ~10-30 R/yr portfolio uplift if DSR selects a better 7
- **Implementation cost:** MED (compute DSR per-lane; γ₃/γ₄/V[SR] from trailing returns; not trivial but bounded)
- **Blast radius:** LOW (replaces one ranking function; unit-testable)

### A2b-3 (third): **Half-Kelly sizing — replace flat-1-contract with Carver vol-target-scaled**
- **Why third:** highest aggregate dollar-EV unlock but riskier (affects every trade's dollar footprint + DD profile). Depends on per-lane SR estimate which is itself an input to ranking — interaction with A2b-2.
- **Literature:** Carver 2015 Ch 9 Table 25 — explicit lookup: realistic SR → vol target, with Half-Kelly haircut built in. Numerics given.
- **Proposed sub-audit design** (locked pre-implementation in A2b-3 Stage-1):
  - Estimate realistic SR per live lane (C8 haircut + DSR if A2b-2 ships first)
  - Map to vol target via Carver Table 25
  - Convert vol target → contract count per lane (given instrument point-value, daily σ)
  - Simulate forward on IS data: dollar-R curve vs flat-1-contract baseline
  - Check DD stays under max_dd budget
- **Expected EV δ:** HIGH (10-15% dollar-R uplift at same DD per Carver's general claims)
- **Implementation cost:** MED (wire into `CopyOrderRouter` sizing; new test matrix)
- **Blast radius:** MED (execution-layer sizing change; needs Half-Kelly floor to prevent blow-up)

## 6. Bias checks applied during ranking

| Bias | Check applied | Result |
|---|---|---|
| Complexity bias (favour HRP/DCC-GARCH because sophisticated) | Deliberately down-weighted axes 3, 4, 14, 15 when extract coverage was insufficient | HRP and CPCV dropped from top-3 |
| Sophistication bias (dismiss simple alternatives) | Axis 5 (window choice) evaluated fairly as "simpler" vs complex | Retained as candidate; not top-3 due to lower EV |
| Anchoring to current allocator | Included "fundamentally different paradigm" (HRP) as candidate 4 | Dropped only due to extract gap, not bias |
| Recency bias from adversarial audit | A2b axes enumerated BEFORE consulting audit findings; bug-fix priority applied at final ordering, not ranking | Axis 8 bug-fix is a priority heuristic not a data-fit |
| Tunnel vision on A2a items | Deliberately re-surfaced axes A2a did NOT raise (DSR, Half-Kelly, FST benchmark, MinBTL) | 4 axes not in A2a |
| Literature-memory bias | Every "LIT ✓" claim verified against actual extract content, not title-grep | 3 claims downgraded: LdP 2020 HRP, LdP 2020 CPCV, Carver forecast combiner |

## 7. Meta-caveats (apply to all sub-audits)

- **Interaction effects:** changing ranking (A2b-2) AND sizing (A2b-3) at once is not additive — they interact through SR estimates. Sub-audits should be ordered so A2b-2 outputs feed A2b-3 inputs.
- **Regime-conditional optimality:** what's optimal in bull ≠ bear. Each sub-audit's backtest should break results by regime (bull/bear, high/low vol).
- **Implementation lag:** even if a rule is better per backtest, 3-6 months of live forward required before the change shows in live P&L.
- **Prop-firm constraints:** topstep $2500 DD cap invalidates some literature-canonical benchmarks (e.g., Markowitz MVO has no DD constraint). All backtests must respect DD + drawdown-rule compliance.
- **Mode A discipline:** every sub-audit must be IS-only for decision criteria. 2026 forward data is MONITOR-ONLY, not selection input.

## 8. Next actions — A2b-1, A2b-2, A2b-3 each get full plan-iterate-design cycle

When ready to proceed:
1. **A2b-1 (regime-gate filtered patch):** open a new Stage-1 plan-iterate-design cycle in a fresh audit thread. Deliverables: pre-reg → patch → tests → regression → drift check → commit.
2. **A2b-2 (DSR ranking):** open Stage-1 after A2b-1 is shipped. DSR implementation and forward-backtest on IS.
3. **A2b-3 (Half-Kelly sizing):** open Stage-1 after A2b-2 SR estimates are validated. Sizing wiring into execution layer.

Each sub-audit's Stage-1 plan must:
- Cite the exact literature extract file(s)
- Propose a specific backtest/simulation design with numeric pass/kill criteria
- Pre-commit success and kill criteria in writing before running
- Protect Mode A holdout (2026-01-01 sacred)
- Define rollback plan if verified-bad result

## 9. A2a status note

Parallel audit A2a (allocator rho audit of 15 excluded lanes) is **paused at Stage-2 design locked** per user decision. Stage-2 design spec is in the conversation record and will be committed to a separate scope doc if/when A2a is resumed. A2b-1 through A2b-3 do NOT depend on A2a completing first — they can proceed in parallel.

## 10. Deferred / out-of-scope for A2b

- Literature expansion to cover Markowitz / Ledoit-Wolf / Meucci / Artzner / Leland / LdP Ch 4-7 → separate workstream
- Non-allocator topics (per-lane discovery, execution quality, order routing) → separate audits
- MGC / MES cross-instrument discovery → separate cycle (F16 on prior surface map)
- Copies 2→5 scaling gate operationalization → user-decision, parallel to all A2b work

## 11. Kill criteria for this scope doc

If any of these conditions are discovered after commit, revise the top-3:
- Any top-3 axis has literature extract incorrectly verified → re-verify + potentially reorder
- Any literature-gap axis acquires an extract → re-evaluate top-3 ranking
- Any top-3 axis's blast radius proves HIGH during sub-audit Stage-1 planning → drop and replace

**This scope doc does not itself consume OOS.** It is a literature-ranking exercise, read-only against already-extracted literature.
