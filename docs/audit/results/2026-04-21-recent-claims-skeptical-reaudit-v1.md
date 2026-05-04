# Recent-claims skeptical re-audit — v1

**Scope:** all recent research PRs that made deployment-bearing or candidate-promotion claims, audited from canonical truth only (`orb_outcomes`, `daily_features`, `bars_1m`) against the Phase 0 criteria (`docs/institutional/pre_registered_criteria.md`).

**Author perspective:** the auditor ran PR #53, PR #55, PR #56 this session. Self-audit risk acknowledged — included in the table with skeptical reading of own work.

**Rubric:** VALID (passes all applicable gates + mechanism-sense) / ALIVE (unverified, not refuted, plausible) / DEAD (refuted by canonical test) / MISCLASSIFIED (label not supported by canonical test at the stated framing) / UNVERIFIED (test not yet run).

---

## Classification table

| PR | Claim | Test framing | Gate failures | Mechanism-sense check | **Verdict** |
|---|---|---|---|---|---|
| #48 | participation-shape monotonic-up UNIVERSAL on MNQ+MES+MGC at 5m RR=1.5 | Pathway B K=1 per instrument (pooled OLS β₁ with lane FE); IS-only | OOS not run. MES OOS pooled mean = -0.09R (negative). MNQ OOS pooled mean = +0.06R. MGC OOS pooled mean = +0.07R. Without OOS β₁, claim is IS-only. | Regression coefficient ≠ strategy. Deployable rule (e.g., "size top-quintile participation at 2× base") not derived. | **MISCLASSIFIED as deploy candidate — VALID as IS descriptive statistic only** |
| #49 | MNQ 30m RR=1.0 K=4 → 2 RESEARCH_SURVIVORs | Pathway A BH-FDR at K=4 | C8 fails (N_OOS < 50). Also now subsumed under PR #51 K=105 family → FAILS DSR at the larger family framing. | Standalone direction-taker frame only. | **MISCLASSIFIED — 2 cells sit inside PR #51's DSR-failing family; not CANDIDATE-grade at K=105** |
| #50 | MNQ 15m RR=1.0 K=3 → 2 CANDIDATE_READYs | Pathway A BH-FDR at K=3 | At K=3 they pass; at K=105 (PR #51) they FAIL DSR. | Standalone direction-taker frame only. | **MISCLASSIFIED at PR #51 family framing; amendment added.** |
| #51 | MNQ unfiltered cross-family K=105 → 5 CANDIDATE_READY | Pathway A BH-FDR at K=105 | FAILS C5 (DSR 0.0003–0.0081, all << 0.95). Effective-N correction (ρ̂=+0.058, N̂=99) leaves verdict unchanged — family is near-independent. | Standalone direction-taker frame only. | **MISCLASSIFIED → 5 cells are RESEARCH_SURVIVOR at best; amendment added (PR #56).** |
| #52 | 6-lane unfiltered baseline 2×2 decomposition | Aperture-specific IS-only | Superseded by PR #57 which corrected aperture binding for L2/L6. New 2×2: 2 FILTER_CORRELATES + 2 BOTH + 1 VESTIGIAL + 1 UNTESTABLE. | Filter-vestigialness is the right question for deployed lanes. | **PR #52 SUPERSEDED by #57. #57 verdict: VALID within its stated aperture-corrected scope.** |
| #53 | MES + MGC unfiltered cross-family K=176 → 0 survivors | Pathway A BH-FDR at K=176 | None — clean null. | Pooled long+short. Asymmetric-direction edge would be washed out. | **VALID as "MES/MGC unfiltered pooled-direction ORB has no PathwayA family survivor" — DOES NOT rule out asymmetric per-direction edge.** |
| #55 | MES + MGC single-filter overlay K=171 → 0 survivors | Pathway A BH-FDR, 5 canonical filters × 14 instrument-sessions × 3 RR at 5m | None — clean null. | Only 5m aperture tested. Pooled direction. Single-filter only (no composition). | **VALID as "MES/MGC 5m single-filter pooled ORB has no survivor" — DOES NOT rule out 15m/30m, multi-filter, per-direction, or conditioner/sizer/confluence roles.** |
| #56 | PR #51 5 cells FAIL Phase 0 C5 (DSR) | Bailey-LdP 2014 Eq. 2 + Exhibit 4 correction | Sanity-checked against paper worked example (0.1132 / 0.9004 reproduced exactly). Effective-N correction shown insufficient to rescue. | DSR is the canonical family-selection-bias correction. | **VALID self-audit — hardens PR #51 MISCLASSIFICATION.** |
| #57 | 6-lane baseline decomposition corrected at canonical aperture | Aperture-canonical IS rerun | Memory says drift 111/111 passed; L2 unfilt IS +0.050R t=+1.86, L6 2026 OOS +0.161R. No new Phase-0-complete claim. | Correct question for deployed-lane filter-vestigialness. | **VALID within its scope; produces CORRELATES / BOTH / VESTIGIAL / UNTESTABLE labels only — no deploy promotion.** |

---

## Canonical-truth verification points

- **PR #48 OOS (direct query against `orb_outcomes`):** MNQ OOS 5m RR=1.5 E2 CB1 pooled N=771 mean=+0.059R; MES OOS N=702 mean=−0.090R; MGC OOS N=601 mean=+0.070R. MES OOS pooled is NEGATIVE — incompatible with an "unfiltered universal edge" deploy framing on MES. (This does NOT refute the IS regression β₁ result; it refutes "deploy the unfiltered MES 5m RR=1.5 lane" as a standalone rule.)
- **PR #56 Exhibit 4 correction:** ρ̂ = +0.0578 across the PR #51 105-cell family; N̂ = 99.0; SR_0 barely moves. Confirms PR #51 misclassification.
- **Cost/risk math:** `pnl_r` is canonical net per `pipeline/cost_model.py`. Sanity-checked on one cell per PR #53/#55/#56.
- **Look-ahead discipline:** PR #55 excluded Asian-window sessions for OVNRNG_50 per canonical docstring; PR #53/#56 use only `trading_day` comparisons. No look-ahead in scope.
- **Honest K:** PR #51 K=105, PR #53 K=176, PR #55 K=171 (runnable), PR #56 uses PR #51's K=105 reproduced exactly.

---

## Where we're tunnel-visioned — alternative framings

| Role | What we tested | What we did NOT test |
|---|---|---|
| **Standalone direction-taker** | PR #49–#56 all in this frame | — |
| **Filter** (binary eligibility gate) | PR #55 on MES/MGC; PR #52/#57 on deployed MNQ lanes | Multi-filter composition; non-binary percentile buckets; filter × aperture interaction |
| **Conditioner** (modifies RR/hold time/direction of ANOTHER lane) | NONE — zero tests | MES-conditioner-on-MNQ; prev-day-direction conditioner; volatility-regime conditioner |
| **Allocator state** (portfolio-level gate based on cross-instrument signal) | NONE — zero tests | MES vol regime → MNQ size; MGC divergence → COMEX_SETTLE gate; breadth metric across instruments |
| **Confluence** (two signals must co-fire for size bump) | NONE — zero tests | MES+MNQ co-break; participation + size co-qualifier; overnight-took-PDH + break |
| **Sizer overlay** (continuous scaling, Carver Ch 10) | PR #48 documents the pattern; no deployable rule written | quintile-based size curve; rank-weighted position; Kelly-linked sizing on rank |
| **Per-direction edge** (long-only vs short-only) | All tests pool long+short | Per-direction MES/MGC screens; direction-conditional filters; short-only sell-climax rules |

## Fairly tested vs prematurely ruled out

**Fairly tested and DEAD:**
- MES/MGC unfiltered pooled-direction 5m/15m/30m × all RRs (PR #53) — K=176 clean null.
- MES/MGC 5m single-filter pooled-direction × 5 mechanisms × 3 RRs (PR #55) — K=171 clean null.
- MNQ K=105 unfiltered pooled-direction PathwayA (PR #51) — FAILS DSR under PR #56.

**Fairly tested and VALID:**
- PR #57 6-lane aperture-corrected filter-vestigialness decomposition (within its stated scope).

**Prematurely ruled out (not truly refuted by our tests):**
- MES/MGC as CONDITIONER for MNQ — never tested.
- MES/MGC as CONFLUENCE with MNQ — never tested.
- MES/MGC in long-only or short-only direction subspaces — never tested. Pooled means could hide asymmetric edge.
- MES/MGC 15m/30m × filter overlay — untested since PR #55 only did 5m.
- Multi-filter composition on any instrument — never tested.

**MISCLASSIFIED (labelled stronger than truth supports):**
- PR #48 "UNIVERSAL deploy candidate" — VALID as IS descriptive; OOS not run; no deployable rule derived; MES OOS pooled negative.
- PR #50/#51 CANDIDATE_READYs — FAIL DSR at family K=105.

---

## Best-opportunity / blocker / miss / next-best-test

### 1. Best opportunity (highest honest EV)

**PR #48 concrete-rule derivation + OOS validation.** The IS monotonic-up pattern is real and cross-instrument. Derive a DEPLOYABLE rule (e.g., "size top-quintile rank(rel_vol) at 2× base, bottom-quintile at 0") and run it on OOS. If OOS β₁ and rule-simulated P&L survive, this is a genuine sizer overlay — the first Carver Ch 10-style deploy candidate in the portfolio.

**ROI/EV rough estimate:**
- MNQ OOS N=771 × +0.06R ≈ +46R base. Top-quintile scaling (2×) could lift to +60–80R over the 4-month OOS.
- At $51.80 net/R per MNQ contract: +60R × $51.80 ≈ $3,100 per contract incremental annualised. Non-trivial.
- DOWNSIDE: if OOS β₁ is flat or reversed, overlay fails and we simply don't deploy it. No capital at risk.
- Probability of survival: MODERATE. MNQ/MGC OOS means are positive; MES is negative but only on pooled unfiltered — conditioned on rank-top the sign could still be positive.

### 2. Biggest blocker

**Missing conditioner / confluence / allocator infrastructure.** Every test I've run uses `filter_signal` returning {0,1}. Zero research code or canonical abstraction exists for:
- percentile-bucket features (rank-based conditioners),
- two-instrument co-occurrence signals (confluence),
- portfolio-state allocator gates.

Without these abstractions, the 4 prematurely-ruled-out framings stay untested. Building them is the highest leverage infrastructure work this session could do.

### 3. Biggest miss

**PR #48's "universal" framing was celebrated without OOS validation.** Memory lists it as "#48 monotonic-up universal on MNQ/MES/MGC" — but PR #48's own result doc line 66 says "Does NOT run OOS validation." The pattern is strong IS but treated as a finding instead of a platform. And MES OOS pooled mean is NEGATIVE — a red flag that should have been surfaced in the week since.

### 4. Next-best test (smallest, highest-signal)

**PR #48 OOS β₁ replication — K=3 Pathway B, 1 query per instrument.**
- For each of MNQ/MES/MGC 5m E2 CB1 RR=1.5: fit `pnl_r ~ rank(rel_vol)` with lane fixed-effects on OOS only.
- Report β₁_OOS, t_OOS, per-year stability.
- Compare to IS β₁ (MNQ +0.278, MES +0.330, MGC +0.300).
- Pass if OOS β₁ same-sign AND t_OOS ≥ 2.0.
- Expected time: < 1 hour. Budget: K=3. MinBTL trivial.

---

## ROI / EV summary per candidate path

| Path | Est. time | Est. probability of deploy-ready outcome | If deploys, annualised $ uplift (MNQ 1-contract baseline) | If fails | **Priority** |
|---|---:|---:|---:|---|---|
| PR #48 OOS β₁ replication + concrete sizer rule derivation | 3–6 hrs | 40–60% | $2,000–$4,000 / contract | No deploy; pattern downgraded to IS-descriptive | **1st (top)** |
| Build conditioner/confluence abstraction + rerun MES/MGC-as-conditioner-on-MNQ family (K ≤ 20 per family) | 10–20 hrs | 20–40% | $500–$2,500 / contract | Expanded null; closes three framings cleanly | 2nd |
| PR #51 Pathway-B K=1 rewrite on NYSE_OPEN 15m RR=1.0 (single theory-driven test, bypasses family multiplicity) | 2–4 hrs | 30–50% | $1,000–$2,500 / contract | Clean single-cell null; strengthens DSR finding | 3rd |
| Filter-composition family on MNQ 6 deployed lanes (explicit multi-filter combo pre-reg) | 4–8 hrs | 10–25% | $0–$2,000 / contract | Multi-filter also dead, sizer path dominant | 4th |
| Long-only / short-only per-direction scan on PR #53/#55 DEAD universe | 6–10 hrs | 10–20% | $0–$1,500 / contract | Asymmetric null, universes truly dead | 5th |

---

## Recommendations

1. **Stop standalone-filter discovery.** PR #53 + PR #55 + PR #56 closed three thick doors cleanly. More of the same is tunnel.
2. **Run Path 1 (PR #48 OOS β₁) as the next autonomous test.** Smallest, highest signal-to-ROI ratio. Either rescues the deploy-candidate story or moves PR #48 to IS-descriptive.
3. **Do not deploy any PR #49/#50/#51 cell.** All MISCLASSIFIED under the Phase 0 + DSR framework. Amendments landed under PR #56.
4. **Long-term: build Path 2 abstractions.** Percentile-bucket signal, two-instrument co-occurrence, portfolio-state allocator. Without these, we have a very sharp hammer and only nail-shaped problems.
5. **Record this re-audit in HANDOFF + memory so next session doesn't retrace.**

---

## Not done by this re-audit

- Does NOT writeto `validated_setups` / `edge_families` / `lane_allocation` / `live_config`.
- Does NOT run the PR #48 OOS β₁ test (recommended as next step).
- Does NOT build conditioner/confluence abstractions (Path 2).
- Does NOT rewrite PR #51 as Pathway B K=1 (Path 3).

Pure documentation audit against canonical-truth cross-checks.
