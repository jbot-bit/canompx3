---
pooled_finding: true
per_cell_breakdown_path: docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md
flip_rate_pct: 67.0
heterogeneity_ack: true
---

# Cross-walk — Sizing-Substrate Stage-1 (2026-04-27) vs PR #51 Participation-Shape (2026-04-20)

**Type:** definitional + theory cross-reference. **No new statistic.** Read-only doctrine note.

## Scope

Future sessions seeing the SUBSTRATE_WEAK Stage-1 PASS cells (`MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5` and `MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12`, both via `rel_vol_session`) may be tempted to author a fresh Path-β pre-reg testing "rel_vol_session as session-level conditioner." This note prevents that duplicate work by establishing two facts:

1. The Stage-1 Tier-B feature `rel_vol_session` is the **same column** as the PR #51 regression variable.
2. PR #51 already settled the universal-scope question with grounded mechanism (Fitschen Ch 3 + Chan Ch 7 — both extracted to `docs/institutional/literature/`).

**Inherited methodology caveat:** the institutional code+quant audit on `research/audit_sizing_substrate_diagnostic.py` returned `PASS_WITH_RISKS` (verdict: SUBSTRATE_WEAK upheld; conditional on documented MED findings being addressed — see § Caveats below). All conclusions in this note are conditional on that verdict standing.

## Definitional equivalence (verified from source)

| Source | File | Line | Column derivation |
|---|---|---|---|
| Stage-1 pre-reg | `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml` | 307–308 | `feature_id: "rel_vol_session"`, `column_template: "rel_vol_{ORB_LABEL}"` |
| Stage-1 runner | `research/audit_sizing_substrate_diagnostic.py` | 277 | `col = feat.get("column") or feat["column_template"].format(ORB_LABEL=lane["orb_label"])` |
| PR #51 script | `research/participation_shape_cross_instrument_v1.py` | 67 | `rel_col = f"rel_vol_{session}"` |
| Canonical source | `daily_features.rel_vol_{SESSION}` per session | — | Trade-time-knowable per `.claude/rules/backtesting-methodology.md` RULE 6.1 |

**Equivalence is exact.** Both studies regress trade-level `pnl_r` on the same `daily_features.rel_vol_{ORB_LABEL}` column. They differ only in:

- Statistical method: PR #51 = OLS rank with lane-FE, HC3 SE, Pathway-B K=1 per instrument. Stage-1 = Spearman ρ + Q1→Q5 monotonic + bootstrap delta + ex-ante sign + split-half + Carver fn78 stability gate, K=48.
- Aggregation level: PR #51 pools sessions within instrument (universal scope). Stage-1 evaluates per (lane × feature) cell (lane scope).
- Filter universe: PR #51 unfiltered (raw lane). Stage-1 deployed-binary-filtered (RULE 2 Pass-2 overlay framing).

## Theory grounding (verified from extracts)

PR #51 cites in script header:

> "Theory: Fitschen 2013 Ch 3 + Chan 2013 Ch 7 (both verified via local extracts)."

Both extracts are present in `docs/institutional/literature/`:

- `fitschen_2013_path_of_least_resistance.md` — grounds intraday trend-follow on commodities + stock indices (CORE ORB premise per `.claude/rules/institutional-rigor.md` §7).
- `chan_2013_ch7_intraday_momentum.md` — grounds intraday momentum on participation/volume.

The Stage-1 pre-reg's `ex_ante_direction: "+"` for `rel_vol_session` ("Higher participation predicted to give cleaner break per-trade", line 309) is consistent with this same Fitschen/Chan grounding. **No new mechanism citation is required**, and an attention-theory citation (e.g., Hong-Stein) would be invented — the actual mechanism is intraday momentum on participation, already extracted.

## Heterogeneity verdict

Stage-1 lane-level coverage on `rel_vol_session`:

| Lane | rel_vol_session result | Sign vs PR #51 pooled (positive) |
|---|---|---|
| MNQ EUROPE_FLOW (ORB_G5 universe) | PASS | + (agrees) |
| MNQ TOKYO_OPEN (COST_LT12 universe) | PASS | + (agrees) |
| MNQ SINGAPORE_OPEN (ATR_P50 universe) | FAIL | + sign, fails magnitude/monotonic gate |
| MNQ COMEX_SETTLE (ORB_G5 universe) | FAIL | + sign, fails monotonic gate |
| MNQ NYSE_OPEN (COST_LT12 universe) | FAIL | sign +0.045 trivial, fails BH-FDR + magnitude |
| MNQ US_DATA_1000 (ORB_G5 universe) | FAIL | + sign, fails magnitude gate |

**Lane-level pass rate: 2/6 = 33%. Lane-level failure rate: 4/6 = 67%.** This is the `flip_rate_pct` recorded in front-matter — `heterogeneity_ack: true` because flip rate ≥ 25% per `.claude/rules/pooled-finding-rule.md`.

The ρ sign is positive in 5/6 lanes; only the strict cell-gate battery (monotonic Q1→Q5 + |Q5−Q1|≥0.20R + bootstrap CI > 0 + Carver fn78 stability) eliminates the lane-level effect on 4 lanes that show the universal sign. This is **consistent with PR #51's universal effect existing but being attenuated below the strict cell-level threshold at lane scope**, not evidence against PR #51.

The "67% lane-level flip" framing should NOT be quoted in memory or doctrine as if `rel_vol_session` works on only 2/6 lanes universally — that misrepresents the relationship to PR #51's pooled finding. Trading or research decisions on `rel_vol_session` MUST consult PR #51's universal-scope evidence first; Stage-1's per-lane breakdown is a post-hoc cross-check, not the primary evidence.

## Verdict

**Cross-walk conclusion (subject to audit caveat above):**

- Stage-1 PASS cells are **lane-level cuts of PR #51's universal monotonic-up base**, attenuated by stricter cell gates and the Carver fn78 stability gate.
- The 4 Stage-1 FAILS that show the same positive sign reinforce — not refute — PR #51.
- The two PASS cells are sizer-role-PASS but stage2_eligible=False due to Carver fn78 UNSTABLE — therefore not deployable as Carver-style continuous scalers without a fresh Stage-2 pre-reg addressing forecast normalization for unstable signals.
- Follow-on work routes to the **existing PR #51 candidate-activation plan** (in flight per `memory/amendment_3_2_and_cpcv_parked_apr21.md`), not a new Path-β pre-reg.

## Recommendations

**Do:**
- Execute the existing PR #51 candidate-activation plan.
- Treat Stage-1 PASS cells as a corroborating cross-check on PR #51, not as new evidence.
- If a future session investigates lane-conditional vs universal participation effects, scope the test as a refinement of PR #51 (not new discovery), with K-budget accounting for cumulative trial count per Bailey MinBTL.

**Do not:**
- Author a new Path-β pre-reg testing `rel_vol_session` as a session-level conditioner — duplicates PR #51 at lane scope, breaches cumulative-trial Bailey MinBTL bound.
- Cite an attention-theory mechanism (e.g., Hong-Stein) — the canonical mechanism is Fitschen Ch 3 + Chan Ch 7 intraday momentum, already in `docs/institutional/literature/`.
- Re-promote UNSTABLE Stage-1 PASS cells to Stage-2 sizer eligibility — Carver Ch.7 fn78 explicitly forbids sizing on unstable σ.

## Caveats / Limitations / Disconfirming Considerations

- **Audit-pending status (now resolved as PASS_WITH_RISKS):** the cross-walk's framing depends on the SUBSTRATE_WEAK Stage-1 verdict standing. The audit upheld the verdict but flagged 5 load-bearing findings:
  - **A. ATR_P50 vol_norm = raw identity (MED).** Effective unique cells = 42, not 48 (6 ATR_P50 raw/vol_norm cells are identical column derivations). Conservative direction; does not flip any verdict.
  - **B. COST_LT12 inline formula not delegated to canonical (LOW).** Drift risk only; runtime equivalent to `CostRatioFilter`.
  - **C. Pooled-finding YAML front-matter absent on Stage-1 result MD (MED — rule violation).** Fixed in companion commit.
  - **D. RULE 13 pressure-test absent from test suite (MED — mandatory rule).** Fixed in companion commit.
  - **E. Monotonicity gate dominates failures — verified correct (INFO).** Direction-agnostic gate; failing cells are genuinely non-monotonic.
- **Single-pass discipline.** This cross-walk is read-only and writes nothing to `validated_setups`, `experimental_strategies`, `live_config`, or `lane_allocation`. It is descriptive doctrine that re-routes future work, not a new statistical test.
- **Inherited methodology caveat.** The mechanism cite (Fitschen Ch 3 + Chan Ch 7) is verified in extracts. The bridge inference ("Stage-1 lane PASS cells re-discover PR #51 universal base") is INFERRED — it is the most parsimonious explanation given column equivalence + theory equivalence + sign agreement on 5/6 lanes, but is not itself a new statistical test.

## Reproduction / Outputs

- Stage-1 result: `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`
- PR #51 result: `docs/audit/results/2026-04-20-participation-shape-cross-instrument-v1.md`
- PR #51 pre-reg: `docs/audit/hypotheses/2026-04-20-participation-shape-cross-instrument-v1.yaml`
- Canonical mechanism extracts: `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md`, `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`
- Backtesting RULES applied: RULE 6.1 (trade-time knowability of `rel_vol_{session}`), Bailey MinBTL cumulative-trial accounting per `.claude/rules/backtesting-methodology.md` §4.2.
- Pooled-finding rule applied: `.claude/rules/pooled-finding-rule.md` (front-matter `flip_rate_pct: 67.0`, `heterogeneity_ack: true`).

## Not done by this cross-walk

- Does NOT modify any pre-reg, deployment config, or production code.
- Does NOT supersede the PR #51 verdicts.
- Does NOT authorize Stage-2 implementation.
- Does NOT compute new statistics; equivalence and sign-agreement claims are read-only inspections of source files and the existing Stage-1 result table.
