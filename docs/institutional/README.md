# Institutional Methodology — Literature-Grounded Framework

**Established:** 2026-04-07
**Purpose:** Ground all research, backtesting, and deployment decisions in specific, citable passages from institutional-grade literature in `resources/`. Replaces informal "training memory" citations.

**Trigger for creation:** April 2026 audit revealed that prior discovery methodology tested ~35,000 combinations on ~6 years of clean MNQ data, violating Bailey-López de Prado MinBTL bounds by ~600x. Root cause: brute-force combinatorial discovery without pre-registered hypothesis budget. This directory locks in corrected institutional standards.

---

## Directory layout

```
docs/institutional/
├── README.md                                   ← you are here
├── HANDOFF.md                                  ← session handoff (SUPERSEDED 2026-04-07)
├── literature/                                 ← verbatim passages from resources/*.pdf
│   ├── bailey_et_al_2013_pseudo_mathematics.md      ← MinBTL theorem (CRITICAL)
│   ├── bailey_lopez_de_prado_2014_deflated_sharpe.md ← DSR formula (CRITICAL)
│   ├── lopez_de_prado_bailey_2018_false_strategy.md  ← K>>1 false positive
│   ├── harvey_liu_2015_backtesting.md                ← BHY haircut / profitability hurdles
│   ├── chordia_et_al_2018_two_million_strategies.md  ← t=3.79 threshold
│   ├── pepelyshev_polunchenko_2015_cusum_sr.md       ← live monitoring
│   └── lopez_de_prado_2020_ml_for_asset_managers.md  ← theory-first, CPCV
├── finite_data_framework.md                    ← synthesized approach (v1 + v2 amendment note)
├── pre_registered_criteria.md                  ← LOCKED thresholds, v2 with 5 Codex amendments
└── hypothesis_registry_template.md             ← template for pre-registered discovery

Sibling directories (referenced from here but not part of this tree):
├── ../audit/hypotheses/                        ← pre-registered hypothesis files (infra, 0 files)
├── ../audits/2026-04-07-finite-data-orb-audit.md ← Codex audit that produced v2 amendments
└── ../plans/2026-04-07-canonical-data-redownload.md ← Phase 2-5 plan (awaiting e2-fix merge)
```

## v2.7 status (as of 2026-04-08 — read before applying any threshold)

`pre_registered_criteria.md` has been amended from v1 to v2.7. The Codex audit
(`../audits/2026-04-07-finite-data-orb-audit.md`) produced amendments 2.1-2.5 on
2026-04-07; Amendment 2.6 was a brief Mode B declaration that was rescinded by
Amendment 2.7 on 2026-04-08 per explicit user correction.

- **Amendment 2.1:** DSR > 0.95 downgraded from binding to cross-check until `N_eff` is formally solved in-repo (validator already treats DSR as informational).
- **Amendment 2.2:** Chordia t ≥ 3.79 reframed as severity benchmark with banded thresholds, not a universal hard bar.
- **Amendment 2.3:** Criterion 8 (2026 OOS) contingent on pre-run holdout policy declaration. Resolved by Amendment 2.7 below.
- **Amendment 2.4:** Current 5 deployed lanes reclassified as *research-provisional + operationally deployable*, not *production-grade institutional proof*.
- **Amendment 2.5:** Execution overlays (calendar skip, ATR velocity, E2 timeout) must be reported separately from discovery-filter evidence.
- ~~**Amendment 2.6 (RESCINDED 2026-04-08):**~~ Mode B (post-holdout-monitoring) declaration. Rescinded by Amendment 2.7 because the Apr 2 16yr rebuild plan explicitly prescribed `--holdout-date 2026-01-01` and the 117-strategy baseline was legitimately Mode A. The Apr 3 re-run that consumed the holdout for +7 strategies was a premature autonomous error.
- **Amendment 2.7 (BINDING):** **Mode A (holdout-clean) operative.** Sacred holdout window is 2026-01-01 onwards, currently ~3.2 months deep, growing daily. Every discovery run MUST use `--holdout-date 2026-01-01` (or earlier). Existing 124 validated_setups are grandfathered as research-provisional per Amendment 2.4. See `../plans/2026-04-07-holdout-policy-decision.md` for the audit trail.

`finite_data_framework.md` has a top-of-file warning block pointing readers to the criteria file for current binding state — where the framework and criteria disagree, the criteria file wins. The framework's § 4.4 OOS protocol is now fully active under Mode A.

**Enforcement** (added 2026-04-08 in 6-stage refactor, commits `81a7079`-`0edf8e8`): the canonical Mode A constants live in `trading_app/holdout_policy.py`. Three downstream consumers import from it: `pipeline.check_drift.check_holdout_contamination()` (contamination detector + grandfather logic), `trading_app.strategy_discovery.main()` (CLI rejects post-sacred values), `trading_app.strategy_validator._check_mode_a_holdout_integrity()` (pre-promotion gate). A new drift check `check_holdout_policy_declaration_consistency` (#83) catches future doc-code drift before commit.

## How to use this directory

1. **Before writing any research code:** Read `finite_data_framework.md` for the overall approach.
2. **Before running discovery:** Read `pre_registered_criteria.md` for locked thresholds. Never relax them after seeing results.
3. **Before proposing a new strategy hypothesis:** Use `hypothesis_registry_template.md` to write it up BEFORE testing.
4. **When citing a statistical method:** Link to the specific literature file in `literature/` — do not cite from memory.
5. **When updating:** Preserve verbatim quotes. Only add interpretation/application sections.

## Summary of core findings from Phase 0 extraction (2026-04-07)

| Source | Key result | Our status |
|---|---|---|
| Bailey et al 2013 (MinBTL) | 5 years data → max 45 independent trials | **FAIL — we did ~35,000** |
| Bailey-LdP 2014 (DSR) | N=100 trials → need annualized SR > 1.5 | **FAIL — deployed avg SR ~0.79** |
| Chordia et al 2018 | Finance MHT threshold: t ≥ 3.79 | **3/4 deployed MNQ lanes FAIL** (only COMEX passes) |
| Harvey-Liu 2015 | BHY profitability hurdle: ~7.4% annual at 20yr/300 tests | Marginal |
| LdP-Bailey 2018 (False Strategy) | K=1000 trials under zero edge → E[max SR] = 3.26 | Our max SR << 3.26 |
| LdP 2020 (ML for AM) | "Backtests are not a research tool. Theories are." | **We were backtest-driven, not theory-driven** |
| Pepelyshev-Polunchenko 2015 | SR procedure > CUSUM for multi-cyclic monitoring | Not yet implemented |

**Synthesized implication:** The deployed 5 lanes may contain real edge but the discovery methodology was statistically too weak to prove it. The fix is pre-registered hypothesis discovery with a far smaller trial budget (~100-300, not 35,000), applying DSR correctly at the true N, using CPCV for OOS extraction from limited history, and CUSUM/SR monitoring in live deployment.

## What's NOT in this directory

- Specific strategy details (live in `prop_profiles.py`, `live_config`, `validated_setups`)
- Trading rules (see `TRADING_RULES.md`)
- Code implementation details (see `CLAUDE.md` + codebase)
- Memory entries (see `.claude/memory/`)
- Operational runtime state (see `docs/runtime/`)

## Cross-references

- `CLAUDE.md` — project-wide instructions. **References this directory** under Document Authority and Institutional Rigor sections (committed `390e408`, 2026-04-07).
- `.claude/rules/research-truth-protocol.md` — enforcement of literature grounding. **Has Phase 0 section** requiring `pre_registered_criteria.md` reading + hypothesis file before any discovery (committed `0028333`, 2026-04-07).
- `.claude/rules/institutional-rigor.md` — Seven Sins of quant investing. **Rule 7 updated** to point at `literature/` as canonical citation source (committed `0028333`, 2026-04-07).
- `.claude/rules/quant-audit-protocol.md` — audit checklist
- `resources/` — source PDFs (the canonical text — literature files in `literature/` are extracts only)
- `RESEARCH_RULES.md` — should be updated to reference this directory (NOT YET DONE — pending separate commit)
