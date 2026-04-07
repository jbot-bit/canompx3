# Institutional Methodology — Literature-Grounded Framework

**Established:** 2026-04-07
**Purpose:** Ground all research, backtesting, and deployment decisions in specific, citable passages from institutional-grade literature in `resources/`. Replaces informal "training memory" citations.

**Trigger for creation:** April 2026 audit revealed that prior discovery methodology tested ~35,000 combinations on ~6 years of clean MNQ data, violating Bailey-López de Prado MinBTL bounds by ~600x. Root cause: brute-force combinatorial discovery without pre-registered hypothesis budget. This directory locks in corrected institutional standards.

---

## Directory layout

```
docs/institutional/
├── README.md                                   ← you are here
├── literature/                                 ← verbatim passages from resources/*.pdf
│   ├── bailey_et_al_2013_pseudo_mathematics.md      ← MinBTL theorem (CRITICAL)
│   ├── bailey_lopez_de_prado_2014_deflated_sharpe.md ← DSR formula (CRITICAL)
│   ├── lopez_de_prado_bailey_2018_false_strategy.md  ← K>>1 false positive
│   ├── harvey_liu_2015_backtesting.md                ← BHY haircut / profitability hurdles
│   ├── chordia_et_al_2018_two_million_strategies.md  ← t=3.79 threshold
│   ├── pepelyshev_polunchenko_2015_cusum_sr.md       ← live monitoring
│   └── lopez_de_prado_2020_ml_for_asset_managers.md  ← theory-first, CPCV
├── finite_data_framework.md                    ← synthesized approach for short micro data
├── pre_registered_criteria.md                  ← LOCKED thresholds (no post-hoc changes)
└── hypothesis_registry_template.md             ← template for pre-registered discovery
```

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

- `CLAUDE.md` — project-wide instructions (should reference this directory under "Research methodology")
- `.claude/rules/research-truth-protocol.md` — enforcement of literature grounding
- `.claude/rules/integrity-guardian.md` — Seven Sins of quant investing
- `.claude/rules/quant-audit-protocol.md` — audit checklist
- `resources/` — source PDFs (the canonical text — literature files in `literature/` are extracts only)
- `RESEARCH_RULES.md` — should be updated to reference this directory
