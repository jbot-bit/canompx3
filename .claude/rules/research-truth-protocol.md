---
paths:
  - "trading_app/strategy_*"
  - "trading_app/outcome_*"
  - "research/**"
  - "scripts/tools/*"
  - "scripts/reports/*"
---
# Research Truth Protocol (enforced 2026-03-24)

Canonical authority: `RESEARCH_RULES.md` § Discovery Layer Discipline.
This file enforces the protocol in Claude Code sessions.

## Layer Classification

| Layer | Status | Safe for discovery? |
|-------|--------|---------------------|
| `bars_1m` | CANONICAL | YES |
| `daily_features` | CANONICAL | YES |
| `orb_outcomes` | CANONICAL | YES — primary trade-level truth |
| `validated_setups` | DERIVED | **NO** — may be stale/contaminated |
| `edge_families` | DERIVED | **NO** |
| `live_config` / LIVE_PORTFOLIO | HARDCODED | **NO** — deployment state, not research truth |
| Docs, comments, memory files | META | **NO** — verify against canonical layers first |

## Mode B grandfathered validated_setups baselines (MANDATORY warning, added 2026-04-18)

**NEVER use `validated_setups.expectancy_r` as a positive-control or
comparison baseline for a Mode A scan without first recomputing the value
against strict Mode A IS from canonical `orb_outcomes`.**

Background: the project transitioned from Mode B (grandfathered holdout) to
Mode A (strict 2026-01-01 holdout) via Amendment 2.7 on 2026-04-08
(`docs/plans/2026-04-07-holdout-policy-decision.md`). `validated_setups`
rows with `last_trade_day` in the window `[2026-01-01, 2026-04-08]` were
computed under Mode B — their `expectancy_r` reflects an IS window that
includes 2026 Q1 data which is now sacred OOS under Mode A.

Example trap (2026-04-18 VWAP comprehensive family scan):
- Pre-reg H3 positive control cited `validated_setups.expectancy_r = +0.2101`
  for the L6 lane (MNQ US_DATA_1000 O15 RR1.5 long VWAP_MID_ALIGNED) based
  on Mode B grandfathered row with N=701, `last_trade_day=2026-04-02`
- Scan harness correctly computed strict Mode A IS (trading_day < 2026-01-01),
  giving N=435 and ExpR=+0.1817 — a 247-trade gap and 0.028 R difference
  from the cited Mode B baseline
- Scan's K2 positive-control threshold of ±0.02 fired as a FAIL because of
  this specification error, NOT because of any harness defect (independent
  SQL at run time reproduced the harness's +0.1817 exactly)
- Post-run correction required; scan's K1 substantive verdict (0 family
  survivors) was independent of the H3 calibration error

**Required practice for positive-control baselines:**
1. Identify the exact `(instrument, orb_label, orb_minutes, rr_target,
   entry_model, confirm_bars, filter_type, direction)` tuple under test
2. Compute IS ExpR from canonical `orb_outcomes JOIN daily_features` with
   `trading_day < trading_app.holdout_policy.HOLDOUT_SACRED_FROM` — the
   SAME window the harness will consume
3. Cite that computed value in the pre-reg, NOT the `validated_setups` row
4. If the pre-reg writer cannot run SQL, the pre-reg must state the
   baseline as "to be computed by the harness" and include a separate
   independent-SQL verification step before the K2 kill criterion is
   evaluated

The pre-reg writer prompt (`docs/prompts/prereg-writer-prompt.md`) enforces
this at self-review time; violations block the pre-reg from committing.

**Also applies to:**
- WFE baseline comparisons (Mode B WFE may differ from Mode A WFE)
- OOS extrapolation tests (Mode B `oos_exp_r` is NOT the Mode A OOS)
- Sharpe ratio comparisons (Mode B IS Sharpe is computed on a different
  sample)

When in doubt, recompute. Canonical layers (`bars_1m`/`daily_features`/
`orb_outcomes`) are the only admissible source for baselines; `validated_setups`
is a stale snapshot whose IS window may not match the current Mode A spec.

## Canonical filter delegation (MANDATORY, added 2026-04-18)

Research scan scripts MUST NOT re-encode filter logic from
`trading_app.config.ALL_FILTERS`. Use the canonical helper
`research.filter_utils.filter_signal(df, key, orb_label)` which delegates
to `ALL_FILTERS[key].matches_df(df, orb_label)`.

Origin: 2026-04-18 VWAP comprehensive family scan code review caught two
local functions (`vwap_signal`, `deployed_filter_signal`) re-encoding
canonical filter logic. When the refactor landed on the canonical path,
one cell's N_on_IS count corrected from 436 to 435 — a 1-row divergence
invisible to any test except direct equivalence. Parallel filter implementations
WILL drift. The canonical wrapper exists to make drift impossible.

- Wrapper: `research/filter_utils.py`
- Equivalence tests: `tests/test_research/test_filter_utils.py` (15 tests
  covering VWAP_MID_ALIGNED, VWAP_BP_ALIGNED, ORB_G5, ORB_G8, OVNRNG_100,
  ATR_P50)
- Prohibited patterns: any function in `research/` that computes a 0/1
  signal mimicking a registered `StrategyFilter.matches_df` implementation
  without delegating to it. Review these as code-review B-grade violations
  minimum.

## Research Claim Requirements

Every research claim must include:
1. Source layer (must be canonical)
2. Data state timestamp (e.g., "orb_outcomes through 2026-03-23")
3. Exact query or script path
4. Sample size (N)
5. p-value (two-tailed t-test, exact)
6. K used for BH FDR (report both global K and instrument/family K; use instrument/family K for promotion decisions, global K for headline claims; never swap K post-hoc)
7. WFE (if walk-forward was performed)

## Validated Universe Rule (MANDATORY — added 2026-04-04)

**NEVER run research queries against the full unfiltered `orb_outcomes` table.**

`orb_outcomes` contains every possible outcome for every session, entry model, RR target, and confirm_bars — 3M+ rows of noise. Testing a new feature against this undifferentiated mass is testing noise against noise. Any "signal" found is meaningless because:
1. Most of those parameter combos are NOT validated and have negative expectancy
2. The massive N (millions) makes even 0.01R random fluctuations "statistically significant"
3. Filters are not applied — you're mixing filtered and unfiltered trade populations

**Research queries MUST be scoped to one of:**
- The 124 validated strategies (join `validated_setups` to get the strategy dimensions, then query `orb_outcomes` with those exact filters applied via `daily_features`)
- A specific hypothesis about a specific session+instrument+filter combo
- The deployed portfolio (strategy IDs from `prop_profiles.ACCOUNT_PROFILES`)

**Template for valid research query:**
```sql
-- Get outcomes for VALIDATED strategies only, with filters applied
SELECT o.*, d.prev_day_close, d.prev_day_low, d.prev_day_range
FROM orb_outcomes o
JOIN daily_features d
    ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
JOIN validated_setups v
    ON o.symbol = v.instrument AND o.orb_label = v.orb_label
    AND o.orb_minutes = 5 AND o.entry_model = v.entry_model
    AND o.confirm_bars = v.confirm_bars AND o.rr_target = v.rr_target
WHERE v.status = 'active'
  AND [apply v.filter_type condition from daily_features]
```

**If you catch yourself writing `FROM orb_outcomes WHERE symbol IN ('MGC','MNQ','MES')` without a validated_setups join or explicit filter application — STOP. You are about to test noise.**

## Hard Rules

- If docs conflict with canonical data → docs are STALE. Mark them, do not trust them.
- 2026 holdout is sacred. Do not use for discovery. Forward-test = monitoring only.
- No edits before read-only audit. Non-trivial changes require PASS 1 (audit) before PASS 2 (implement).
- Derived layers marked `DISCOVERY SAFETY: UNSAFE` in their module docstrings.
- **NEVER simulate strategy P&L without applying the strategy's filter.** Querying `orb_outcomes` with only instrument+session+entry_model+RR but WITHOUT the filter_type (COST_LT, VOL_RV, ATR70_VOL, OVNRNG, ORB_G, etc.) produces UNFILTERED results that overcount trades and misrepresent both P&L and risk. Every simulation MUST join `daily_features` and apply the exact filter condition. If the filter column is missing or broken, say UNVERIFIED — do not substitute `1=1`.

## Phase 0 Literature Grounding (MANDATORY — added 2026-04-07)

Before running ANY discovery that writes to `experimental_strategies` or `validated_setups`, you MUST:

1. **Read `docs/institutional/pre_registered_criteria.md`** — the 12 locked criteria every validated strategy must meet. These were derived from the Phase 0 literature extraction and are NOT subject to post-hoc relaxation.

2. **Write a pre-registered hypothesis file** at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml` BEFORE any backtest. Template: `docs/institutional/hypothesis_registry_template.md`. The file must include:
   - Numbered hypotheses with economic theory citations
   - Exact filter dimensions and threshold ranges
   - Total expected trial count (N ≤ 300 clean-MNQ, N ≤ 2000 proxy-extended)
   - Kill criteria stating what outcome refutes the hypothesis

2a. **Pre-reg writer gate (added 2026-04-18, binding).** Every new file under `docs/audit/hypotheses/` MUST be generated via `docs/prompts/prereg-writer-prompt.md` OR satisfy its output schema 1:1. Before committing the pre-reg, a pre-commit self-review against the prompt's § FORBIDDEN and § failure-mode table is mandatory. In particular:
   - `testing_mode` declared: `family` (Pathway A / BH FDR) or `individual` (Pathway B / theory-driven K=1 per pre_registered_criteria.md Amendment 3.0).
   - `pathway` field set matching `testing_mode`.
   - For Pathway B, every hypothesis has a `theory_citation`; `testing_discipline.mandatory_downstream_gates_non_waivable` lists C6/C8/C9.
   - Upstream scan K values (if any) live under `upstream_discovery_provenance` with `role: PROVENANCE_ONLY` — never under the current test's K framing.
   - Kill criteria are numeric. "Reconsider" / "investigate" is forbidden.
   - No `TO_FILL_*` placeholders other than `commit_sha: "TO_FILL_AFTER_COMMIT"` (which is legitimate until the first commit lands).

   Origin: the 2026-04-18 Phase D D-0 pre-reg was written without this gate, producing a framing error (testing_mode=family on a K=1 confirmatory pilot, and upstream K=14,261 presented as the pilot's own K). Caught post-run; required a documentation-only correction commit (`93a8e53a`). The D-0 KILL verdict was unaffected but the patch cycle wasted cycles. This gate prevents recurrence.

3. **Cite the specific literature file** for each statistical method used. Files live in `docs/institutional/literature/`. Never cite thresholds from training memory.

4. **MinBTL check:** Before running any enumeration, compute `MinBTL = 2·Ln[N] / E[max_N]²` using the committed pre-registered N. If `MinBTL > available_clean_data_years`, reduce N. Source: `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`.

5. **No brute-force sweeps >300 trials.** The April 2026 audit established that prior brute-force discovery (~35,000 trials on 2.2-6 years of clean data) violated Bailey et al 2013 MinBTL by ~600x. Enumeration budgets must be pre-committed and bounded.

### Criteria summary (see `pre_registered_criteria.md` for full text)

| # | Criterion | Threshold | Source |
|---|-----------|-----------|--------|
| 1 | Pre-registered hypothesis file | exists before run | LdP 2020 + Bailey 2013 |
| 2 | MinBTL constraint | N ≤ 300 (or 2000 proxy) | Bailey et al 2013 |
| 3 | BH FDR | q < 0.05 on pre-reg family | Harvey-Liu 2015 + Chordia 2018 |
| 4 | Chordia t-statistic | t ≥ 3.00 (w/ theory) or 3.79 (w/o) | Chordia et al 2018 |
| 5 | Deflated Sharpe Ratio | DSR > 0.95 | Bailey-LdP 2014 Eq. 2 |
| 6 | Walk-forward efficiency | WFE ≥ 0.50 | LdP 2020 + project convention |
| 7 | Sample size | N ≥ 100 trades deployable | HL 2015 Exhibit 4 |
| 8 | 2026 OOS positive | OOS ExpR ≥ 0.40 × IS | OOS principle |
| 9 | Era stability | no era ExpR < −0.05 (N≥50) | 2026-04-07 audit finding |
| 10 | Data era compatibility | volume filters MICRO-only | 2026-04-07 audit finding |
| 11 | Account death Monte Carlo | 90-day survival ≥ 70% | prop firm rulesets |
| 12 | Shiryaev-Roberts monitor | live drift detection active | Pepelyshev-Polunchenko 2015 |

### Applying to existing validated_setups

Strategies written to `validated_setups` BEFORE 2026-04-07 were discovered under the pre-Phase-0 regime and are provisionally grandfathered but MUST be re-audited against the 12 criteria before any scaling decision. The current 5 deployed MNQ/MGC lanes in `topstep_50k_mnq_auto` are in this provisional bucket — no scaling until re-audit.

### Interaction with this rule file

- This Phase 0 section adds to, does not replace, the Validated Universe Rule above.
- The two are complementary: Validated Universe Rule prevents testing noise against noise on unfiltered `orb_outcomes`; Phase 0 prevents the resulting pre-filtered universe from being brute-forced without a hypothesis budget.
- Both must be followed for any research claim to be valid.
