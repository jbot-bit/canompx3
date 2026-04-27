# Sizing-Substrate Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Stage-1 sizing-substrate diagnostic per `docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md` v0.2 — produce pre-reg YAML, diagnostic script, tests, and result artifacts that decide whether deployed binary filters have continuous substrate justifying a Stage-2 sizing layer.

**Architecture:** A single read-only Python script (`research/audit_sizing_substrate_diagnostic.py`) that reads `gold.db`, computes 48 cells (6 lanes × 8 features) of (Spearman ρ, quintile-lift, sized-vs-flat ExpR delta with bootstrap CI, split-half stability), applies BH-FDR at q=0.05, and emits a markdown + JSON result. Pre-reg YAML is locked in git BEFORE the script runs. TDD: every gate in spec §5.2-§5.4a has a test before implementation.

**Tech Stack:** Python 3.12, DuckDB (read-only), pandas, numpy, scipy.stats, PyYAML; pytest.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml` | Create | Locked pre-reg: feature list, weights, thresholds, ex-ante directions, literature extracts |
| `research/audit_sizing_substrate_diagnostic.py` | Create | Read-only diagnostic; loads tape, computes per-cell metrics, applies BH-FDR, emits artifacts |
| `tests/test_research/test_audit_sizing_substrate_diagnostic.py` | Create | Pure-function tests for every gate (holdout-raise, NULL-guard, power, BH-FDR, sized-vs-flat, split-half, stability) |
| `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md` | Generate | Run output: per-cell table + per-lane summary + global verdict |
| `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json` | Generate | Machine-readable twin of the markdown |

The script is deliberately split into pure functions (testable) + a thin `main()` (I/O). No production code under `pipeline/` or `trading_app/` is modified.

---

## Task 1: Pre-precheck — verify lane IS sample sizes

**Files:**
- No file changes; produces a console report used to choose the power-floor strategy.

**Why this comes first:** Spec §5.2 step 4 requires N≥902 per cell for ρ=0.10 detection at t≥3.00. SINGAPORE_OPEN (lane 2, O15, ATR_P50) had N=137 trailing 12mo. If full-IS N is below 902 for any lane, that lane's cells will all be UNDERPOWERED and the diagnostic structurally cannot achieve its global pass condition. We need to confirm IS-N before locking pre-reg.

- [ ] **Step 1: Run IS sample size precheck**

```bash
cd C:\Users\joshd\canompx3
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
LANES = [
    ('MNQ', 'EUROPE_FLOW',     5,  1.5, 1, 'E2'),
    ('MNQ', 'SINGAPORE_OPEN', 15,  1.5, 1, 'E2'),
    ('MNQ', 'COMEX_SETTLE',    5,  1.5, 1, 'E2'),
    ('MNQ', 'NYSE_OPEN',       5,  1.0, 1, 'E2'),
    ('MNQ', 'TOKYO_OPEN',      5,  1.5, 1, 'E2'),
    ('MNQ', 'US_DATA_1000',   15,  1.5, 1, 'E2'),
]
print(f'{\"lane\":24s} {\"N_IS\":>8s} {\"first\":>12s} {\"last\":>12s}  power_ok')
for inst, lab, om, rr, cb, em in LANES:
    q = '''SELECT COUNT(*), MIN(trading_day), MAX(trading_day) FROM orb_outcomes
           WHERE symbol=? AND orb_label=? AND orb_minutes=? AND rr_target=?
             AND confirm_bars=? AND entry_model=? AND pnl_r IS NOT NULL
             AND trading_day < DATE \\'2026-01-01\\''''
    n, fd, ld = con.execute(q, [inst, lab, om, rr, cb, em]).fetchone()
    ok = 'YES' if n >= 902 else 'NO  (UNDERPOWERED at rho=0.10)'
    print(f'{inst}_{lab}_{em}_O{om}_RR{rr}'.ljust(24) + f' {n:>8d} {str(fd):>12s} {str(ld):>12s}  {ok}')
"
```

Expected: each lane prints N. Pass criterion for proceeding to pre-reg lock: at least 4 of 6 lanes have N ≥ 902. If fewer, replan: either use t≥2.0 (Pathway A weak grounding) with explicit caveat, or drop the lane from the diagnostic.

- [ ] **Step 2: Record numbers in plan**

Manually paste the output table into a comment at the top of `docs/plans/2026-04-27-sizing-substrate-diagnostic-implementation-plan.md` under Task 1 step 2 marker once observed. (Do NOT modify pre-reg yet — it doesn't exist.)

- [ ] **Step 3: Commit precheck note**

```bash
git add docs/plans/2026-04-27-sizing-substrate-diagnostic-implementation-plan.md
git commit -m "research(sizing): record lane IS sample sizes for power gate (precheck)"
```

---

## Task 2: Verify filter formula line numbers

**Files:**
- Read-only inspection: `trading_app/config.py`, `pipeline/cost_model.py`, `research/filter_utils.py`

Spec §5.1 cites filter classes at `trading_app/config.py:2980, 3072, 3274` (verified at design time). Step 1 here re-confirms before pre-reg lock; Step 2 captures `research/filter_utils.py:filter_signal()` semantics (the canonical filter-application function used by every audit script).

- [ ] **Step 1: Re-confirm filter class line numbers**

```bash
grep -nE '"G5":|"ATR_P50":|"COST_LT12":' trading_app/config.py
```

Expected: three matches near lines 2980, 3072, 3274. If line numbers drift, update Task 4's pre-reg YAML accordingly.

- [ ] **Step 2: Read `research/filter_utils.filter_signal` signature**

```bash
grep -nA 5 "^def filter_signal" research/filter_utils.py
```

Expected: signature `filter_signal(df, filter_type: str, session: str) -> pd.Series` returning a 0/1 fire vector. The diagnostic does NOT call `filter_signal` for the binary cut (we already pull only deployed-filter trades from `orb_outcomes`); but the continuous substrate definitions in step 3 must agree with the binary's underlying feature.

- [ ] **Step 3: Inspect `OrbSizeFilter`, `OwnATRPercentileFilter`, `CostRatioFilter` definitions**

```bash
grep -nA 20 "^class OrbSizeFilter" trading_app/config.py
grep -nA 25 "^class OwnATRPercentileFilter" trading_app/config.py
grep -nA 25 "^class CostRatioFilter" trading_app/config.py
```

Record on a scratch line for Task 4: which `daily_features` column each filter uses (e.g. `OrbSizeFilter` → `orb_<SESSION>_size`; `OwnATRPercentileFilter` → `atr_20_pct`; `CostRatioFilter` → cost ratio computation). Capture the exact line ranges to cite in the pre-reg YAML.

---

## Task 3: Extract literature passages for pre-reg

**Files:**
- Read-only inspection of `resources/Robert Carver - Systematic Trading.pdf`, `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`, `resources/Pseudo-mathematics-and-financial-charlatanism.pdf`, `resources/deflated-sharpe.pdf`, `resources/false-strategy-lopez.pdf`.
- Output: a temporary scratch file `docs/audit/hypotheses/_scratch-2026-04-27-extracts.md` (gitignored, not committed) with literal extracts for paste into Task 4.

- [ ] **Step 1: Extract Carver Ch. 7 passages**

```bash
pdftotext -layout "resources/Robert Carver - Systematic Trading.pdf" - | sed -n '/^Chapter 7$/,/^Chapter 8$/p' > /tmp/carver_ch7.txt
grep -nE "expected absolute value of 10|capped at|forecast scalar|forecasts shouldn't be binary|stable standard deviation" /tmp/carver_ch7.txt
```

Expected: located passages on target=10, cap=20, scalar formula, binary-vs-continuous quote, fn 78 stability. Copy literal text into the scratch file under headings: **Carver Ch.7 forecast scaling**, **Carver Ch.7 cap mechanics**, **Carver Ch.7 binary vs continuous**, **Carver Ch.7 fn 78 stability**.

- [ ] **Step 2: Extract Carver Ch. 8 combined-forecast multiplier passage**

```bash
pdftotext -layout "resources/Robert Carver - Systematic Trading.pdf" - | sed -n '/^Chapter 8$/,/^Chapter 9$/p' > /tmp/carver_ch8.txt
grep -nE "Getting to 10|forecast diversification multiplier|combined forecast" /tmp/carver_ch8.txt
```

Expected: paragraph on "Getting to 10" and the diversification multiplier formula. Copy literal text into scratch under **Carver Ch.8 combined-forecast multiplier (Stage 2 reference)**.

- [ ] **Step 3: Extract ML4AM §1 footnote 7 passage**

```bash
pdftotext -layout "resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf" - | sed -n '1,80p' > /tmp/ml4am_ch1.txt
grep -nB1 -A2 "meta-labeling\|sign and size" /tmp/ml4am_ch1.txt
```

Expected: §1 mention of sign-vs-size decoupling + footnote 7 deferral to AFML 2018a Ch. 19. Copy literal text into scratch under **ML4AM §1 fn 7 sign-vs-size decoupling**.

- [ ] **Step 4: Extract Bailey/Lopez selection-bias passages**

```bash
pdftotext -layout "resources/Pseudo-mathematics-and-financial-charlatanism.pdf" - > /tmp/pseudo.txt
grep -nB1 -A3 "Minimum Backtest Length\|MinBTL\|necessary, non-sufficient" /tmp/pseudo.txt | head -20

pdftotext -layout "resources/deflated-sharpe.pdf" - > /tmp/dsr.txt
grep -nB1 -A3 "Selection bias under multiple testing\|Deflated Sharpe Ratio" /tmp/dsr.txt | head -10

pdftotext -layout "resources/false-strategy-lopez.pdf" - > /tmp/false_strategy.txt
grep -nB1 -A3 "selection bias under multiple testing\|undisclosed N of trials\|single test" /tmp/false_strategy.txt | head -10
```

Expected: passages on multiple-testing selection bias + DSR + MinBTL. Copy literal text into scratch under **Bailey/Lopez selection-bias frame** with citations to each PDF's section.

- [ ] **Step 5: Verify Aronson Ch. 6 is data-mining bias (NOT quantization)**

```bash
pdftotext -layout "resources/Evidence_Based_Technical_Analysis_Aronson.pdf" - | grep -nA 1 "^Chapter 6\|Data-Mining Bias"
```

Expected: confirms Ch. 6 title is "Data-Mining Bias". Record in scratch under **Aronson Ch.6 dropped (verified topic mismatch)**. The continuous-vs-binary doctrinal claim is grounded directly in Carver Ch. 7 from Step 1.

- [ ] **Step 6: Save scratch (gitignored, do NOT commit)**

```bash
echo "# Literature extracts scratch — DO NOT COMMIT" > docs/audit/hypotheses/_scratch-2026-04-27-extracts.md
# Append all extracted passages from steps 1-5 here.
echo "_scratch-*.md" >> docs/audit/hypotheses/.gitignore  # if not already gitignored
```

Verify: `git status` should NOT show the scratch file as tracked.

---

## Task 4: Write the pre-registered hypothesis YAML

**Files:**
- Create: `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`

This is the lock file. Once committed, K, features, thresholds, weights, and ex-ante directions are frozen. The diagnostic script reads this file at run time to drive its computation.

- [ ] **Step 1: Write the YAML**

```yaml
# Hypothesis Registry — 2026-04-27 Sizing-Substrate Diagnostic (Stage 1)
#
# Locked: 2026-04-27T<HH:MM>+10:00
# Author: Claude session (joshd)
# Committing commit hash: <fill at commit time>
# Origin: docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md v0.2 (commit c0d18bca)
#
# Decision-authority: Stage-1 falsifier of the "convert binary filters into
# continuous sizing" thesis. Diagnostic only — passing cells qualify a Stage-2
# sizing-model pre-reg, they do not authorize live changes.

metadata:
  name: "sizing_substrate_diagnostic_stage1"
  purpose: |
    Test whether the 6 deployed lanes' binary filters have continuous
    substrate that justifies a Carver-style forecast-scaling sizing layer.
    Per design v0.2: K=48 cells (6 lanes x 8 features), BH-FDR q=0.05,
    sized-vs-flat ExpR delta with 10k bootstrap CI seed=42, split-half
    sign stability, ex-ante direction prediction, power floor N>=902,
    NULL coverage <= 20%, forecast-stability gate.
  date_locked: "2026-04-27T<HH:MM>+10:00"
  commit_sha: "<fill at commit time>"
  holdout_date: "2026-01-01"
  testing_mode: "diagnostic_descriptive"  # Per pre_registered_criteria.md RULE 10 carve-out: descriptive diagnostics that do not write to validated_setups
  research_question_type: "edge_elasticity"
  total_expected_trials: 48
  k_global: 48
  bh_fdr_q: 0.05  # Pre-reg Criterion 3 Pathway A canonical
  bootstrap_B: 10000
  bootstrap_seed: 42
  power_floor_N: 902  # for rho=0.10 at t>=3.00 (Harvey-Liu-Zhu with-theory)
  null_coverage_max: 0.20
  stability_sd_variation_max: 0.50
  rho_min: 0.10
  q5_minus_q1_min: 0.20

# ---------------------------------------------------------------------------
# Mechanism prior — locked BEFORE any data is examined
# ---------------------------------------------------------------------------
mechanism_priors:
  carver_ch7_per_rule_scaling: |
    [PASTE LITERAL EXTRACT FROM TASK 3 STEP 1: target abs forecast = 10, cap at +/-20, scalar = 10 / natural-mean-abs-forecast]
  carver_ch7_continuous_vs_binary: |
    [PASTE LITERAL EXTRACT FROM TASK 3 STEP 1: "forecasts shouldn't be binary... it's better to see forecasts changing continuously rather than jumping around"]
  carver_ch7_fn78_stability: |
    [PASTE LITERAL EXTRACT FROM TASK 3 STEP 1: forecasts should have well defined and relatively stable standard deviations]
  carver_ch8_combined_multiplier: |
    [PASTE LITERAL EXTRACT FROM TASK 3 STEP 2: Getting to 10 + diversification multiplier — Stage 2 reference]
  ml4am_section1_fn7_sign_vs_size: |
    [PASTE LITERAL EXTRACT FROM TASK 3 STEP 3: ML for Asset Managers section 1 footnote 7 — sigmoid bet-sizer recipe deferred to AFML 2018a Ch.19, NOT in resources/]
  bailey_selection_bias: |
    [PASTE LITERAL EXTRACT FROM TASK 3 STEP 4: pseudo-mathematics + DSR + false-strategy on multiple-testing selection]
  aronson_dropped_note: |
    Aronson Ch.6 verified as "Data-Mining Bias", not quantization loss. Citation
    dropped from design v0.2. Continuous-vs-binary doctrine grounded directly
    in Carver Ch.7 quote above.

# ---------------------------------------------------------------------------
# Universe of cells — locked
# ---------------------------------------------------------------------------
lanes:
  - id: "MNQ_EUROPE_FLOW_E2_O5_RR1.5_CB1_ORB_G5"
    instrument: "MNQ"
    orb_label: "EUROPE_FLOW"
    orb_minutes: 5
    rr_target: 1.5
    confirm_bars: 1
    entry_model: "E2"
    deployed_filter: "ORB_G5"
    deployed_filter_class: "OrbSizeFilter"
    deployed_filter_class_line: 2980  # trading_app/config.py — verify in Task 2
    binary_threshold: "min_size=5.0 (ORB size >= 5 points)"
  - id: "MNQ_SINGAPORE_OPEN_E2_O15_RR1.5_CB1_ATR_P50"
    instrument: "MNQ"
    orb_label: "SINGAPORE_OPEN"
    orb_minutes: 15
    rr_target: 1.5
    confirm_bars: 1
    entry_model: "E2"
    deployed_filter: "ATR_P50"
    deployed_filter_class: "OwnATRPercentileFilter"
    deployed_filter_class_line: 3072
    binary_threshold: "median split on atr_20_pct"
  - id: "MNQ_COMEX_SETTLE_E2_O5_RR1.5_CB1_ORB_G5"
    instrument: "MNQ"
    orb_label: "COMEX_SETTLE"
    orb_minutes: 5
    rr_target: 1.5
    confirm_bars: 1
    entry_model: "E2"
    deployed_filter: "ORB_G5"
    deployed_filter_class: "OrbSizeFilter"
    deployed_filter_class_line: 2980
    binary_threshold: "min_size=5.0"
  - id: "MNQ_NYSE_OPEN_E2_O5_RR1.0_CB1_COST_LT12"
    instrument: "MNQ"
    orb_label: "NYSE_OPEN"
    orb_minutes: 5
    rr_target: 1.0
    confirm_bars: 1
    entry_model: "E2"
    deployed_filter: "COST_LT12"
    deployed_filter_class: "CostRatioFilter"
    deployed_filter_class_line: 3274
    binary_threshold: "cost_ratio < 0.12 R"
  - id: "MNQ_TOKYO_OPEN_E2_O5_RR1.5_CB1_COST_LT12"
    instrument: "MNQ"
    orb_label: "TOKYO_OPEN"
    orb_minutes: 5
    rr_target: 1.5
    confirm_bars: 1
    entry_model: "E2"
    deployed_filter: "COST_LT12"
    deployed_filter_class: "CostRatioFilter"
    deployed_filter_class_line: 3274
    binary_threshold: "cost_ratio < 0.12 R"
  - id: "MNQ_US_DATA_1000_E2_O15_RR1.5_CB1_ORB_G5"
    instrument: "MNQ"
    orb_label: "US_DATA_1000"
    orb_minutes: 15
    rr_target: 1.5
    confirm_bars: 1
    entry_model: "E2"
    deployed_filter: "ORB_G5"
    deployed_filter_class: "OrbSizeFilter"
    deployed_filter_class_line: 2980
    binary_threshold: "min_size=5.0"

features:
  tier_a:
    # 3 functional forms per substrate, applied to each lane's deployed-filter substrate.
    # Ex-ante direction is set per lane based on filter mechanism — see directions section.
    - form_id: "raw"
      description: "Raw substrate value at trade time (lane-relative)"
    - form_id: "vol_norm"
      description: "Substrate divided by atr_20_pct (vol-adjusted)"
    - form_id: "rank_252d"
      description: "Substrate as 252d rolling-percentile rank (regime-relative)"
  tier_b:
    # 5 orthogonal continuous features applied to all 6 lanes.
    - feature_id: "rel_vol_session"
      column_template: "rel_vol_{ORB_LABEL}"
      ex_ante_direction: "+"  # Higher participation predicted to give cleaner break per-trade.
    - feature_id: "overnight_range_pct"
      column: "overnight_range_pct"
      ex_ante_direction: "+"  # Wider overnight range predicted to prefigure bigger trend day.
    - feature_id: "atr_vel_ratio"
      column: "atr_vel_ratio"
      ex_ante_direction: "+"  # Rising-vol regime predicted to favour breakout follow-through.
    - feature_id: "garch_forecast_vol_pct"
      column: "garch_forecast_vol_pct"
      ex_ante_direction: "+"  # Forecast-vol elevated days predicted to have larger realized R.
    - feature_id: "pit_range_atr"
      column: "pit_range_atr"
      ex_ante_direction: "+"  # Wider pit range predicted to indicate already-active price discovery.

ex_ante_directions_tier_a:
  # Per (lane, substrate-form) — set BEFORE running, locked at YAML commit
  ORB_G5_raw: "+"            # Bigger ORB predicted to signal stronger trend day (Crabel "compression-then-expansion" — bigger expansion -> bigger follow-through)
  ORB_G5_vol_norm: "+"
  ORB_G5_rank_252d: "+"
  ATR_P50_raw: "+"           # Higher ATR predicted to give bigger R per trade (Carver scale-with-volatility)
  ATR_P50_vol_norm: "+"
  ATR_P50_rank_252d: "+"
  COST_LT12_raw: "-"         # HIGHER cost_ratio predicted to REDUCE realized R (cost is a drag); inverse direction
  COST_LT12_vol_norm: "-"
  COST_LT12_rank_252d: "-"

# ---------------------------------------------------------------------------
# Decision rule (binding once committed)
# ---------------------------------------------------------------------------
decision_rule:
  cell_pass_requires_all:
    - "not INVALID (NULL coverage > 20%)"
    - "not UNDERPOWERED (N < 902 after NULL drop)"
    - "|spearman_rho| >= 0.10"
    - "monotonic Q1->Q5 AND |Q5_mean_R - Q1_mean_R| >= 0.20"
    - "sized-vs-flat ExpR delta 95% bootstrap CI strictly positive"
    - "two-sided p-value survives BH-FDR at q=0.05 over K=48"
    - "split-half: sign of rho matches both halves AND sign of sized-flat delta matches both halves"
    - "realized rho sign matches ex-ante prediction (prediction-flipped cells barred from Stage 2)"
  lane_has_substrate_iff: ">=1 of its 8 cells passes"
  substrate_confirmed_globally_iff: ">=3 of 6 lanes have substrate"
  substrate_weak_iff: "1-2 lanes have substrate (park, no Stage 2)"
  thesis_killed_iff: "0 lanes have substrate (NO-GO entry; reopen requires new mechanism citation)"
  diagnostic_inconclusive_iff: ">=50% INVALID+UNDERPOWERED in any tier (does NOT confirm H0)"

# ---------------------------------------------------------------------------
# Boundary discipline
# ---------------------------------------------------------------------------
boundary:
  read_only: true
  raises_on_holdout_row: true   # script raises RuntimeError if any row has trading_day >= 2026-01-01
  single_pass: true             # no rerunning with different feature lists or thresholds
  reopen_requires_new_mechanism: true
  output_artifact_md: "docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md"
  output_artifact_json: "docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json"
  no_writes_to_validated_setups: true
  no_writes_to_experimental_strategies: true
  no_writes_to_lane_allocation: true
```

- [ ] **Step 2: Replace literal-extract placeholders**

Open `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml` and paste the actual passages from `docs/audit/hypotheses/_scratch-2026-04-27-extracts.md` (Task 3) into each `[PASTE LITERAL EXTRACT FROM TASK 3 STEP N: ...]` placeholder. Verify no placeholder remains:

```bash
grep -n "PASTE LITERAL EXTRACT" docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml
```

Expected: zero matches.

- [ ] **Step 3: Replace timestamp + commit_sha placeholders**

Replace `<HH:MM>` with current Brisbane time. Leave `<fill at commit time>` for now; will be filled by a follow-up edit after the YAML lands its first commit (since the commit SHA is not knowable until commit lands).

- [ ] **Step 4: Validate YAML parses**

```bash
python -c "import yaml; d = yaml.safe_load(open('docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml')); print('lanes:', len(d['lanes']), 'tier_a forms:', len(d['features']['tier_a']), 'tier_b features:', len(d['features']['tier_b']))"
```

Expected: `lanes: 6 tier_a forms: 3 tier_b features: 5`. (3 forms × 6 lanes = 18 Tier-A; 5 × 6 = 30 Tier-B; total 48.)

- [ ] **Step 5: Commit YAML**

```bash
git add docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml
git commit -m "research(sizing): lock pre-reg YAML for sizing-substrate diagnostic (K=48)"
```

Pre-commit checks must pass.

- [ ] **Step 6: Backfill commit_sha into the YAML**

```bash
SHA=$(git log -1 --format=%H docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml)
sed -i "s/<fill at commit time>/$SHA/" docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml
git add docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml
git commit -m "research(sizing): record pre-reg commit_sha"
```

---

## Task 5: TDD — write the failing tests for pure functions

**Files:**
- Create: `tests/test_research/test_audit_sizing_substrate_diagnostic.py`

The script will be split into pure functions. We test the pure ones; `main()` is integration-tested by running it once and snapshotting the output (Task 9). This task writes **all tests first**, runs them to confirm they all FAIL (because the script doesn't exist yet), then commits the failing tests.

- [ ] **Step 1: Write the test file**

```python
# tests/test_research/test_audit_sizing_substrate_diagnostic.py
"""Tests for research/audit_sizing_substrate_diagnostic.py.

Per spec v0.2 §5.2-§5.4a: every gate must be testable in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.audit_sizing_substrate_diagnostic import (
    apply_bh_fdr,
    bootstrap_sized_vs_flat_ci,
    check_forecast_stability,
    classify_cell,
    compute_quintile_lift,
    is_holdout_clean,
    null_coverage_mark,
    power_floor_mark,
    sign_match_split_half,
)


# -- Holdout guard --------------------------------------------------------------

def test_holdout_clean_returns_true_when_all_pre_2026():
    df = pd.DataFrame({"trading_day": pd.to_datetime(["2025-12-30", "2025-12-31"])})
    assert is_holdout_clean(df, holdout="2026-01-01") is True


def test_holdout_clean_raises_when_any_row_in_holdout():
    df = pd.DataFrame({"trading_day": pd.to_datetime(["2025-12-31", "2026-01-02"])})
    with pytest.raises(RuntimeError, match="holdout row"):
        is_holdout_clean(df, holdout="2026-01-01")


# -- NULL coverage --------------------------------------------------------------

def test_null_coverage_pass_when_drop_under_20pct():
    f = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])  # 1/5 = 20% — boundary inclusive
    status, drop_frac = null_coverage_mark(f, threshold=0.20)
    assert status == "OK"
    assert drop_frac == pytest.approx(0.20)


def test_null_coverage_marks_invalid_when_drop_over_20pct():
    f = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])  # 2/5 = 40%
    status, drop_frac = null_coverage_mark(f, threshold=0.20)
    assert status == "INVALID"
    assert drop_frac == pytest.approx(0.40)


# -- Power floor ----------------------------------------------------------------

def test_power_floor_pass_at_902():
    assert power_floor_mark(902, min_n=902) == "OK"


def test_power_floor_fails_at_901():
    assert power_floor_mark(901, min_n=902) == "UNDERPOWERED"


# -- Quintile lift --------------------------------------------------------------

def test_quintile_lift_monotonic_increasing():
    rng = np.random.default_rng(7)
    f = np.linspace(0, 10, 1000)
    pnl = f * 0.05 + rng.normal(0, 0.5, 1000)  # signal + noise, sign POS
    df = pd.DataFrame({"f": f, "pnl_r": pnl})
    res = compute_quintile_lift(df, feature_col="f", outcome_col="pnl_r")
    assert res["monotonic"] is True
    assert res["q5_mean_r"] > res["q1_mean_r"]


def test_quintile_lift_non_monotonic_flagged():
    rng = np.random.default_rng(7)
    f = np.arange(1000)
    pnl = np.sin(f / 50) + rng.normal(0, 0.1, 1000)  # oscillating
    df = pd.DataFrame({"f": f, "pnl_r": pnl})
    res = compute_quintile_lift(df, feature_col="f", outcome_col="pnl_r")
    assert res["monotonic"] is False


# -- Bootstrap sized-vs-flat ----------------------------------------------------

def test_bootstrap_sized_vs_flat_ci_positive_when_real_edge():
    rng = np.random.default_rng(7)
    n = 2000
    f = rng.uniform(0, 1, n)
    pnl = f * 0.5 + rng.normal(0, 0.4, n)  # strong positive edge
    df = pd.DataFrame({"f": f, "pnl_r": pnl})
    ci = bootstrap_sized_vs_flat_ci(
        df, feature_col="f", outcome_col="pnl_r",
        weights=(0.6, 0.8, 1.0, 1.2, 1.4), predicted_sign="+",
        B=10000, seed=42,
    )
    assert ci["lo"] > 0
    assert ci["hi"] > ci["lo"]


def test_bootstrap_seed_reproducibility():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"f": rng.uniform(0, 1, 500), "pnl_r": rng.normal(0, 0.5, 500)})
    a = bootstrap_sized_vs_flat_ci(df, "f", "pnl_r", (0.6, 0.8, 1.0, 1.2, 1.4), "+", B=1000, seed=42)
    b = bootstrap_sized_vs_flat_ci(df, "f", "pnl_r", (0.6, 0.8, 1.0, 1.2, 1.4), "+", B=1000, seed=42)
    assert a == b


# -- BH-FDR ---------------------------------------------------------------------

def test_bh_fdr_passes_strong_signal():
    pvals = [0.0001, 0.001, 0.5, 0.6, 0.7]
    out = apply_bh_fdr(pvals, q=0.05)
    assert out[0] is True   # very small p must survive
    assert out[1] is True
    assert out[3] is False
    assert out[4] is False


def test_bh_fdr_no_survivors_under_uniform_pvals():
    pvals = [0.30, 0.40, 0.50, 0.60, 0.70]
    out = apply_bh_fdr(pvals, q=0.05)
    assert all(v is False for v in out)


# -- Forecast stability ---------------------------------------------------------

def test_forecast_stability_stable_over_time():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "trading_day": pd.date_range("2010-01-01", periods=2000, freq="D"),
        "f": rng.normal(10, 1.0, 2000),  # stable SD
    })
    assert check_forecast_stability(df, feature_col="f", window=252, max_rel_var=0.50) == "STABLE"


def test_forecast_stability_unstable_when_sd_drifts():
    rng = np.random.default_rng(7)
    early = rng.normal(10, 0.2, 1000)   # tight
    late = rng.normal(10, 5.0, 1000)    # loose
    df = pd.DataFrame({
        "trading_day": pd.date_range("2010-01-01", periods=2000, freq="D"),
        "f": np.concatenate([early, late]),
    })
    assert check_forecast_stability(df, feature_col="f", window=252, max_rel_var=0.50) == "UNSTABLE"


# -- Split-half sign stability --------------------------------------------------

def test_sign_match_split_half_passes_when_signs_match():
    df = pd.DataFrame({
        "trading_day": pd.date_range("2010-01-01", periods=1000, freq="D"),
        "f": np.linspace(0, 1, 1000),
        "pnl_r": np.linspace(-0.5, 0.5, 1000),  # both halves positive correlation
    })
    assert sign_match_split_half(df, feature_col="f", outcome_col="pnl_r") is True


def test_sign_match_split_half_fails_when_signs_flip():
    df = pd.DataFrame({
        "trading_day": pd.date_range("2010-01-01", periods=1000, freq="D"),
        "f": np.concatenate([np.linspace(0, 1, 500), np.linspace(0, 1, 500)]),
        "pnl_r": np.concatenate([np.linspace(0, 1, 500), np.linspace(1, 0, 500)]),  # flip
    })
    assert sign_match_split_half(df, feature_col="f", outcome_col="pnl_r") is False


# -- Final classifier (compose) -------------------------------------------------

def test_classify_cell_pass_when_all_gates_pass():
    cell = {
        "null_status": "OK",
        "power_status": "OK",
        "rho": 0.15, "rho_p": 0.0001, "bh_fdr_pass": True,
        "monotonic": True, "q5_minus_q1": 0.30,
        "sized_flat_delta_lo": 0.02, "sized_flat_delta_hi": 0.08,
        "split_half_rho_match": True, "split_half_delta_match": True,
        "predicted_sign": "+", "realized_sign": "+",
        "stability_status": "STABLE",
    }
    assert classify_cell(cell) == "PASS"


def test_classify_cell_fail_when_underpowered():
    cell = {"null_status": "OK", "power_status": "UNDERPOWERED", "rho": 0.30}
    assert classify_cell(cell) == "FAIL"


def test_classify_cell_fail_when_prediction_flipped():
    cell = {
        "null_status": "OK", "power_status": "OK",
        "rho": 0.15, "rho_p": 0.0001, "bh_fdr_pass": True,
        "monotonic": True, "q5_minus_q1": 0.30,
        "sized_flat_delta_lo": 0.02, "sized_flat_delta_hi": 0.08,
        "split_half_rho_match": True, "split_half_delta_match": True,
        "predicted_sign": "+", "realized_sign": "-",  # flipped
        "stability_status": "STABLE",
    }
    assert classify_cell(cell) == "FAIL"


def test_classify_cell_invalid_when_null_heavy():
    cell = {"null_status": "INVALID"}
    assert classify_cell(cell) == "INVALID"
```

- [ ] **Step 2: Run tests — confirm ALL fail (script not yet implemented)**

```bash
pytest tests/test_research/test_audit_sizing_substrate_diagnostic.py -v
```

Expected: every test errors with `ModuleNotFoundError: No module named 'research.audit_sizing_substrate_diagnostic'` OR collection error from missing imports.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_research/test_audit_sizing_substrate_diagnostic.py
git commit -m "test(sizing): failing tests for sizing-substrate diagnostic gates (TDD)"
```

---

## Task 6: Implement pure functions to pass the tests

**Files:**
- Create: `research/audit_sizing_substrate_diagnostic.py` (pure functions only at this step; `main()` and SQL come in Task 7)

- [ ] **Step 1: Write the pure-function module skeleton**

```python
# research/audit_sizing_substrate_diagnostic.py
"""Sizing-substrate Stage-1 diagnostic.

Per docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md v0.2 and the
locked pre-reg YAML at docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml.

Read-only over gold.db. Raises on any 2026 row.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def is_holdout_clean(df: pd.DataFrame, holdout: str) -> bool:
    """Return True iff no row's trading_day >= holdout. Raise RuntimeError otherwise."""
    holdout_ts = pd.Timestamp(holdout)
    if (pd.to_datetime(df["trading_day"]) >= holdout_ts).any():
        raise RuntimeError(f"holdout row detected (>= {holdout})")
    return True


def null_coverage_mark(f: pd.Series, threshold: float) -> tuple[str, float]:
    """Return ('OK', drop_frac) if drop_frac<=threshold, else ('INVALID', drop_frac)."""
    drop_frac = float(f.isna().mean())
    return ("OK" if drop_frac <= threshold else "INVALID", drop_frac)


def power_floor_mark(n: int, min_n: int) -> str:
    return "OK" if n >= min_n else "UNDERPOWERED"


def compute_quintile_lift(df: pd.DataFrame, feature_col: str, outcome_col: str) -> dict[str, Any]:
    """Bin into quintiles by feature_col, return per-quintile mean outcome + monotonic flag."""
    quintiles = pd.qcut(df[feature_col], q=5, labels=False, duplicates="drop")
    means = df.groupby(quintiles)[outcome_col].mean().sort_index().tolist()
    if len(means) < 5:
        return {"q_means": means, "monotonic": False, "q1_mean_r": float("nan"),
                "q5_mean_r": float("nan"), "q5_minus_q1": 0.0}
    inc = all(means[i] <= means[i + 1] for i in range(4))
    dec = all(means[i] >= means[i + 1] for i in range(4))
    return {"q_means": means, "monotonic": inc or dec,
            "q1_mean_r": means[0], "q5_mean_r": means[4],
            "q5_minus_q1": means[4] - means[0]}


def bootstrap_sized_vs_flat_ci(
    df: pd.DataFrame, feature_col: str, outcome_col: str,
    weights: tuple[float, float, float, float, float],
    predicted_sign: str, B: int, seed: int, alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap mean(sized) - mean(flat). Weights applied per-quintile in predicted-sign direction."""
    quintiles = pd.qcut(df[feature_col], q=5, labels=False, duplicates="drop")
    pnl = df[outcome_col].to_numpy()
    qarr = quintiles.to_numpy()
    w_pos = np.array(weights)
    w_neg = w_pos[::-1]
    w = w_pos if predicted_sign == "+" else w_neg
    weight_per_trade = w[qarr]  # mean is 1.0 by construction since qcut bins are equi-count
    sized_pnl = pnl * weight_per_trade
    delta_obs = sized_pnl.mean() - pnl.mean()

    rng = np.random.default_rng(seed)
    n = len(pnl)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        deltas[b] = sized_pnl[idx].mean() - pnl[idx].mean()
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {"observed": float(delta_obs), "lo": lo, "hi": hi}


def apply_bh_fdr(pvals: list[float], q: float) -> list[bool]:
    """Benjamini-Hochberg FDR control. Returns list of pass/fail in original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, m + 1)) / m
    passes_sorted = ranked <= thresholds
    if passes_sorted.any():
        max_k = int(np.max(np.where(passes_sorted)[0]))
        passes_sorted = np.arange(m) <= max_k
    out = np.zeros(m, dtype=bool)
    out[order] = passes_sorted
    return out.tolist()


def check_forecast_stability(
    df: pd.DataFrame, feature_col: str, window: int, max_rel_var: float,
) -> str:
    """STABLE if rolling SD relative-variation <= max_rel_var, else UNSTABLE."""
    rolling_sd = df[feature_col].rolling(window=window, min_periods=window // 2).std().dropna()
    if len(rolling_sd) < 2:
        return "STABLE"  # too short to flag instability
    sd_med = rolling_sd.median()
    if sd_med <= 0:
        return "STABLE"
    rel_var = (rolling_sd.max() - rolling_sd.min()) / sd_med
    return "STABLE" if rel_var <= max_rel_var else "UNSTABLE"


def sign_match_split_half(df: pd.DataFrame, feature_col: str, outcome_col: str) -> bool:
    """Split by median trading_day; require Spearman rho sign match in both halves."""
    df_sorted = df.sort_values("trading_day").reset_index(drop=True)
    median_day = df_sorted["trading_day"].iloc[len(df_sorted) // 2]
    h1 = df_sorted[df_sorted["trading_day"] < median_day]
    h2 = df_sorted[df_sorted["trading_day"] >= median_day]
    if len(h1) < 30 or len(h2) < 30:
        return False
    r1, _ = stats.spearmanr(h1[feature_col], h1[outcome_col])
    r2, _ = stats.spearmanr(h2[feature_col], h2[outcome_col])
    return bool(np.sign(r1) == np.sign(r2) and r1 != 0 and r2 != 0)


def classify_cell(cell: dict[str, Any]) -> str:
    """Return PASS / FAIL / INVALID per spec §5.3 + §5.4a."""
    if cell.get("null_status") == "INVALID":
        return "INVALID"
    if cell.get("power_status") == "UNDERPOWERED":
        return "FAIL"
    required = [
        abs(cell.get("rho", 0.0)) >= 0.10,
        cell.get("bh_fdr_pass") is True,
        cell.get("monotonic") is True,
        abs(cell.get("q5_minus_q1", 0.0)) >= 0.20,
        cell.get("sized_flat_delta_lo", 0.0) > 0,
        cell.get("split_half_rho_match") is True,
        cell.get("split_half_delta_match") is True,
        cell.get("predicted_sign") == cell.get("realized_sign"),
    ]
    return "PASS" if all(required) else "FAIL"
```

- [ ] **Step 2: Run pure-function tests — confirm ALL pass**

```bash
pytest tests/test_research/test_audit_sizing_substrate_diagnostic.py -v
```

Expected: every test PASSES.

- [ ] **Step 3: Commit pure-function implementation**

```bash
git add research/audit_sizing_substrate_diagnostic.py
git commit -m "research(sizing): pure-function gates for sizing-substrate diagnostic (TDD green)"
```

---

## Task 7: Implement SQL loader and main() runner

**Files:**
- Modify: `research/audit_sizing_substrate_diagnostic.py` (append loader + main)

- [ ] **Step 1: Append the loader + main**

```python
# Append to research/audit_sizing_substrate_diagnostic.py

import json
import sys
from pathlib import Path

import duckdb
import yaml

from pipeline.paths import GOLD_DB_PATH


PREREG_PATH = Path("docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml")
RESULT_MD = Path("docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md")
RESULT_JSON = Path("docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json")


def load_prereg() -> dict:
    return yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))


def load_lane_tape(con: duckdb.DuckDBPyConnection, lane: dict, holdout: str) -> pd.DataFrame:
    """Load IS trade tape for one lane joined with daily_features. Raise on holdout row."""
    q = """
    SELECT o.trading_day, o.symbol, o.pnl_r, d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.rr_target = ?
      AND o.confirm_bars = ?
      AND o.entry_model = ?
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < DATE ?
    ORDER BY o.trading_day
    """
    df = con.execute(q, [
        lane["instrument"], lane["orb_label"], lane["orb_minutes"],
        lane["rr_target"], lane["confirm_bars"], lane["entry_model"],
        holdout,
    ]).fetchdf()
    is_holdout_clean(df, holdout=holdout)  # raises if leak
    return df


def resolve_substrate_column(lane: dict, form_id: str) -> str | None:
    """Map (lane.deployed_filter, form_id) to a daily_features column name OR a derivation key."""
    f = lane["deployed_filter"]
    sess = lane["orb_label"]
    if f == "ORB_G5":
        if form_id == "raw":         return f"orb_{sess}_size"
        if form_id == "vol_norm":    return f"orb_{sess}_size__div__atr_20_pct"
        if form_id == "rank_252d":   return f"orb_{sess}_size__rank252"
    if f == "ATR_P50":
        if form_id == "raw":         return "atr_20_pct"
        if form_id == "vol_norm":    return "atr_20_pct"  # already vol-normalized
        if form_id == "rank_252d":   return "atr_20_pct__rank252"
    if f == "COST_LT12":
        if form_id == "raw":         return "_cost_ratio"  # derived in derive_features()
        if form_id == "vol_norm":    return "_cost_ratio__div__atr_20_pct"
        if form_id == "rank_252d":   return "_cost_ratio__rank252"
    return None


def derive_features(df: pd.DataFrame, lane: dict) -> pd.DataFrame:
    """Add derived columns (vol-norm, 252d rank, cost ratio) to the lane tape."""
    sess = lane["orb_label"]
    out = df.copy()
    # Tier-A vol-normalized + rank forms (ORB_G5 substrate)
    if lane["deployed_filter"] == "ORB_G5":
        size_col = f"orb_{sess}_size"
        if size_col in out.columns:
            out[f"{size_col}__div__atr_20_pct"] = out[size_col] / out["atr_20_pct"].replace(0, np.nan)
            out[f"{size_col}__rank252"] = out[size_col].rolling(252, min_periods=63).rank(pct=True)
    # Tier-A vol-normalized + rank forms (ATR_P50 substrate is already atr_20_pct, no vol-norm needed)
    if lane["deployed_filter"] == "ATR_P50":
        out["atr_20_pct__rank252"] = out["atr_20_pct"].rolling(252, min_periods=63).rank(pct=True)
    # Tier-A COST_LT12 substrate — cost ratio per trade
    if lane["deployed_filter"] == "COST_LT12":
        from pipeline.cost_model import get_session_cost_spec
        cs = get_session_cost_spec(lane["instrument"], lane["orb_label"])
        cost_pts = cs.total_cost_points
        # Cost ratio: cost in points divided by per-trade ORB-distance in points (proxy for R distance)
        avg_orb_pts_col = f"orb_{sess}_size"
        out["_cost_ratio"] = cost_pts / out[avg_orb_pts_col].replace(0, np.nan)
        out["_cost_ratio__div__atr_20_pct"] = out["_cost_ratio"] / out["atr_20_pct"].replace(0, np.nan)
        out["_cost_ratio__rank252"] = out["_cost_ratio"].rolling(252, min_periods=63).rank(pct=True)
    return out


def run_diagnostic() -> dict:
    """Top-level. Returns the structured result dict."""
    prereg = load_prereg()
    holdout = prereg["metadata"]["holdout_date"]
    rho_min = prereg["metadata"]["rho_min"]
    n_min = prereg["metadata"]["power_floor_N"]
    null_max = prereg["metadata"]["null_coverage_max"]
    sd_max = prereg["metadata"]["stability_sd_variation_max"]
    seed = prereg["metadata"]["bootstrap_seed"]
    B = prereg["metadata"]["bootstrap_B"]
    weights = (0.6, 0.8, 1.0, 1.2, 1.4)

    cells: list[dict] = []
    pvals: list[float] = []

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    for lane in prereg["lanes"]:
        df = load_lane_tape(con, lane, holdout=holdout)
        df = derive_features(df, lane)

        # --- Tier-A: 3 forms of the deployed-filter substrate ---
        for form in prereg["features"]["tier_a"]:
            col = resolve_substrate_column(lane, form["form_id"])
            pred_sign = prereg["ex_ante_directions_tier_a"][f"{lane['deployed_filter']}_{form['form_id']}"]
            cells.append(_compute_cell(df, lane, col, pred_sign, "tier_a", form["form_id"],
                                        rho_min, n_min, null_max, sd_max, seed, B, weights))
        # --- Tier-B: 5 orthogonal features ---
        for feat in prereg["features"]["tier_b"]:
            col = feat.get("column") or feat["column_template"].format(ORB_LABEL=lane["orb_label"])
            pred_sign = feat["ex_ante_direction"]
            cells.append(_compute_cell(df, lane, col, pred_sign, "tier_b", feat["feature_id"],
                                        rho_min, n_min, null_max, sd_max, seed, B, weights))

    # BH-FDR over the full K=48 family
    pvals = [c["rho_p"] if c.get("rho_p") is not None else 1.0 for c in cells]
    bh = apply_bh_fdr(pvals, q=prereg["metadata"]["bh_fdr_q"])
    for c, p in zip(cells, bh):
        c["bh_fdr_pass"] = bool(p)
        c["status"] = classify_cell(c)

    # Lane-level + global verdict
    by_lane: dict[str, list[dict]] = {}
    for c in cells:
        by_lane.setdefault(c["lane_id"], []).append(c)
    lanes_with_substrate = [lid for lid, lc in by_lane.items() if any(x["status"] == "PASS" for x in lc)]
    n_passing_lanes = len(lanes_with_substrate)
    if n_passing_lanes >= 3:
        verdict = "SUBSTRATE_CONFIRMED"
    elif n_passing_lanes in (1, 2):
        verdict = "SUBSTRATE_WEAK"
    else:
        verdict = "THESIS_KILLED"
    # Tier-level inconclusive guard
    tier_a_cells = [c for c in cells if c["tier"] == "tier_a"]
    tier_b_cells = [c for c in cells if c["tier"] == "tier_b"]
    inv_a = sum(1 for c in tier_a_cells if c["status"] in ("INVALID",) or c.get("power_status") == "UNDERPOWERED")
    inv_b = sum(1 for c in tier_b_cells if c["status"] in ("INVALID",) or c.get("power_status") == "UNDERPOWERED")
    if inv_a / max(1, len(tier_a_cells)) >= 0.5 or inv_b / max(1, len(tier_b_cells)) >= 0.5:
        verdict = "INCONCLUSIVE"

    db_sha = con.execute("SELECT md5(string_agg(table_name, ',')) FROM information_schema.tables").fetchone()[0]
    git_sha = _git_head_sha()

    result = {
        "design_doc": "docs/plans/2026-04-27-sizing-substrate-diagnostic-design.md",
        "prereg": str(PREREG_PATH),
        "git_head_sha": git_sha,
        "db_sha_proxy": db_sha,
        "bootstrap_seed": seed,
        "bootstrap_B": B,
        "K": len(cells),
        "verdict": verdict,
        "lanes_with_substrate": lanes_with_substrate,
        "cells": cells,
    }
    return result


def _compute_cell(df, lane, col, pred_sign, tier, form_id,
                  rho_min, n_min, null_max, sd_max, seed, B, weights) -> dict:
    cell = {
        "lane_id": lane["id"], "tier": tier, "form_or_feature": form_id, "column": col,
        "predicted_sign": pred_sign,
    }
    if col is None or col not in df.columns:
        cell.update({"null_status": "INVALID", "power_status": "n/a", "rho_p": 1.0,
                     "rho": 0.0, "monotonic": False, "q5_minus_q1": 0.0,
                     "sized_flat_delta_lo": 0.0, "sized_flat_delta_hi": 0.0,
                     "split_half_rho_match": False, "split_half_delta_match": False,
                     "stability_status": "n/a", "realized_sign": "?", "n": 0,
                     "drop_frac": 1.0, "note": "column missing"})
        return cell

    # NULL coverage
    null_status, drop_frac = null_coverage_mark(df[col], threshold=null_max)
    cell["null_status"] = null_status
    cell["drop_frac"] = drop_frac
    df2 = df.dropna(subset=[col, "pnl_r"]).copy()
    cell["n"] = len(df2)
    cell["power_status"] = power_floor_mark(len(df2), min_n=n_min)
    if cell["null_status"] == "INVALID" or cell["power_status"] == "UNDERPOWERED":
        cell["rho_p"] = 1.0
        cell.update({"rho": 0.0, "monotonic": False, "q5_minus_q1": 0.0,
                     "sized_flat_delta_lo": 0.0, "sized_flat_delta_hi": 0.0,
                     "split_half_rho_match": False, "split_half_delta_match": False,
                     "stability_status": "n/a", "realized_sign": "?"})
        return cell

    # Spearman rho
    rho, p = stats.spearmanr(df2[col], df2["pnl_r"])
    cell["rho"] = float(rho)
    cell["rho_p"] = float(p)
    cell["realized_sign"] = "+" if rho >= 0 else "-"
    # Quintile lift
    ql = compute_quintile_lift(df2, feature_col=col, outcome_col="pnl_r")
    cell.update({"q1_mean_r": ql["q1_mean_r"], "q5_mean_r": ql["q5_mean_r"],
                 "q5_minus_q1": ql["q5_minus_q1"], "monotonic": ql["monotonic"]})
    # Sized vs flat
    ci = bootstrap_sized_vs_flat_ci(df2, feature_col=col, outcome_col="pnl_r",
                                    weights=weights, predicted_sign=pred_sign, B=B, seed=seed)
    cell.update({"sized_flat_delta_obs": ci["observed"],
                 "sized_flat_delta_lo": ci["lo"], "sized_flat_delta_hi": ci["hi"]})
    # Split-half
    cell["split_half_rho_match"] = sign_match_split_half(df2, feature_col=col, outcome_col="pnl_r")
    df_sorted = df2.sort_values("trading_day").reset_index(drop=True)
    median_day = df_sorted["trading_day"].iloc[len(df_sorted) // 2]
    h1 = df_sorted[df_sorted["trading_day"] < median_day]
    h2 = df_sorted[df_sorted["trading_day"] >= median_day]
    if len(h1) > 30 and len(h2) > 30:
        ci1 = bootstrap_sized_vs_flat_ci(h1, col, "pnl_r", weights, pred_sign, B=1000, seed=seed)
        ci2 = bootstrap_sized_vs_flat_ci(h2, col, "pnl_r", weights, pred_sign, B=1000, seed=seed)
        cell["split_half_delta_match"] = bool(np.sign(ci1["observed"]) == np.sign(ci2["observed"])
                                              and ci1["observed"] != 0 and ci2["observed"] != 0)
    else:
        cell["split_half_delta_match"] = False
    # Forecast stability
    cell["stability_status"] = check_forecast_stability(df2, feature_col=col, window=252, max_rel_var=sd_max)
    return cell


def _git_head_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def render_markdown(result: dict) -> str:
    lines: list[str] = []
    lines.append("# Sizing-Substrate Diagnostic — Result")
    lines.append("")
    lines.append(f"- Design doc: `{result['design_doc']}`")
    lines.append(f"- Pre-reg: `{result['prereg']}`")
    lines.append(f"- Git HEAD: `{result['git_head_sha']}`")
    lines.append(f"- DB schema fingerprint: `{result['db_sha_proxy']}`")
    lines.append(f"- Bootstrap seed: {result['bootstrap_seed']}; B={result['bootstrap_B']}")
    lines.append(f"- K = {result['K']}")
    lines.append(f"- **VERDICT: {result['verdict']}**")
    lines.append(f"- Lanes with substrate: {result['lanes_with_substrate']}")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| lane | tier | feature/form | n | rho | p | bh-fdr | Q5-Q1 R | mono | delta CI | split | stable | pred | real | status |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in result["cells"]:
        ci_str = f"[{c.get('sized_flat_delta_lo', 0):+.3f}, {c.get('sized_flat_delta_hi', 0):+.3f}]"
        split = "Y" if c.get("split_half_rho_match") and c.get("split_half_delta_match") else "N"
        lines.append(f"| {c['lane_id']} | {c['tier']} | {c.get('form_or_feature')} | "
                     f"{c.get('n', 0)} | {c.get('rho', 0):+.3f} | {c.get('rho_p', 1):.4f} | "
                     f"{'Y' if c.get('bh_fdr_pass') else 'N'} | {c.get('q5_minus_q1', 0):+.3f} | "
                     f"{'Y' if c.get('monotonic') else 'N'} | {ci_str} | {split} | "
                     f"{c.get('stability_status', '?')[:1]} | {c.get('predicted_sign', '?')} | "
                     f"{c.get('realized_sign', '?')} | **{c.get('status', '?')}** |")
    return "\n".join(lines) + "\n"


def main() -> int:
    result = run_diagnostic()
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    RESULT_MD.write_text(render_markdown(result), encoding="utf-8")
    print(f"VERDICT: {result['verdict']}")
    print(f"Lanes with substrate: {result['lanes_with_substrate']}")
    print(f"Wrote {RESULT_MD}")
    print(f"Wrote {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Re-run pure-function tests — confirm still passing**

```bash
pytest tests/test_research/test_audit_sizing_substrate_diagnostic.py -v
```

Expected: every test still PASSES (loader added; pure functions untouched).

- [ ] **Step 3: Commit loader + main**

```bash
git add research/audit_sizing_substrate_diagnostic.py
git commit -m "research(sizing): SQL loader + main runner for sizing-substrate diagnostic"
```

---

## Task 8: Smoke test — script imports clean

**Files:**
- No file changes; smoke test only.

- [ ] **Step 1: Verify clean import + dry parse**

```bash
python -c "import research.audit_sizing_substrate_diagnostic as m; print('imports OK'); print('main:', callable(m.main))"
python -c "import research.audit_sizing_substrate_diagnostic as m; p = m.load_prereg(); print('prereg loaded; lanes=', len(p['lanes']))"
```

Expected: both print success lines. If `load_prereg` fails, the YAML from Task 4 is malformed — fix YAML first.

- [ ] **Step 2: Verify holdout-raise on synthetic post-2026 row**

```bash
python -c "
import pandas as pd
from research.audit_sizing_substrate_diagnostic import is_holdout_clean
df = pd.DataFrame({'trading_day': pd.to_datetime(['2026-04-15'])})
try:
    is_holdout_clean(df, '2026-01-01')
    print('FAIL — should have raised')
except RuntimeError as e:
    print('OK — raised as expected:', e)
"
```

Expected: `OK — raised as expected: holdout row detected ...`

---

## Task 9: Run the diagnostic + write artifacts

**Files:**
- Will create: `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`
- Will create: `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json`

- [ ] **Step 1: Run the diagnostic**

```bash
cd C:\Users\joshd\canompx3
python research/audit_sizing_substrate_diagnostic.py
```

Expected stdout:
```
VERDICT: <SUBSTRATE_CONFIRMED|SUBSTRATE_WEAK|THESIS_KILLED|INCONCLUSIVE>
Lanes with substrate: [...]
Wrote docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md
Wrote docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json
```

If RuntimeError on holdout row: investigate immediately — should be impossible because SQL filters `trading_day < '2026-01-01'`. If raised, the SQL is broken.

- [ ] **Step 2: Spot-check the JSON for sanity**

```bash
python -c "
import json
r = json.load(open('docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json'))
print('K:', r['K'])
print('verdict:', r['verdict'])
n_pass = sum(1 for c in r['cells'] if c['status'] == 'PASS')
n_invalid = sum(1 for c in r['cells'] if c['status'] == 'INVALID')
n_underpow = sum(1 for c in r['cells'] if c.get('power_status') == 'UNDERPOWERED')
print(f'PASS={n_pass}  INVALID={n_invalid}  UNDERPOWERED={n_underpow}')
"
```

Expected: K=48, verdict prints one of the four allowed values, sums of statuses are sensible.

- [ ] **Step 3: Spot-check the markdown opens cleanly**

```bash
head -30 docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md
```

Expected: header with verdict, design doc, prereg link, git/db SHAs, table header.

- [ ] **Step 4: Run drift checks**

```bash
python pipeline/check_drift.py
```

Expected: PASSED (no production code touched).

- [ ] **Step 5: Run pure-function tests one more time (regression)**

```bash
pytest tests/test_research/test_audit_sizing_substrate_diagnostic.py -v
```

Expected: every test PASSES.

- [ ] **Step 6: Commit results**

```bash
git add docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json
git commit -m "research(sizing): Stage-1 sizing-substrate diagnostic — verdict=<paste>"
```

Replace `<paste>` with the actual VERDICT from Step 1.

---

## Task 10: Self-review and decision routing

**Files:**
- May modify `memory/MEMORY.md` to add a "Recent findings" entry
- May add NO-GO entry to `docs/STRATEGY_BLUEPRINT.md` if THESIS_KILLED

- [ ] **Step 1: Self-review the result**

Re-read `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`. Sanity check:
- Verdict is one of the four allowed values.
- All 48 cells have a status.
- INVALID/UNDERPOWERED counts match what Task 1 precheck implied.
- PASS cells (if any) have ρ, monotonicity, sized-vs-flat CI > 0, BH-FDR Y, split-half Y, prediction-confirmed.

- [ ] **Step 2: Add memory entry**

Append to `memory/MEMORY.md` under "## Recent findings":

```
- **Stage-1 sizing-substrate diagnostic (2026-04-27, K=48):** <verdict>. Lanes with substrate: <list>. Result: docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md. Pre-reg: docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml. Decision: <"Stage 2 sizing model pre-reg" | "park" | "NO-GO entry, sizing thesis dead">.
```

- [ ] **Step 3: Branch on verdict**

- **SUBSTRATE_CONFIRMED:** Brainstorm Stage-2 sizing-model pre-reg in a new session (Carver Ch. 7 forecast scaling on the passing cells). This implementation plan ends.
- **SUBSTRATE_WEAK:** Document and park. No Stage-2.
- **THESIS_KILLED:** Add NO-GO entry to `docs/STRATEGY_BLUEPRINT.md` §5 NO-GO Registry. Reopen requires new mechanism citation.
- **INCONCLUSIVE:** Diagnose which tier was unusable (NULL-heavy or N-low). Do NOT re-run blindly — investigate and amend in a new pre-reg if substrate detection is salvageable.

- [ ] **Step 4: Final commit**

```bash
git add memory/MEMORY.md docs/STRATEGY_BLUEPRINT.md  # only the files actually touched
git commit -m "research(sizing): record Stage-1 verdict + decision routing"
```

---

## Self-review of this plan

**Spec coverage:**
- §3 ground truth → Tasks 1, 2 (precheck + filter formula verification)
- §4 hypothesis & decision rule → Task 4 YAML `decision_rule` + Task 7 `run_diagnostic`
- §5.1 cell definition → Task 4 YAML `lanes` + `features` + Task 7 `resolve_substrate_column` + `derive_features`
- §5.2 step 1 holdout-raise → Task 5 test `test_holdout_clean_raises_when_any_row_in_holdout` + Task 6 `is_holdout_clean` + Task 7 `load_lane_tape` + Task 8 step 2
- §5.2 step 3 NULL guard → Task 5 + Task 6 `null_coverage_mark` + Task 7 cell loop
- §5.2 step 4 power floor → Task 5 + Task 6 `power_floor_mark` + Task 7 cell loop
- §5.2 step 5 ex-ante prediction → Task 4 YAML `ex_ante_directions_tier_a` + `ex_ante_direction` per Tier-B + Task 7 cell loop uses `predicted_sign`
- §5.2 step 6 Spearman ρ → Task 7 cell loop
- §5.2 step 7 quintile lift → Task 5 + Task 6 `compute_quintile_lift`
- §5.2 step 9 sized-vs-flat (linear-rank, dollar-vol-matched, ex-ante direction) → Task 5 + Task 6 `bootstrap_sized_vs_flat_ci`
- §5.2 step 10 forecast stability → Task 5 + Task 6 `check_forecast_stability`
- §5.2 step 11 bootstrap seed=42, B=10000 → Task 6 + Task 7
- §5.2 step 12 split-half stability → Task 5 + Task 6 `sign_match_split_half` + Task 7 delta-sign-match
- §5.3 cell pass criteria → Task 6 `classify_cell` + tests
- §5.4a tier guardrail → Task 7 `run_diagnostic` `inv_a/inv_b` check
- §6 boundary discipline → Task 4 `boundary` block + Task 9 step 4 drift check
- §7 outputs → Task 7 `render_markdown` + Task 9 step 6 commit
- §8 success criteria → Task 9 step 5 regression + Task 10 step 1 self-review
- §10 risks → Task 1 (substrate definition), Task 5/6 (NULL/power/stability/seed), Task 9 step 4 (drift)

**Placeholder scan:** No "TBD/TODO/etc" remaining. All code is concrete. The pre-reg YAML has `[PASTE LITERAL EXTRACT FROM TASK 3 STEP N]` placeholders that Task 4 Step 2 explicitly replaces and Step 4 verifies are gone.

**Type consistency:**
- `is_holdout_clean(df, holdout: str) -> bool` — same signature in Task 5, 6, 7, 8.
- `null_coverage_mark(f: pd.Series, threshold: float) -> tuple[str, float]` — same.
- `power_floor_mark(n: int, min_n: int) -> str` — same.
- `compute_quintile_lift(df, feature_col, outcome_col)` returns dict with keys `q_means, monotonic, q1_mean_r, q5_mean_r, q5_minus_q1` — used identically in tests and in the loader.
- `bootstrap_sized_vs_flat_ci(df, feature_col, outcome_col, weights, predicted_sign, B, seed)` returns `{observed, lo, hi}` — used identically.
- `apply_bh_fdr(pvals, q) -> list[bool]` — same.
- `check_forecast_stability(df, feature_col, window, max_rel_var)` returns `STABLE`/`UNSTABLE` — same.
- `sign_match_split_half(df, feature_col, outcome_col) -> bool` — same.
- `classify_cell(cell: dict) -> str` returns `PASS`/`FAIL`/`INVALID` — same.

**Frequent commits:** Tasks 1, 2 (precheck/verify, no commit), 4 step 5 (YAML lock), 4 step 6 (commit_sha backfill), 5 step 3 (failing tests), 6 step 3 (pure functions), 7 step 3 (loader+main), 9 step 6 (results), 10 step 4 (memory + decision). 7 commits across the work.

**TDD:** Task 5 writes failing tests; Task 6 implements to green; Task 7 adds I/O without breaking the green; Task 9 step 5 re-runs tests as regression. No code lands without a failing test that drives it (except the I/O loader — which is tested by Task 8 smoke test + Task 9 step 1 actual run).

**DRY:** Pure functions in Task 6 are reused by `run_diagnostic` in Task 7 for both Tier-A and Tier-B cells. No duplication of bootstrap/null-guard/power logic.

**YAGNI:** No portfolio simulator, no per-regime decomposition, no block-bootstrap, no AFML sigmoid. All deferred to Stage 2 per spec §9.

---

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-04-27-sizing-substrate-diagnostic-implementation-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good for institutional rigor — each task gets a clean context and the literature-extraction work in Task 3 benefits from a focused subagent.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Faster for the typing-heavy YAML/script tasks but the main session loses context-budget doing PDF extracts.

**Which approach?**
