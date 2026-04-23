# Hypothesis Registry Template

**Purpose:** Every pre-registered discovery run must produce a hypothesis registry file at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml` BEFORE any code runs. This file is the commitment device per `pre_registered_criteria.md` § Criterion 1.

**Why this exists:** Per `literature/bailey_et_al_2013_pseudo_mathematics.md` § "If only 5 years of data are available, no more than 45 independent model configurations should be tried, or we are almost guaranteed to produce strategies with an annualized Sharpe ratio IS of 1, but an expected Sharpe ratio OOS of zero." Pre-registration limits the effective search space and keeps MinBTL achievable.

**Authority:** Discovery code reads this file and restricts itself to the pre-registered specifications. A drift check verifies validated_setups entries match a registered hypothesis.

**Operator contract:** The human-facing workflow is natural language. Agents use
this file and the prereg front door internally; users should not need to
remember command syntax. See `docs/institutional/research_pipeline_contract.md`.

---

## Template format

File path: `docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml`

```yaml
# Hypothesis Registry — {date} {slug}
#
# Locked: {ISO timestamp}
# Author: {session id or user}
# Committing commit hash: (filled in by pre-commit hook)
# Data horizon: {clean_years}yr clean + {proxy_years}yr proxy
# Intended discovery holdout: {date}

metadata:
  name: "{short slug}"
  purpose: "{one line purpose}"
  date_locked: "2026-04-07T00:00:00+10:00"
  data_horizon_years_clean: 2.2
  data_horizon_years_proxy: 16.0
  holdout_date: "2026-01-01"
  total_expected_trials: 0   # sum across all hypotheses; must be ≤ 300 (clean) or 2000 (proxy)
  research_question_type: "standalone_edge"   # or "conditional_role"

  # ---- Phase 4 Stage 4.1b: MinBTL proxy-mode opt-in (OPTIONAL) ----
  # By default, Criterion 2's clean-data bound of 300 trials applies.
  # To exceed 300 (up to the proxy-data bound of 2000), set BOTH fields:
  #
  #   data_source_mode: "proxy"
  #   data_source_disclosure: "<explicit text describing the proxy data>"
  #
  # Both are REQUIRED together when total_expected_trials > 300. The
  # canonical enforcement point is
  # ``trading_app.hypothesis_loader.enforce_minbtl_bound`` which is called
  # by both ``strategy_validator._check_criterion_2_minbtl`` and
  # ``strategy_discovery.run_discovery`` in Phase 4 enforcement mode.
  #
  # The disclosure string must be non-empty and should name the specific
  # proxy data source (e.g., "NQ parent futures pre-2024-02-05 as proxy
  # for MNQ micro" for pre-micro-launch MNQ research). Criterion 2's
  # locked text in pre_registered_criteria.md requires "explicit
  # data-source disclosure" — this field mechanizes that requirement.
  #
  # Example (for a clean-data hypothesis file — both fields omitted):
  #   (no data_source_mode or data_source_disclosure set)
  #   total_expected_trials: 60
  #
  # Example (for a proxy-extended hypothesis file):
  #   data_source_mode: "proxy"
  #   data_source_disclosure: "NQ parent futures 2009-2024-02-05 as proxy for MNQ micro"
  #   total_expected_trials: 1500

# Optional operator routing block.
# Use this when the prereg is a bounded study with a dedicated runner rather
# than a broad grid-discovery write into experimental_strategies.
#
# execution:
#   mode: "bounded_runner"            # optional; inferred for conditional_role
#   entrypoint: "research/my_study.py"
#   default_args:
#     - "--output"
#     - "docs/audit/results/YYYY-MM-DD-my-study.md"

hypotheses:
  - id: 1
    name: "Overnight range predicts EUROPE_FLOW follow-through"
    theory_citation: "Crabel 1990 opening range; Bessembinder 2018 volatility clustering; academic literature on institutional flow at London open"
    economic_basis: |
      Asian-session overnight range reflects institutional positioning from
      Tokyo/Singapore desks. When that range is large, London-open participants
      are more likely to continue the directional move. This is a well-known
      microstructural pattern in FX and equity index futures.
    filter:
      type: OVNRNG
      column: overnight_range
      thresholds: [50, 75, 100, 125, 150]
    scope:
      instruments: [MNQ]
      sessions: [EUROPE_FLOW]
      rr_targets: [1.0, 1.5, 2.0, 3.0]
      entry_models: [E2]
      confirm_bars: [1]
      stop_multipliers: [1.0]
      orb_minutes: [5]
    expected_trial_count: 20
    kill_criteria:
      - "BH FDR q=0.05 fails across all 20 trials"
      - "DSR < 0.95 for best trial"
      - "2026 OOS ExpR < 0.40 * IS ExpR"
      - "Pre-2020 era shows ExpR < -0.05 with N >= 50"

  - id: 2
    name: "ORB size gate predicts MGC CME_REOPEN follow-through"
    theory_citation: "Crabel 1990 opening range; Toby Crabel 'Day Trading with Short Term Price Patterns and Opening Range Breakout'"
    economic_basis: |
      Larger ORB sizes in MGC at CME_REOPEN indicate stronger overnight
      positioning carried into the session. MGC is liquid enough post-2024
      (real micro data) for this to be tested cleanly.
    filter:
      type: ORB_G
      column: "orb_CME_REOPEN_size"
      thresholds: [4, 6, 8, 10]
    scope:
      instruments: [MGC]
      sessions: [CME_REOPEN]
      rr_targets: [1.5, 2.0, 2.5, 3.0]
      entry_models: [E2]
      confirm_bars: [1]
      stop_multipliers: [1.0]
      orb_minutes: [5]
    expected_trial_count: 16
    kill_criteria:
      - "BH FDR q=0.05 fails across all 16 trials"
      - "DSR < 0.95 for best trial"
      - "Insufficient real MGC sample (N < 100 on MICRO-era data only)"
      - "Data era compatibility (Criterion 10) fails — volume filter on non-micro data"

  # Add more hypotheses up to the total_expected_trials budget
  # Each must have:
  #  - id, name, theory_citation (REQUIRED), economic_basis (REQUIRED)
  #  - filter (type, column, thresholds)
  #  - scope (instruments, sessions, rr_targets, entry_models, confirm_bars, stop_multipliers, orb_minutes)
  #  - expected_trial_count (sum across all hypotheses ≤ budget)
  #  - kill_criteria (must be pre-registered, not post-hoc)
  #
  # If metadata.research_question_type == "conditional_role", each hypothesis
  # must ALSO include:
  #
  #   role:
  #     kind: "filter" | "conditioner" | "allocator" | "confluence" | "execution" | "standalone"
  #     parent: "<exact parent population>"
  #     comparator: "<exact comparison being judged>"
  #     primary_metric: "<policy_ev_per_opportunity_r / selected_trade_mean_r / portfolio_ev_r / ...>"
  #     promotion_target: "<shadow_only / deployable_filter / portfolio_only / ...>"
  #
  # This is enforced by trading_app.hypothesis_loader for role-aware studies.

total_hypothesis_count: 2
total_expected_trials: 36   # must equal sum of expected_trial_count above
budget_check:
  max_allowed_clean: 300
  max_allowed_proxy: 2000
  status: "under budget"

# Pre-registration checklist
#
# [ ] Every hypothesis has a theory citation (not just "ORB is a thing")
# [ ] Every hypothesis has an economic_basis explaining WHY
# [ ] Sum of expected_trial_count is within budget
# [ ] Kill criteria are pre-registered (not post-hoc)
# [ ] No hypothesis uses volume filters on parent-proxy data (C10)
# [ ] Holdout date is set and will not be moved after results are seen
# [ ] File is committed to git BEFORE discovery runs
```

---

## How to use

### Step 1 — Write the hypothesis file

Copy the template above, fill in hypotheses with real theory citations and specifications. Do not skip the theory citation step. A hypothesis without a theory citation is by definition a data-mined pattern and cannot be accepted under `pre_registered_criteria.md` § Criterion 1.

If the study is about a conditional state variable rather than a new standalone lane, set:

```yaml
metadata:
  research_question_type: "conditional_role"
```

Then add a `role:` block to every hypothesis. This prevents a filter or
allocator question from being judged as if it were a standalone strategy.

### Step 2 — Compute MinBTL sanity check

```
MinBTL = 2·Ln[total_expected_trials] / 1.0²
```

If MinBTL > available data years, reduce hypotheses until it fits. The budget gates (300 clean / 2000 proxy) are hard caps.

### Step 3 — Commit to git

```
git add docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml
git commit -m "hypotheses: pre-register <slug> for YYYY-MM-DD discovery run"
```

### Step 4 — Run discovery with the file

Agent/operator note: this is an internal routing step. If a human asks to run
the discovery, the agent should execute the correct branch and report the
result. Do not make the human memorize this command.

```bash
scripts/infra/prereg-loop.sh \
  --hypothesis-file docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml \
  --execute
```

For `standalone_edge` preregs this routes to `trading_app.strategy_discovery`
with the prereg's `holdout_date` and `--hypothesis-file` wired correctly.

For `conditional_role` preregs, the same front door will inspect and report the
route, but execution requires either:

```yaml
execution:
  entrypoint: "research/my_bounded_runner.py"
```

or a manual override:

```bash
scripts/infra/prereg-loop.sh \
  --hypothesis-file docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml \
  --runner research/my_bounded_runner.py \
  --execute
```

### Step 5 — Apply criteria from `pre_registered_criteria.md`

After discovery completes, each candidate is evaluated against all 12 criteria. No candidate is "validated" until all 12 pass.

Validation does not require live routing or `paper_trades`. Those are deployment
and operations gates. A candidate can be validated research inventory and still
not be selected for the live book.

### Step 6 — Write a post-mortem

For each hypothesis that PASSED, record in the post-mortem file why it survived.
For each hypothesis that FAILED, record exactly which kill criterion triggered.
Post-mortem file: `docs/audit/hypotheses/YYYY-MM-DD-<slug>-postmortem.md`.

This post-mortem is permanent — it documents what we tested and what we rejected, so we don't accidentally re-test the same dead hypothesis.

---

## What NOT to do

1. **Do not use this as a retroactive label** on already-run discoveries. Pre-registration must be BEFORE the data is seen.
2. **Do not loosen thresholds** in kill_criteria after seeing results.
3. **Do not add hypotheses post-hoc** to "catch" a strategy that almost passed. Run a new pre-registered file if you want a new hypothesis.
4. **Do not cite theory from training memory** — the theory_citation must reference specific papers or books that are either in `resources/` or verifiable online.
5. **Do not exceed the trial budget** — if the budget is too small, reduce hypotheses, don't inflate the budget without amending `pre_registered_criteria.md` with written justification.

---

## Example — a bad hypothesis file (do not write like this)

```yaml
hypotheses:
  - id: 1
    name: "Try lots of filters on all sessions"
    theory_citation: "general intuition"  # BAD — no citation
    economic_basis: "we want to find good strategies"  # BAD — not a theory
    filter:
      type: ANY
      column: "*"  # BAD — not specific
    scope:
      instruments: [MNQ, MES, MGC, M2K, MCL, SIL, M6E, MBT]  # BAD — includes dead instruments
      sessions: "ALL"  # BAD — brute force
      rr_targets: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]  # BAD — too many
      ...
    expected_trial_count: 25000  # BAD — exceeds budget
```

This is exactly the brute-force pattern we are moving away from. Any file like this should be rejected at Step 3 before commit.

---

## Example — a good hypothesis file (write like this)

See the main template section above for the good pattern. Key features of a good file:

- Small number of hypotheses (typically 3-10)
- Each with a specific theory citation (paper or book chapter)
- Each with a clear economic basis explanation
- Trial count stays within budget (≤ 300 clean / ≤ 2000 proxy)
- Kill criteria are concrete and pre-registered
- Holdout date is committed to

### Example — conditional role study

```yaml
metadata:
  name: "pr48_conditional_role_implementation_v1"
  holdout_date: "2026-01-01"
  total_expected_trials: 9
  research_question_type: "conditional_role"

hypotheses:
  - id: 1
    name: "MES participation acts as a filter or allocator"
    theory_citation: "docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md"
    economic_basis: "Participation state should improve selection or sizing quality."
    role:
      kind: "filter"
      parent: "All canonical MES O5 E2 CB1 RR1.5 trades"
      comparator: "Parent vs Q4+Q5 vs Q5 vs continuous quintile sizer"
      primary_metric: "policy_ev_per_opportunity_r"
      promotion_target: "deployable_filter_or_sizer"
    filter:
      type: "REL_VOL_STATE"
      column: "rel_vol_{session}"
      thresholds: ["q4_plus_q5", "q5_only", "continuous"]
    scope:
      instruments: [MES]
      sessions: [ALL_CANONICAL_5M]
      rr_targets: [1.5]
      entry_models: [E2]
      confirm_bars: [1]
      stop_multipliers: [1.0]
    expected_trial_count: 3
    kill_criteria:
      - "No role improves policy EV versus parent"
```

---

## Related files

- `README.md` — master index
- `finite_data_framework.md` — the reasoning behind pre-registration
- `pre_registered_criteria.md` — the 12 locked criteria hypotheses must meet
- `literature/bailey_et_al_2013_pseudo_mathematics.md` — MinBTL justification
- `literature/lopez_de_prado_2020_ml_for_asset_managers.md` — theory-first principle (pending)
- `conditional-edge-framework.md` — when the right question is filter / allocator / confluence instead of standalone
