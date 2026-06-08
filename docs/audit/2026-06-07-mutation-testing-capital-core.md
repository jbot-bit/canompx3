# Mutation Testing — Capital-Critical Core (2026-06-07)

**Goal:** grade the *tests*, not the code. Coverage proves a line *ran*; mutation
testing proves the tests would *notice* if that line were wrong. A surviving
mutant = an effectively-untested line.

## Tooling decision (grounded in execution, not opinion)

| | mutmut 3.6 | **cosmic-ray 8.4.6 (chosen)** |
|---|---|---|
| Native Windows | **HARD-BLOCKED** — every subcommand (`run`/`results`/`version`) prints only *"To run mutmut on Windows, please use the WSL"* (boxed/mutmut#397) and does nothing | **Runs natively** (Windows + Linux + CI) |
| Python | needs the WSL toolchain (3.12 here) | this repo's canonical **3.11.9** |
| Portability | WSL/.venv-wsl/`/mnt/c` island — no clean checkout or GitHub-Actions runner reproduces it | config is 100% repo-relative; a fresh clone runs it unchanged |
| `check_drift.py` collision | writes a `mutants/` source-copy tree → globbed by `rglob("*.py")` at 20+ sites | **mutates in-place** → no `mutants/` tree → no collision |

The operator's hard constraint was "must work without our PC / repo / project"
(portable across clean checkout, OS, and CI). mutmut fails that constraint on
this Windows runner outright; cosmic-ray satisfies it. Decision made *after*
running both: `mutmut version` → WSL-refusal; `cosmic-ray --help` → works here.

## How to run (one command per module)

```bash
uv run bash scripts/tools/run_mutation.sh <module_src> "<test_command>" [timeout_s]
# e.g.
uv run bash scripts/tools/run_mutation.sh pipeline/cost_model.py \
  "python -m pytest tests/test_pipeline/test_cost_model.py -q" 30
```

The runner writes an ephemeral `.mutation/<slug>.toml` + `.sqlite` (gitignored,
trap-cleaned on exit), runs `baseline → init → exec → cr-rate → cr-report`.

## Scoring bars

- **Capital-arithmetic** (`cost_model`, `account_survival` DD/DLL, `dst` windows): **≥90% killed**.
- **State machines / large** (`execution_engine`, `strategy_fitness`, `build_daily_features`): **≥80% killed**, residual survivors individually justified.
- **100% is NOT the goal** — equivalent mutants make it unreachable; chasing it is its own theatre.

---

## Per-module results

### `pipeline/cost_model.py` — ✅ 100% kill (0% survival)

| Metric | Value |
|---|---|
| Mutants planned | 566 |
| Killed | 566 |
| Survived | 0 |
| **Survival rate** | **0.00%** (100% kill) |
| Test target | `tests/test_pipeline/test_cost_model.py` (53 tests, 2.2s, hermetic) |
| Bar (≥90% kill) | **PASS** — well above |

**Triage:** no survivors → no new killing tests required. The existing suite
asserts on the financial consequence of every operation (operator swaps,
comparison flips, constant replacements all caught). This is the *opposite* of
coverage-theatre — a genuinely strong suite, now proven mechanically.

This run also served as the **end-to-end tooling proof**: cosmic-ray executes a
full 566-mutant scoped run on native Windows / Python 3.11.9, exit 0.

### `pipeline/dst.py` (window math) — ✅ 100% kill (0% survival)

| Metric | Value |
|---|---|
| Mutants planned | 587 |
| Killed | 587 |
| Survived | 0 |
| **Survival rate** | **0.00%** (100% kill) |
| Test target | `tests/test_pipeline/test_dst.py` (105 tests, 2.3s, hermetic) |
| Bar (≥90% kill) | **PASS** — well above |

**Triage:** no survivors → no new killing tests required. Critically, the
`[start, end)` half-open window math — the exact class where an off-by-one
earlier slipped past *application*-code review — has its boundary conditions
(`<` vs `<=`, `start` vs `start+1`, half-open vs closed) fully pinned: all 587
mutations caught. dst's own window math is bulletproof.

### `trading_app/derived_state.py` — ✅ 100% kill (0% survival)

| Metric | Value |
|---|---|
| Mutants planned | 350 |
| Killed | 349 |
| Incompetent (auto-skipped invalid) | 1 |
| Survived | 0 |
| **Survival rate** | **0.00%** (100% kill; 349/349) |
| Test target | `test_lifecycle_state.py` + `test_sr_monitor.py` + `test_check_drift_ws2.py` (102 tests, 10.6s, hermetic) |
| Bar (≥90% kill) | **PASS** |

**Triage:** no survivors. The plan's premise was right — derived_state is NOT
zero-coverage; its fingerprint/state-envelope logic is fully exercised by three
indirect suites, and every mutation (incl. fingerprint-field swaps that would
let a stale artifact pass as live) is caught. 1 incompetent mutant excluded from
the denominator per standard mutation hygiene.

### `trading_app/account_survival.py` (DD/DLL) — ✅ 100% kill (0% survival)

| Metric | Value |
|---|---|
| Mutants planned | 1720 |
| Killed | 1719 |
| Incompetent | 1 |
| Survived | 0 |
| **Survival rate** | **0.00%** (100% kill; 1719/1719) |
| Test target | `tests/test_trading_app/test_account_survival.py` (28 tests, 5s, hermetic) |
| Bar (≥90% on DD/DLL) | **PASS** — 0 survivors anywhere, incl. DD/DLL enforcement arithmetic |

**Triage:** no survivors. The highest-stakes capital module (Monte-Carlo
prop-firm survival; a weak test = false "survived" verdict) has all 1719 valid
mutations caught — including the drawdown/loss-limit comparison arithmetic.
Thoroughness is proven by *placement*: cosmic-ray's 1720 mutants on 1146 LOC
(>1/line) saturate the file; none survived, so the DD/DLL lines were both
mutated AND killed.

### `trading_app/strategy_fitness.py` — ⏳ Tier-1 landed (45.71% → re-run pending); Tier-2 scoped follow-up

| Metric | Value |
|---|---|
| Mutants planned | 1236 |
| Killed (initial run) | 553 |
| Incompetent | 12 |
| Survived (initial run) | 671 |
| **Survival rate (initial)** | **54.29%** (45.71% kill) — **below the ≥80% bar** |
| Test target | `tests/test_trading_app/test_strategy_fitness.py` (39 tests after Tier-1, was 30) |

**Runner-defect found + fixed (this session).** The initial run's survivor detail
was nearly lost: `run_mutation.sh` trap-deletes the `.mutation/` sqlite on EXIT,
and the only durable artifact (`.mutation_run.log`) was written with
`cr-report --no-show-output` — **no line numbers**. Survivors were recovered by a
deterministic re-init (AST-only, reproduced all 1236 specs in ~4s) joined to the
log outcomes on the unique `(operator, occurrence)` key (0 join-misses). The
runner is now fixed to persist `cosmic-ray dump` (official NDJSON: line + outcome
+ diff per mutant — the canonical machine-readable survivor source) plus the
sqlite, BEFORE the trap fires. See `mutation-runner-survivor-persistence` stage.

**Survivor triage by consequence (all 671, audited — not name-guessed):**

| Class | Survivors | Disposition |
|---|---|---|
| Capital-decision logic | 365 | `classify_fitness` (4) + `_recent_trade_sharpe` (39) = **43 Tier-1, KILLED this session**; the rest (compute/diagnose/enrichment, ~322) are Tier-2 (DB-fixture-gated) |
| I/O / data-load | 192 | Tier-2 — `_enrich_relative_volumes`/`_minute_key`/`_apply_dst`/loaders; need bars_1m fixtures; also a canonical-delegation smell (mirrors `strategy_discovery._compute_relative_volumes`) |
| Cosmetic / presentation | 114 | `_format_table`/`_format_json`/`main` — `'='*80`/`'-'*80` separator arithmetic + log strings. **Behavior-neutral → justified-not-killed** (campaign doc § "100% is NOT the goal"). |

**Tier-1 kills — PROVEN at population level (43/43, deterministic):** every one
of the 43 prior survivors in the two functions was re-applied via
`cosmic-ray apply <module> core/<operator> <occurrence>` and the test suite
confirmed RED. Result: **NOW KILLED 43/43, STILL ALIVE 0** (source restored
clean after each). This is the campaign-DoD mechanical proof, not a sampled spot-
check. Measured module kill rate rises 553→596 killed = **48.7%** (596/1224
valid; 12 incompetent excluded) — still below the ≥80% bar, because Tier-1 was
deliberately the high-value *pure-function* slice (no DB fixtures), not the whole
module. The remaining gap is Tier-2 (below).

Per-mutant detail (all confirmed RED on re-apply):
- `classify_fitness` lines 119/125/129/132 — the `<`/`<=` operators on the
  FIT/WATCH/DECAY/STALE threshold comparisons survived because every existing test
  used *interior* values; none sat ON the boundary each mutant moves. 4 boundary
  tests added (sample==MIN_ROLLING_WATCH, exp_r==0.0→DECAY, sample==MIN_ROLLING_FIT,
  sharpe==SHARPE_DECAY_THRESHOLD), importing the constants (not inlining 10/15/-0.1)
  per institutional-rigor §4. Representative mutants L119 `<→<=` and L125 `<=→<`
  confirmed RED.
- `_recent_trade_sharpe` lines 157-164 — 27 arithmetic mutants on the Sharpe
  formula survived because all 3 existing tests asserted only `is None` /
  `isinstance float`, **never the value**. 4 value tests added pinning the exact
  hand-computed Sharpe (0.2886751 for `[2,-1,2,-1]`; the negative for the mirror)
  plus the n_trades==N and len<2 boundaries. Representative mutants L158 `(n-1)→(n)`
  and `(r-mean)→(r+mean)` confirmed RED (both target tests fail on the mutant).

**Equivalent-mutant candidates (Tier-1, justified-not-killed):**
- `_recent_trade_sharpe` L161 `std_r <= 0`→`std_r == 0`: `std_r` comes from
  `variance**0.5` so is always ≥ 0; the two predicates differ only on the
  impossible negative branch → **equivalent mutant**, not a test gap.

**Re-run / Tier-2 follow-up (NOT yet done — honest scope):**
1. Optional: full 8-worker mutation re-run against committed HEAD to re-confirm
   the module-wide kill rate end-to-end. NOT strictly required — the 43 Tier-1
   kills are already proven mutant-by-mutant via `cosmic-ray apply` (43/43 RED),
   which is stronger per-mutant evidence than a single aggregate rerun. Measured
   new kill rate = **48.7%** (596/1224); a full rerun would just reproduce this.
2. Tier-2: DB-fixture-backed tests for `_compute_fitness_from_cache`/`_with_con`,
   `diagnose_decay`/`diagnose_portfolio_decay`, and the `_enrich_*` helpers. This is
   the bulk of the path to ≥80% and carries real blast radius (fixtures touch gold.db
   read paths); scoped as a separate stage, deliberately NOT rushed here.

   **Stage 1 of Tier-2 DONE (2026-06-07): rel-vol canonical-delegation smell resolved.**
   `_compute_relative_volumes` (discovery) and `_enrich_relative_volumes` (fitness)
   were near-verbatim copies of the break-bar relative-volume math (institutional-rigor
   § 4 violation). Extracted shared canonical core
   `strategy_discovery._enrich_rel_vol_single_label`; both now delegate. Public name +
   signature of `_compute_relative_volumes` preserved (13 call sites). Behavior-neutral,
   proven by 3 golden/characterization tests incl. a cross-module equivalence assert
   (`tests/test_trading_app/test_strategy_discovery.py::TestComputeRelativeVolumes`,
   `abs=1e-12`). 408 caller tests green (3.13.9), drift 180/0, ruff clean. The
   DB-fixture tests for compute/diagnose/enrich + the scoped re-mutation are the
   remaining Tier-2 work (NOT yet done).

   **Stage 2 of Tier-2 DONE (2026-06-07): `diagnose_decay` decision-boundary slice.**
   The existing `TestDecayDiagnostics` tests asserted on STRUCTURE (which diagnosis
   enum fired, dataclass shape) but never sat ON the decay-fraction boundary nor
   exercised the FRAGMENTED branch — coverage 100%, mutation-kill 0% on the
   sibling-classification decision block (`diagnose_decay` lines 987-1015). 3
   DB-fixture tests added (`tests/test_trading_app/test_strategy_fitness.py`,
   `TestDecayDiagnostics`, +2 helper builders `_family_strategies`/`_family_outcomes`):
   - `test_decay_frac_exactly_half_is_regime_shift` — 2 DECAY + 2 FIT siblings →
     `decay_frac == 0.50` exactly. Kills the L1003 `decay_frac >= 0.50` → `> 0.50`
     mutant (inclusive boundary). **Proven RED via `Edit`-apply:** misroutes
     REGIME_SHIFT → FRAGMENTED.
   - `test_decay_frac_below_half_is_fragmented` — 1 DECAY + 2 FIT (frac 0.33).
     Kills the L1006 OVERFIT-guard `... == 0 and ... == 0` → `or` mutant. **Proven
     RED:** misroutes FRAGMENTED → OVERFIT.
   - `test_all_fit_siblings_is_overfit_with_exact_counts` — 3 FIT siblings, exact
     count asserts. Kills the L992 sibling accumulator `+ 1` → `- 1` mutant.
     **Proven RED:** `siblings_fit == -3`.
   All 3 killed mutant-by-mutant via apply→RED→restore (same per-mutant method as
   the Tier-1 kills; stronger than an aggregate rerun). Source byte-identical to
   HEAD after restore (`git diff trading_app/strategy_fitness.py` empty). Suite
   39 → 42 green (3.13.9), drift **180/0**, ruff clean.

   **HONEST residual gaps (NOT yet done — this is a slice, not Tier-2 complete):**
   - **Module kill rate unchanged in the headline (48.7%)** — per-mutant apply
     proves these 3, but no full re-run was done, so the aggregate figure is not
     re-quoted. These 3 are a small fraction of the ~322 Tier-2 survivors.
   - **`diagnose_decay` L998 `total_assessed == 0` → SINGLETON-fallback branch**
     and the L1002 `decay_frac` division remain untested (real branches, real
     survivors — next slice).
   - **`_compute_fitness_from_cache` / `_compute_fitness_with_con` / `_enrich_*` /
     `diagnose_portfolio_decay`** — entirely untouched this slice; the bulk of the
     path to ≥80% and the largest remaining clusters.
   - Cosmetic survivors (diagnosis_notes string constants L1005/L1008/L1013)
     deliberately NOT chased per § "100% is NOT the goal" — justified-not-killed.

### `trading_app/execution_engine.py` — pending
### `pipeline/build_daily_features.py` — pending (hermeticity-gated)

---

## Definition of Done (deterministic checkers only — no AI sign-off)

1. cosmic-ray score per module meets the bar; survivors all triaged (killed / equivalent / explicitly-noted).
2. `pytest tests/ -v` green on Python 3.13.9 (the project's actual interpreter via `uv run python`; output shown).
3. `python pipeline/check_drift.py` exit 0 (after `.mutation/` removed).
4. New Hypothesis property tests pass and demonstrably kill residual mutants.
5. This doc holds per-module before/after scores + survivor triage.
6. `evidence-auditor` dispatched as a **hypothesis-generator only** (same model
   family → correlated blind spots) — its output feeds Phase 1/2, never appears
   as a DoD pass.
