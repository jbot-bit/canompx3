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

### `trading_app/strategy_fitness.py` — pending
### `trading_app/execution_engine.py` — pending
### `pipeline/build_daily_features.py` — pending (hermeticity-gated)

---

## Definition of Done (deterministic checkers only — no AI sign-off)

1. cosmic-ray score per module meets the bar; survivors all triaged (killed / equivalent / explicitly-noted).
2. `pytest tests/ -v` green on Python 3.11.9 (output shown).
3. `python pipeline/check_drift.py` exit 0 (after `.mutation/` removed).
4. New Hypothesis property tests pass and demonstrably kill residual mutants.
5. This doc holds per-module before/after scores + survivor triage.
6. `evidence-auditor` dispatched as a **hypothesis-generator only** (same model
   family → correlated blind spots) — its output feeds Phase 1/2, never appears
   as a DoD pass.
