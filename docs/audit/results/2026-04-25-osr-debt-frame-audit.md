# O-SR Debt Frame Audit

**Date:** 2026-04-25
**Branch:** `research/osr-shiryaev-roberts-audit`
**Anchor commit:** `73329cd1` (origin/main at audit time)
**Stage:** `docs/runtime/stages/osr-debt-frame-audit.md` (RESEARCH).
**Design:** `docs/plans/2026-04-25-osr-debt-frame-audit-design.md`.

## Scope

Audit whether the `HANDOFF.md` "Next Steps - Active" line "**O-SR debt** - `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Multi-stage; not autonomous." (`HANDOFF.md:30`) corresponds to actual missing canonical work, or is stale framing.

Method: anchor-first, verdict-second. Four grounding bars are drafted before the verdict section so the verdict is mechanically derivable from primary sources.

**Out of scope:**

- Any change to monitor implementations (`cusum_monitor.py`, `sr_monitor.py`).
- Any backtest or canonical-layer query against `gold.db`.
- Empirical adequacy of the ARL approximately 60 trading days target.
- Parameter-tuning quality of the SR baseline window or threshold.
- Integration smoke-test of either monitor.
- Any decision-ledger / HANDOFF / docstring write — those belong to a separate Stage 2 design.

## Literature anchor

Source: `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` (arXiv:1509.01570 distilled passage).

### CUSUM (Eq 3 — verbatim from line 25)

> "**W_n ≜ max{0, W_{n-1} + ℒ_n}, n ≥ 1, W_0 = 0.**   (Equation 3)"

Stopping rule (Eq 4 — line 29):

> "𝒞_h ≜ min{n ≥ 1: W_n ≥ h}"

### Original Shiryaev-Roberts (Eq 10, 11 — verbatim from lines 37, 41)

> "𝒮_A ≜ min{n ≥ 1: R_n ≥ A}   (Equation 10)
> R_n ≜ (1 + R_{n-1}) Λ_n, n ≥ 1, R_0 = 0;   (Equation 11)"

### Score-based SR (Eq 13 — verbatim from line 59)

> "**R̃_n ≜ (1 + R̃_{n-1}) e^{S_n}, n ≥ 1, R̃_0 = 0**   (Equation 13)"

with stopping rule (Eq 14 — line 63):

> "𝒮̃_A ≜ min{n ≥ 1: R̃_n ≥ A}"

### Score-based CUSUM (Eq 15-16 — verbatim from lines 71, 75)

> "**W̃_n ≜ max{0, W̃_{n-1} + S_n}, n ≥ 1, W̃_0 = 0,**   (Equation 15)
> 𝒞̃_h ≜ min{n ≥ 1: W̃_n ≥ h}"

This is the score-function generalisation of Eq 3 — the same recursion shape as Eq 3 but with any computable score `S_n` in place of the LLR. Used below as the closer match for `cusum_monitor.py`'s implementation than the strict Eq 3 framing in the HANDOFF line.

### Score function (Eq 17, 18 — verbatim from lines 83, 91)

> "S_n(X̄_n) = C_1·X̄_n + C_2·X̄_n² - C_3   (Equation 17)
> C_1 = δq², C_2 = (1-q²)/2, C_3 = (δ²q²/2 - log q)   (Equation 18)"

with `X̄_n ≜ (X_n - μ_∞)/σ_∞` (centered + standardized observation), `q ≜ σ_∞/σ`, `δ ≜ (μ - μ_∞)/σ_∞`.

### Optimality claim relevant to verdict (literature line 47)

> "the Shiryaev-Roberts (SR) procedure is *exactly* optimal for every γ > 1 with respect to the stationary average detection delay STADD(T). Thus, in the multi-cyclic setting the SR procedure is a better alternative to the popular CUSUM chart."

### Project synthesis paragraph in literature passage (lines 130-134)

> "Implemented 2026-04-10 as:
> - `trading_app/live/sr_monitor.py` — score-function SR recursion + ARL calibration
> - `trading_app/sr_monitor.py` — deployed-lane runner using `paper_trades` first and canonical-forward fallback second
>
> The legacy `trading_app/live/cusum_monitor.py` remains as an intraday heuristic helper and should not be confused with the binding Criterion 12 monitor."

The literature passage is itself a derived layer. This synthesis paragraph is treated as a CLAIM in this audit, not as authority. The next four anchors verify or refute it from primary sources.

## Code anchor — CUSUM

Source: `trading_app/live/cusum_monitor.py` (62 lines total, anchor commit).

### Module-level intent

Module docstring (lines 1-12):

> "CUSUM (Cumulative Sum) control chart for detecting strategy performance drift.
> Reference: arXiv:1509.01570 — 'Real-time financial surveillance via quickest change-point detection methods'.
> One CUSUMMonitor per strategy. Raises alarm when cumulative downward deviation from the expected R per trade exceeds `threshold` standard deviations.
> threshold=4.0 ≈ 4σ of accumulated drift before alarm — conservative enough to avoid false positives over ~100-trade windows while catching genuine regime change."

The docstring cites the same arXiv paper as the literature passage, but states the intent as "control chart" with "standard deviations" semantics, not as the binding multi-cyclic SR procedure.

### Recursion against Eq 3

`cusum_monitor.py:39-41` — the update step:

```
z = (actual_r - self.expected_r) / self.std_r
self.cusum_neg = min(0.0, self.cusum_neg + z)  # tracks persistent losses
self.cusum_pos = max(0.0, self.cusum_pos + z)  # tracks persistent gains
```

This is a two-sided CUSUM in standardized-z increments. The negative side tracks adverse drift; the positive side tracks favorable drift. The negative side is the alarm channel (`cusum_monitor.py:43`):

```
if -self.cusum_neg > self.threshold and not self.alarm_triggered:
```

Comparison against the literature passage:

- Textbook Eq 3 (`W_n ≜ max{0, W_{n-1} + ℒ_n}`) uses a log-likelihood ratio `ℒ_n` as the score.
- The literature passage also gives Eq 15 — the **score-function modified CUSUM** — at line 71: `W̃_n ≜ max{0, W̃_{n-1} + S_n}, n ≥ 1, W̃_0 = 0`. This variant generalises Eq 3 by allowing any computable score `S_n` in place of the LLR, mirroring the score-based SR generalisation Eq 13.
- The project's `cusum_monitor.py:39-41` implementation matches Eq 15 more precisely than Eq 3: the score is the standardized residual `z = (actual_r - expected_r) / std_r`, not an LLR. The accumulator at line 41 (`cusum_pos = max(0.0, cusum_pos + z)`) is structurally identical to Eq 15's `W̃_n = max(0, W̃_{n-1} + S_n)` with `S_n = z`. The mirror line 40 (`cusum_neg = min(0.0, cusum_neg + z)`) tracks the adverse side — equivalent in absolute-value terms to a parallel Eq 15 accumulator on the negated score, with alarm at line 43 firing when `-cusum_neg > threshold` (analogue of `W̃_n ≥ h` in Eq 16).

**Verdict on CUSUM faithfulness:** the recursion is a polarity-mirrored two-sided **score-function CUSUM (Eq 15-16) with `S_n` chosen as the standardized residual `z`**. The HANDOFF line's framing of this as "CUSUM Eq 3" is technically imprecise but substantively close — Eq 15 IS a CUSUM, just the score-function variant rather than the LLR variant. The implementation is faithful to its own self-described intent (intraday alert with σ-units threshold), not to Eq 13-14 score-based SR.

### Operational behaviour

`cusum_monitor.py:48-57` `clear()` resets accumulators and alarm flag but **preserves `n_trades`** (line 51-53 docstring: "n_trades is intentionally preserved — cumulative trade count across the session is useful for diagnostics in alarm messages").

`cusum_monitor.py:60-62` `drift_severity` returns `-cusum_neg` (positive number when underperforming).

### Test-locked intent

Source: `tests/test_trading_app/test_cusum_monitor.py`.

- `test_cusum_monitor.py:4-8` — no alarm on good performance.
- `test_cusum_monitor.py:11-15` — alarm on persistent losses (20× -1.0 trades).
- `test_cusum_monitor.py:18-22` — drift severity positive when losing.
- `test_cusum_monitor.py:25-29` — clear() resets alarm flag.
- 10 total tests in the file (counted by `grep -c "^def test_" tests/test_trading_app/test_cusum_monitor.py` at anchor commit). Two of the ten — `test_std_r_calibrated_per_strategy` and `test_std_r_rr1_stays_near_one` — assert `PerformanceMonitor` integration semantics (calibrated `std_r` per strategy and RR1 distribution shape); the remaining eight assert pure CUSUM intraday-alert behaviour with σ-units threshold.

The tests do NOT assert any property characteristic of a multi-cyclic Shiryaev-Roberts procedure (no ARL calibration, no exponentiated score, no Eq 13 multiplicative recursion `(1 + R_{n-1}) e^{S_n}`). All 10 pass at anchor commit `73329cd1` (`pytest tests/test_trading_app/test_cusum_monitor.py -q` -> 10 passed).

## Code anchor — SR

Sources: `trading_app/live/sr_monitor.py` (167 lines) and `trading_app/sr_monitor.py` (370 lines).

### Module-level intent — `trading_app/live/sr_monitor.py`

`sr_monitor.py:1-10` (live):

> "Score-based Shiryaev-Roberts drift monitor.
> Implements the nonparametric score-function SR recursion from Pepelyshev & Polunchenko (2015), Eq. 13-18:
>   R_n = (1 + R_{n-1}) * exp(S_n)
> where S_n is a change-sensitive linear-quadratic score applied to the standardized trade outcome stream."

### Module-level intent — `trading_app/sr_monitor.py`

`trading_app/sr_monitor.py:1-7`:

> "Shiryaev-Roberts live drift monitor for deployed lanes.
> This is the Criterion 12 monitor:
> - live/paper trade stream preferred
> - canonical forward outcomes allowed only as an explicitly labeled fallback
> - threshold calibrated to approximately 60 trading days ARL"

The runner module names itself the Criterion 12 monitor.

### Recursion against Eq 13

`trading_app/live/sr_monitor.py:151` — the update step:

```
self.sr_stat = (1.0 + self.sr_stat) * math.exp(self.score(actual_r))
```

This matches Eq 13 (`R̃_n ≜ (1 + R̃_{n-1}) e^{S_n}`) exactly: multiplicative recursion, `R̃_0 = 0` initial state at `sr_stat: float = field(default=0.0, init=False)` (`sr_monitor.py:133`).

Stopping rule against Eq 14 — `sr_monitor.py:152`:

```
if self.sr_stat >= self.threshold and not self.alarm_triggered:
```

Matches Eq 14 (`𝒮̃_A ≜ min{n ≥ 1: R̃_n ≥ A}`).

### Score function against Eq 17-18

`trading_app/live/sr_monitor.py:30-36`:

```
c1 = delta * q * q
c2 = (1 - q * q) / 2
c3 = (delta * delta * q * q) / 2 - math.log(q)
```

These match Eq 18 verbatim: `C_1 = δq², C_2 = (1-q²)/2, C_3 = (δ²q²/2 - log q)`.

`trading_app/live/sr_monitor.py:137-143` — score evaluation:

```
x = (actual_r - self.expected_r) / self.std_r
c1, c2, c3 = _score_coefficients(self.delta, self.variance_ratio)
return c1 * x + c2 * (x * x) - c3
```

This matches Eq 17 verbatim: `S_n(X̄_n) = C_1·X̄_n + C_2·X̄_n² - C_3`, with `X̄_n` defined as `(X_n - μ_∞)/σ_∞` per the literature passage and instantiated at `sr_monitor.py:141` as `(actual_r - expected_r) / std_r`.

### ARL calibration

`trading_app/live/sr_monitor.py:39-65` `_estimate_arl` runs Monte Carlo on standardized iid samples; `sr_monitor.py:68-120` `calibrate_sr_threshold` does bisection bracketing to find the threshold whose pre-change ARL meets the target. Default target is `target_arl: int = 60` at `sr_monitor.py:70`.

### Test-locked intent — SR

Source: `tests/test_trading_app/test_sr_monitor.py` — 10 tests at anchor commit. All 10 pass at anchor commit `73329cd1` (`pytest tests/test_trading_app/test_sr_monitor.py -q` -> 10 passed). Combined with the CUSUM anchor section's pytest result (10 passed), the full monitor-pair sweep is `20 passed` at `73329cd1`.

- `test_sr_monitor.py:11-15` — no alarm on good performance.
- `test_sr_monitor.py:18-25` — alarm on persistent losses + alarm_ratio >= 1.0.
- `test_sr_monitor.py:28-36` — clear() resets sr_stat to 0.0.
- `test_sr_monitor.py:39-41` — `calibrate_sr_threshold(target_arl=60, ...)` returns a value between 20 and 45 (ARL bisection produces a sensible threshold for the target).
- `test_sr_monitor.py:44+` — `prepare_monitor_inputs` prefers live baseline after 50 paper trades.

These tests assert SR-specific properties: multiplicative-recursion alarm, calibrated threshold near the 60-day ARL target, paper-trade-baseline preference. These cannot be satisfied by the CUSUM file.

## Consumer anchor

### CUSUM consumer chain

`trading_app/live/performance_monitor.py:18` imports `CUSUMMonitor`. `performance_monitor.py:69-77` instantiates one CUSUM per strategy in `__init__`; `performance_monitor.py:81-95` `record_trade` updates and emits an in-memory alert when the alarm fires; `performance_monitor.py:109-121` `reset_daily` clears the daily accumulators and the CUSUM monitors at EOD.

This is the **only** production caller of `CUSUMMonitor`. Grep across the worktree finds no other production import. The cadence is intra-session: per-trade update + EOD reset. The consequence is an in-memory alert (`record_trade` returns a string for the operator), not a state-machine action.

### SR consumer chain

`trading_app/sr_monitor.py:32` imports `ShiryaevRobertsMonitor` and `calibrate_sr_threshold` from `trading_app/live/sr_monitor.py`. The runner module:

1. Builds lanes from `prop_profiles.get_profile_lane_definitions(profile_id)` (`sr_monitor.py:82-104`).
2. Loads pre-change baseline preferring live `paper_trades` (`sr_monitor.py:150-184`).
3. Calibrates threshold to `TARGET_ARL_DAYS = 60` (`sr_monitor.py:40`, `186-190`).
4. Streams trades through `monitor.update(trade_r)` looking for alarm (`sr_monitor.py:271-275`).
5. On alarm with `--apply-pauses`, calls `pause_strategy_id(...)` via `lane_ctl` (`sr_monitor.py:201-233`). The pause source is recorded as `"sr_monitor"` (`sr_monitor.py:225`).
6. Persists state envelope to `data/state/sr_state.json` with profile fingerprint, DB identity, code fingerprint, freshness, and results (`sr_monitor.py:327-351`).

Downstream readers of the state envelope:

- `trading_app/lifecycle_state.py:130` reads `expected_state_type="sr_monitor"`; `lifecycle_state.py:135-141` validates against the current code fingerprint of both `trading_app/sr_monitor.py` AND `trading_app/live/sr_monitor.py`. The lifecycle layer treats the SR runner as a canonical state contributor.
- `scripts/tools/refresh_control_state.py:22` imports `run_monitor` from `trading_app.sr_monitor` to refresh control state for a profile.
- `scripts/tools/project_pulse.py:110` registers the SR runner in the `SKILL_SUGGESTIONS` map as a pulse action: `"sr_monitor": "python -m trading_app.sr_monitor"`. `project_pulse.py:969-1085` consumes SR-state freshness via `lifecycle_state.read_lifecycle_state` (not by direct import of `sr_monitor`) and emits pulse items keyed `source="sr_monitor"`, including `action="python -m trading_app.sr_monitor --apply-pauses"` at line 1003. The consumer chain for `project_pulse` is therefore: `sr_monitor` runner persists state envelope -> `lifecycle_state` validates it against current code/profile/DB fingerprint -> `project_pulse` reads the validated envelope. `refresh_control_state.py:22` is the only path that imports `run_monitor` directly.

The SR runner is wired into the live operational loop: state freshness governs operator-pulse output, alarm action writes lane-level pauses through `lane_ctl`, and code fingerprint is bound into the lifecycle envelope.

### Cadence and consequence summary

- CUSUM: per-trade intraday update by `performance_monitor`; EOD reset; consequence is an in-memory operator alert string.
- SR: per-run batch over historical paper-trade or canonical-forward stream; threshold calibrated to ARL approximately 60 trading days; consequence is a `lane_ctl` pause record (suspension) when invoked with `--apply-pauses`.

The two monitors do not share consumers, do not share cadence, and do not share consequence shape.

## Criterion 12 anchor

Source: `docs/institutional/pre_registered_criteria.md:210-218`.

> "## Criterion 12 — Live monitoring via Shiryaev-Roberts
>
> Source: `literature/pepelyshev_polunchenko_2015_cusum_sr.md`.
>
> Rule: Every deployed strategy must have a Shiryaev-Roberts drift monitor running against its live R-multiple stream. Parameters:
> - Pre-change distribution estimated from first 50-100 live trades
> - Score function: linear-quadratic per Eq. 17-18 of the paper
> - Detection threshold A calibrated to ARL to false alarm ≈ 60 trading days
> - On alarm: strategy goes to 'suspended' state pending manual review"

Per-parameter audit against the wired SR runner:

| Criterion 12 parameter | Wired runner state | Verdict |
|---|---|---|
| Pre-change distribution from first 50-100 live trades | `BASELINE_WINDOW = 50` at `trading_app/sr_monitor.py:43`; baseline derived from first 50 paper trades when available at `sr_monitor.py:167-177`; falls back to `validated_setups` backtest stats when paper trades insufficient | VERIFIED at the lower bound of the 50-100 range. The Criterion's "50-100" window is implemented at the floor (50). |
| Score function: linear-quadratic per Eq. 17-18 | `_score_coefficients` at `trading_app/live/sr_monitor.py:30-36` matches Eq 18 verbatim; score evaluation at `sr_monitor.py:137-143` matches Eq 17 verbatim | VERIFIED |
| Threshold A calibrated to ARL approximately 60 trading days | `TARGET_ARL_DAYS = 60` at `trading_app/sr_monitor.py:40`; `calibrate_sr_threshold(TARGET_ARL_DAYS, ...)` invoked at `sr_monitor.py:186-190`; bisection routine at `trading_app/live/sr_monitor.py:68-120`; test `test_sr_monitor.py:39-41` asserts the calibration produces 20.0 < threshold < 45.0 | VERIFIED |
| On alarm: strategy goes to "suspended" state pending manual review | Alarm action at `trading_app/sr_monitor.py:201-233` calls `pause_strategy_id(...)` via `lane_ctl` with reason `"SR alarm: stat=... >= thr=..."` and source `"sr_monitor"`; default `pause_days=30` at `sr_monitor.py:236, 367` | VERIFIED (mapped to lane-level pause; the project's "suspended" semantics is a `lane_ctl` pause override) |

All four Criterion 12 parameters are implemented in the wired SR runner. None are implemented in `cusum_monitor.py`.

## Verdict

**MISFRAMED.**

The `HANDOFF.md:30` line "O-SR debt - `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `pepelyshev_polunchenko_2015_cusum_sr.md`. Multi-stage; not autonomous." conflates two separate canonical surfaces:

1. The **binding Criterion 12 monitor** for post-deployment drift detection. This is `trading_app/sr_monitor.py` (the runner) backed by `trading_app/live/sr_monitor.py` (the score-based SR core). All four Criterion 12 parameters are implemented (verified above). Live wiring through `lifecycle_state`, `refresh_control_state`, and `project_pulse` is in place. Test-locked behaviour is in `tests/test_trading_app/test_sr_monitor.py`. There is **no debt** on this surface.
2. The **intraday operator-alert tool** `trading_app/live/cusum_monitor.py`. This is a CUSUM control chart used only by `performance_monitor.py` for per-trade in-memory alerts during a live session. It does not claim to be the Criterion 12 monitor — its docstring describes it as a control chart with σ-units threshold, and its tests lock that intent. There is **no debt** on this surface either; it is functioning as designed for a different purpose.

The HANDOFF line's literal-reading framing ("cusum_monitor.py should implement SR") is incorrect because cusum_monitor was never the binding monitor. The binding monitor exists, satisfies Criterion 12, and is wired into the live operational loop. The line is a residual artefact from before the 2026-04-10 SR implementation landed.

### Evidence supporting MISFRAMED

- Eq 13-14 score-based SR is implemented at `trading_app/live/sr_monitor.py:151-152` (matches verbatim).
- Eq 17-18 score function and coefficients are implemented at `trading_app/live/sr_monitor.py:30-36, 137-143` (matches verbatim).
- ARL approximately 60 days target is implemented at `trading_app/sr_monitor.py:40, 186-190`.
- "Suspended" alarm action is implemented at `trading_app/sr_monitor.py:201-233` via `lane_ctl.pause_strategy_id`.
- Live wiring is implemented at `trading_app/lifecycle_state.py:130, 135-141`, `scripts/tools/refresh_control_state.py:22`, `scripts/tools/project_pulse.py:110, 969-1085`.
- `cusum_monitor.py` is reachable only via `performance_monitor.py:18`; no other production importer.
- 20/20 tests across both monitors pass at anchor commit `73329cd1`.

### Evidence that would refute MISFRAMED (none found)

To refute the verdict, one of the following would have to be true:

- A primary canonical surface (decision-ledger entry, action-queue item with `status: ready`, or pre-registered criterion) explicitly stating that `cusum_monitor.py` is the Criterion 12 monitor. **Searched: not found in `docs/runtime/decision-ledger.md` (full read, 36 lines) or `docs/runtime/action-queue.yaml` (full read, 143 lines).**
- A live consumer of `cusum_monitor.py` that treats its output as a binding deployment-state action (lane pause, kill-switch). **Searched: only consumer is `performance_monitor.py`, which emits an in-memory alert string; no `lane_ctl` write, no kill-switch wire.**
- A literature requirement satisfied only by Eq 3 CUSUM and not by Eq 13-14 score-based SR. **The literature passage at line 47 explicitly states the opposite: SR is "exactly optimal" relative to CUSUM in the multi-cyclic setting.**
- Test coverage on `cusum_monitor.py` asserting Criterion-12-style multi-cyclic behaviour. **Searched: `test_cusum_monitor.py` 8 tests, all assert intraday-alert σ-units behaviour; none assert SR-style multi-cyclic properties.**

No refuting evidence exists.

## Action recommendation

These are recommendations, not actions taken in this stage. Stage 1 is verdict-only; Stage 2 (separate design) decides what to do with the verdict.

1. **Decision-ledger entry.** Write one durable line capturing the MISFRAMED verdict and pointing to this audit doc. Without a ledger entry, the question is reopenable.
2. **HANDOFF line removal or reframe.** The current line should be either (a) removed entirely, or (b) replaced with a short, accurate note such as "Intraday `CUSUMMonitor` (`trading_app/live/cusum_monitor.py`) is a separate σ-units alert tool, not the Criterion 12 monitor; binding SR lives in `trading_app/sr_monitor.py` and `trading_app/live/sr_monitor.py`." The choice between (a) and (b) is a Stage 2 question.
3. **Optional `cusum_monitor.py` module-docstring clarifier.** Add one sentence to the existing docstring stating "This is the intraday performance-alert helper used by `PerformanceMonitor`. The binding Criterion 12 Shiryaev-Roberts monitor is `trading_app/sr_monitor.py`; do not confuse the two." Net cost: one line, prevents future misframing at the source.
4. **No code change to the monitors themselves.** Both implementations are faithful to their stated intents and pass their tests.

## Outputs

Stage 1 ships exactly these three new files (verified via `git diff --stat origin/main HEAD` on `research/osr-shiryaev-roberts-audit`):

- `docs/plans/2026-04-25-osr-debt-frame-audit-design.md` — design doc (4-turn flow output).
- `docs/runtime/stages/osr-debt-frame-audit.md` — RESEARCH-mode stage doc with scope_lock and acceptance criteria.
- `docs/audit/results/2026-04-25-osr-debt-frame-audit.md` — this audit verdict doc.

No canonical surfaces are touched in Stage 1 (no HANDOFF edit, no decision-ledger entry, no action-queue change, no monitor module edit). Stage 2 design and execution will take the four recommendations listed in the Action recommendation section above.

### Reproduction

To reproduce the audit's verifiable claims at the anchor commit `73329cd1`:

1. `git fetch origin && git checkout 73329cd1`
2. `python -m pytest tests/test_trading_app/test_cusum_monitor.py -q` -> expect `10 passed`.
3. `python -m pytest tests/test_trading_app/test_sr_monitor.py -q` -> expect `10 passed`.
4. `grep -c "^def test_" tests/test_trading_app/test_cusum_monitor.py` -> expect `10`.
5. `grep -n "O-SR debt" HANDOFF.md` -> expect a single hit on line `30`.
6. `grep -nE "CUSUMMonitor|cusum_monitor" trading_app/ scripts/ pipeline/` (excluding `tests/`, `docs/`) -> expect production hits ONLY in `trading_app/live/cusum_monitor.py` (the module itself) and `trading_app/live/performance_monitor.py:18` (the only consumer).
7. `grep -nE "from trading_app.sr_monitor|from trading_app.live.sr_monitor" trading_app/ scripts/ pipeline/` -> expect hits in `trading_app/sr_monitor.py:32`, `trading_app/lifecycle_state.py:130, 137-138` (path references), and `scripts/tools/refresh_control_state.py:22`.
8. Read `docs/institutional/pre_registered_criteria.md:210-218` and confirm Criterion 12's four parameters quoted in this audit are verbatim.

Each step that fails at `73329cd1` is a refutation of the corresponding anchor-section claim and would require the audit to be reworked.

## Limitations

This audit does NOT cover and does NOT claim a verdict on:

- **Empirical adequacy of ARL approximately 60 days.** Whether 60 trading days is the right false-alarm cadence for this project's trade volumes (6 deployed lanes, intra-day, multi-instrument) is a separate question. The audit only confirms the wired runner targets that ARL value.
- **Score-function parameter quality.** `DEFAULT_DELTA = -1.0` and `DEFAULT_VARIANCE_RATIO = 1.0` (`trading_app/sr_monitor.py:41-42`) reflect the literature-passage default of "mean-shift-only monitoring with q=1.0 collapses the quadratic term to zero". Whether δ = -1.0σ is the right effect-size to detect is not audited here.
- **Integration smoke-test.** The audit confirms wiring exists at the static-import level (`lifecycle_state`, `refresh_control_state`, `project_pulse`). It does NOT confirm that the runner has actually been invoked in production with `--apply-pauses` and successfully written a `lane_ctl` pause; that is an operational question.
- **Whether `performance_monitor.py`'s intraday CUSUM is fit-for-purpose.** The audit only confirms `cusum_monitor.py` is faithful to its self-described intent and is not the Criterion 12 monitor. Whether intraday operator alerts are well-served by the σ-units threshold (`CUSUM_THRESHOLD = 4.0` at `performance_monitor.py:66`) is a separate question.
- **Pepelyshev-Polunchenko 2015 paper PDF.** The audit cites the project literature passage (a derived layer) verbatim. The audit does not re-read the underlying PDF (`resources/real_time_strategy_monitoring_cusum.pdf` — not present in this worktree's `resources/` directory; passage extracted 2026-04-07 per the literature passage frontmatter).

## References

### Primary sources

- `HANDOFF.md:30` — the audited line.
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md:25` (Eq 3), `:37` (Eq 10), `:41` (Eq 11), `:47` (optimality claim), `:59` (Eq 13), `:63` (Eq 14), `:71` (Eq 15), `:75` (Eq 16), `:83` (Eq 17), `:91` (Eq 18), `:130-134` (project synthesis claim).
- `docs/institutional/pre_registered_criteria.md:210-218` — Criterion 12 binding rule.
- `trading_app/live/cusum_monitor.py:1-12, 39-46, 48-57, 60-62` — CUSUM module + recursion + clear + drift_severity.
- `trading_app/live/sr_monitor.py:1-10, 30-36, 137-143, 151-152, 39-65, 68-120, 133` — SR module + score + recursion + ARL calibration + initial state.
- `trading_app/sr_monitor.py:1-7, 40-43, 82-104, 150-184, 186-190, 201-233, 271-275, 327-351, 367` — Criterion 12 runner.
- `trading_app/live/performance_monitor.py:18, 66, 69-77, 81-95, 109-121` — intraday CUSUM consumer.
- `tests/test_trading_app/test_cusum_monitor.py:4-8, 11-15, 18-22, 25-29` — CUSUM intent locks.
- `tests/test_trading_app/test_sr_monitor.py:11-25, 28-36, 39-41, 44+` — SR intent locks.
- `trading_app/lifecycle_state.py:130, 135-141` — SR state validation.
- `scripts/tools/refresh_control_state.py:22` — SR runner imported into control refresh.
- `scripts/tools/project_pulse.py:110, 969-1085` — SR runner registered as pulse action.

### Verification at anchor commit

- `pytest tests/test_trading_app/test_cusum_monitor.py -q` -> 10/10 passing on commit `73329cd1` (cited in CUSUM anchor body).
- `pytest tests/test_trading_app/test_sr_monitor.py -q` -> 10/10 passing on commit `73329cd1` (cited in SR anchor body).
- Full monitor-pair sweep: 20 passed.
- `grep -c "^def test_" tests/test_trading_app/test_cusum_monitor.py` -> 10 (cited in CUSUM anchor body).

### Silent canonical surface (negative evidence)

- `docs/runtime/decision-ledger.md` — no entry on O-SR / SR / CUSUM / cusum_monitor / sr_monitor. The silence is itself audit-relevant: per the `runtime-shell-unification` decision (line 8 of decision-ledger), the ledger is canonical; the absence of a closing entry is what allowed the HANDOFF line to persist as misframed without triggering a ledger update. Stage 2 should write the closing entry.
- `docs/runtime/action-queue.yaml` — no entry on O-SR / SR / CUSUM. The HANDOFF line is the only canonical reference to the alleged debt.
