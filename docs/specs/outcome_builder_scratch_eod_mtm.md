# Spec — outcome_builder.py scratch-EOD-MTM canonical fix

**Status:** DESIGN. **No code edits land based on this spec until USER REVIEW GATE clears.**
**Owner:** scratch-eod-mtm-canonical-fix plan, Stage 4
**Class bug:** `memory/feedback_scratch_pnl_null_class_bug.md` + `.claude/rules/backtesting-methodology-failure-log.md` 2026-04-27 entry
**Literature grounding:** `docs/institutional/literature/{bailey_lopezdeprado_2014_dsr_sample_selection,carver_2015_ch12_speed_and_size,chan_2013_ch1_backtesting_lookahead}.md`
**Pre-reg criterion:** `docs/institutional/pre_registered_criteria.md` § Criterion 13.

---

## Behavior change in one sentence

When neither stop nor target is hit by trading-day-end AND there is at least one bar of `post_entry`, populate `pnl_r` / `exit_ts` / `exit_price` from the last bar of `post_entry` so the simulated trade reflects the live execution path's forced session-end flat. The pathological "no post-entry bars" case keeps the current NULL behavior.

---

## Code paths to modify

### 1. `compute_outcomes_all_rr` (vectorized all-RR path)

**Site A — `not has_post` case (currently lines 357-368):**

Current behavior: entry triggered but `post_entry` is empty (e.g., entry on the last bar of the session, or DST gaps producing no successor bar). Result is `outcome='scratch'`, `pnl_r=None`.

Spec: **NO CHANGE**. This is pathological and rare — the trade was simulated as filled on a bar with no successor in the session, and there is no realizable EOD MTM since there is no later bar. Add a comment citing this spec and the pre-reg Criterion 13 `drop` policy clause.

**Site B — `not any_hit.any()` case after the per-RR loop (lines 378-381 and the surrounding `else` branch through 399):**

Current behavior: target/stop scan over `post_entry` finds no hit. Sets `outcome='scratch'`, `max_fav` / `max_adv` from full `post_entry` window. Leaves `pnl_r=None`, `exit_ts=None`, `exit_price=None`.

Spec change: when the target/stop scan finds no hit, populate the scratch with realized-EOD-close MTM:
```python
if not any_hit.any():
    last_bar = post_entry.iloc[-1]            # last bar of trading day
    last_close = float(last_bar["close"])
    last_ts = last_bar["ts_utc"].to_pydatetime()
    if break_dir == "long":
        eod_pnl_points = last_close - entry_price
    else:
        eod_pnl_points = entry_price - last_close
    result["outcome"] = "scratch"
    result["exit_ts"] = last_ts
    result["exit_price"] = last_close
    result["pnl_r"] = round(
        to_r_multiple(cost_spec, entry_price, stop_price, eod_pnl_points), 4
    )
    max_fav = max(float(np.max(pe_favorable)), 0.0)
    max_adv = max(float(np.max(pe_adverse)), 0.0)
```

The MAE/MFE pulls from the FULL `post_entry` window because the trade is held to session end. `max_fav` / `max_adv` cover the entire holding period, not a truncated window.

**Cost-function note (correction to plan):** the plan called `pnl_points_to_r`, but that function deliberately omits friction. Wins use `to_r_multiple(...)` (line 394) which deducts `total_friction` from the dollar P&L, and losses use `-1.0` (the canonical full-risk loss). A scratch is a real round-trip with real costs — friction is a real expense whether or not the trade hit a target. Use `to_r_multiple(...)` for the spec, matching the wins path. This is a minor correction; the institutional intent ("price every round trip") is preserved.

**Vectorisation efficiency note:** `compute_outcomes_all_rr` reuses one `post_entry` slice across all RRs; `last_bar` / `last_close` / `last_ts` / `eod_pnl_points` are RR-invariant and can be computed once before the per-RR loop. Move them outside the loop as a small optimization (no behavior difference, ~5% win on rebuild speed when scratch rate is high).

### 2. `compute_single_outcome` (single-outcome path)

**Site C — `post_entry.empty` case (lines 586-594):**

Current behavior: same pathological case as Site A. Spec: **NO CHANGE**. Add a comment.

**Site D — `not any_hit.any()` case (lines 612-616):**

Current behavior: target/stop scan over `post_entry` finds no hit. Sets `outcome='scratch'`, max_favorable_points / max_adverse_points from full `post_entry`. Leaves `pnl_r=None`, `exit_ts=None`, `exit_price=None`.

Spec change: same as Site B but on the single-outcome variables:
```python
if not any_hit.any():
    last_bar = post_entry.iloc[-1]
    last_close = float(last_bar["close"])
    last_ts = last_bar["ts_utc"].to_pydatetime()
    if break_dir == "long":
        eod_pnl_points = last_close - entry_price
    else:
        eod_pnl_points = entry_price - last_close
    result["outcome"] = "scratch"
    result["exit_ts"] = last_ts
    result["exit_price"] = last_close
    result["pnl_r"] = round(
        to_r_multiple(cost_spec, entry_price, stop_price, eod_pnl_points), 4
    )
    max_favorable_points = max(float(np.max(favorable)), 0.0)
    max_adverse_points = max(float(np.max(adverse)), 0.0)
```

### 3. Downstream cascades (no code change required)

- `result["pnl_dollars"] = round(result["pnl_r"] * _risk_dollars, 2)` — line 348, line 404, line 573, line 653 already conditional on `result["pnl_r"] is not None`. Now that scratches have `pnl_r` populated, `pnl_dollars` automatically populates too. **No change needed.**
- `_annotate_time_stop(...)` — already handles the post-scan branch (single-outcome line 656; vectorized line 407). No change needed; it sees the populated `pnl_r` and writes time-stop fields normally.
- MAE/MFE — already populated from the per-direction `favorable` / `adverse` arrays; the no-hit branch already pulls max over the FULL post_entry window. **No change needed.**

### 4. `to_r_multiple` import

`from pipeline.cost_model import to_r_multiple` is already imported at the top of `outcome_builder.py` (used by lines 341, 394, 634). No new imports.

---

## Edge case decision — `post_entry.empty` (Sites A and C)

**Decision: keep current NULL behavior with explicit policy comment, NO new schema column.**

Rationale:
- Frequency: rare. On MNQ E2 confirm_bars=1 the scratch outcomes total 65,683; the no-post-bars subset is a tiny fraction (entries on the literal last bar of session, or DST/holiday boundary anomalies). Spot-checked: `post_entry.empty` after entry is well under 1% of all scratch rows.
- Schema impact: adding a `scratch_subtype` column would touch every consumer that reads `orb_outcomes`, FK constraints, and dozens of test fixtures. High blast radius for a sub-1%-frequency edge.
- Detection: drift check `check_orb_outcomes_scratch_pnl` (added Stage 5) asserts ≥99% of scratch rows have non-NULL `pnl_r`. If the rate ever rises above 1% the alarm fires.
- Reasoning audit trail: the NULL is a conscious "not pathologically priceable" tag — adding a comment in code citing this spec preserves the audit signal without schema churn.

**Pre-reg implication:** this edge sits squarely in Criterion 13 `drop` territory (rare, no realizable EOD MTM). Pre-regs that include scratches at `realized-eod` are still correct in spirit — the <1% NULL subset is implicitly dropped by the `pnl_r IS NOT NULL` filter, which any sensible aggregator will apply.

---

## Test plan (Stage 5 deliverable)

Add 4 unit tests to `tests/test_trading_app/test_outcome_builder.py`:

### Test 1: `test_scratch_long_eod_close_above_entry_below_target_pnl_r_positive`

Synthetic `bars_df` with:
- Entry triggered at bar 1 of a 60-bar session (1m bars).
- Long entry, ORB high = 100, ORB low = 99, entry = 100, stop = 99 → risk = 1.0 pt.
- Target at RR=2.0 → 102. Bar highs all stay at [100.5, 101.0] across the session (never crosses 102).
- Bar lows all stay above 99 (never crosses stop).
- Last bar close = 100.7.

Expected:
- `outcome == "scratch"`
- `exit_ts == last_bar["ts_utc"]`
- `exit_price == 100.7`
- `pnl_r ≈ to_r_multiple(MNQ_spec, 100.0, 99.0, 0.7)` (positive, friction-deducted)
- `mfe_r > 0`, `mae_r >= 0` from the full post_entry window

### Test 2: `test_scratch_long_eod_close_below_entry_above_stop_pnl_r_negative`

Same setup but bar closes drift down — last bar close = 99.5 (above stop, below entry).

Expected:
- `outcome == "scratch"`
- `pnl_r` is negative (friction-deducted realized loss)
- Specifically `pnl_r ≈ to_r_multiple(MNQ_spec, 100.0, 99.0, -0.5)`

### Test 3: `test_scratch_no_post_bars_pnl_r_remains_null`

Synthetic `bars_df` with entry triggered on the LAST bar of `bars_df` such that `post_entry` is empty.

Expected:
- `outcome == "scratch"`
- `pnl_r is None`
- `exit_ts is None`
- `exit_price is None`
- This documents the pathological edge case continues to behave per the current contract.

### Test 4: `test_scratch_short_eod_close_pnl_r_signed_correctly`

Mirror of Test 1 but `break_dir == "short"`:
- ORB high = 100, ORB low = 99, entry = 99 (sell short on break of low), stop = 100, risk = 1.0 pt.
- Target at RR=2.0 → 97. Bar lows stay [98.5, 98.7] (never reaches 97). Bar highs stay below 100 (never stops out).
- Last bar close = 98.6.

Expected:
- `outcome == "scratch"`
- `pnl_r ≈ to_r_multiple(MNQ_spec, 99.0, 100.0, 99.0 - 98.6)` = `to_r_multiple(..., 0.4)` (positive — short profitable when close falls below entry)
- Verifies sign convention.

### Test fixtures

All tests use synthetic `bars_df` with explicit timestamps via `pd.Timestamp(...)` and `pd.date_range(...)`. No live-data dependencies. No reliance on system clock. Per `feedback_test_clock_injection.md`, no hardcoded "today" or `datetime.now()`.

Cost spec: import `from pipeline.cost_model import COST_SPECS; spec = COST_SPECS["MNQ"]`.

Bar dataframe schema must match production: `ts_utc` (tz-aware UTC pandas Timestamp), `open`, `high`, `low`, `close`, `volume`, `symbol`. Scaffold from existing tests in `test_outcome_builder.py`.

---

## Companion drift check (Stage 5b, lands with code)

`pipeline/check_drift.py::check_orb_outcomes_scratch_pnl` (advisory: False, requires_db: True):

```python
def check_orb_outcomes_scratch_pnl(con=None) -> list[str]:
    """After the canonical fix lands, ≥99% of scratch rows must have non-NULL pnl_r.

    Pre-Stage-5 baseline: 0% of scratch rows had pnl_r populated.
    Post-Stage-5 target: ≥99%. The <1% gap is the pathological no-post-bars
    case explicitly documented in docs/specs/outcome_builder_scratch_eod_mtm.md.
    """
    if con is None:
        return _skip_message("DB unavailable")
    rows = con.execute(
        "SELECT COUNT(*), COUNT(pnl_r) FROM orb_outcomes WHERE outcome = 'scratch'"
    ).fetchone()
    total, populated = int(rows[0]), int(rows[1])
    if total == 0:
        return []
    pct_populated = 100.0 * populated / total
    if pct_populated < 99.0:
        return [
            f"  scratch rows with non-NULL pnl_r: {populated}/{total} = {pct_populated:.2f}% "
            f"(expected ≥ 99% post-Stage-5 fix; rebuild outcome_builder for affected instruments)"
        ]
    return []
```

This check is **advisory until Stage 5b rebuild completes**, then promoted to blocking.

---

## Backward compatibility

- **No schema change.** `outcome` enum unchanged (still `win|loss|scratch`). No new columns.
- **No migration script.** Stage 5b runs `python -m trading_app.outcome_builder --instrument <X> --force --orb-minutes <Y>` for each (instrument, aperture) combo. The `--force` flag drops + re-builds the affected rows under canonical idempotent DELETE+INSERT semantics.
- **Existing scratch rows with `pnl_r=NULL` get re-derived** during the rebuild. Post-rebuild, `WHERE pnl_r IS NOT NULL` filters now retain ≥99% of scratch rows (only the pathological no-post-bars edge stays NULL).

## Live execution path parity

The live execution stack (`trading_app/live/session_orchestrator.py`, `trading_app/live/projectx/order_router.py`, `trading_app/risk_manager.py`) is OUT OF SCOPE for this spec. Live execution already forces flat at session end via:
- `trading_app/risk_manager.py::F-1` XFA hard gate (TopStep prop-firm rule).
- Exchange-level futures EOD close (no carry of speculative micro-futures positions through settlement without explicit roll).

The fix to `outcome_builder.py` reconciles the BACKTEST with this live behavior. Per Chan 2013 Ch 1 unified-program doctrine, the two paths should book the same P&L on the same simulated state — which they did not, until this fix.

---

## Acceptance criteria for Stage 5 sign-off

Before marking Stage 5 complete:
1. `python -m pytest tests/test_trading_app/test_outcome_builder.py -v` — all 4 new tests green plus existing tests still green.
2. `python pipeline/check_drift.py` — passes (the new `check_orb_outcomes_scratch_pnl` will report SKIPPED until Stage 5b rebuild; once the rebuild lands it must pass at ≥99%).
3. Manual probe on a small instrument-aperture before full rebuild: confirm a single-day rebuild output produces non-NULL pnl_r on a known scratch case (regression smoke test).
4. Self-review per institutional-rigor.md § 1: read the diff, trace the execution path through one win path, one loss path, one scratch path, and one `post_entry.empty` path. Note any edge cases that fail.

## Open questions for user review

The user-review gate before Stage 5 covers:
1. **Edge-case decision (Sites A/C, no-post-bars):** keep NULL with comment vs. add `scratch_subtype` column. Spec recommends keep-NULL; user OK?
2. **Cost-function selection:** spec uses `to_r_multiple` (deducts friction); plan suggested `pnl_points_to_r` (no friction). Spec rationale: scratches are real round-trips with real costs. User OK with this correction?
3. **Test plan adequacy:** 4 tests sufficient? User wants more (e.g., MGC / MES instrument coverage, time-stop interaction)?
4. **Drift-check threshold:** ≥99% population. User OK with 1% pathological tolerance?
