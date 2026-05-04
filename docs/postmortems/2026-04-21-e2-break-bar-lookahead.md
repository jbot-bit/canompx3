# Postmortem: `backtesting-methodology.md` § RULE 6.1 Drifted From Canonical E2 Look-Ahead Policy

**Date:** 2026-04-21
**Severity:** MEDIUM — doctrine drift, not a production bug. Code was correct; a doctrine document listed break-bar columns as "safe for E2 CB1" which directly contradicts the canonical E2-exclusion enforced by `trading_app/config.py`. Research authors reading the doc in isolation could build look-ahead scans on E2 data.
**Resolution:** Narrow `§ RULE 6.1` to remove `orb_{s}_break_delay_min` / `orb_{s}_break_bar_continues` / `orb_{s}_break_bar_volume` from the safe list; surface them explicitly in `§ RULE 6.3` with E2 caveat and canonical source cites.
**Detection:** Real-data verification on 2026-04-21 (MNQ EUROPE_FLOW E2 CB1 O5 RR1.5, full pre-holdout IS). Triggered by a pre-reg drafting session that attempted to rely on RULE 6.1 wording.
**Related prior postmortem:** `docs/postmortems/2026-04-07-e2-canonical-window-fix.md` (same underlying E2 mechanic — range-touch vs close-break — measured at 35.5% in that broader audit).

---

## TL;DR

The canonical code comment at `trading_app/config.py:3540-3568` states plainly:

> E2 enters on the FIRST bar whose range touches the ORB boundary after the ORB window closes. On ~42-49% of break-days, this bar is a fakeout (closes back inside) that precedes the confirmed close-based break. Filters that reference break-bar properties (volume, continuation, delay) are therefore look-ahead for E2 — the values are not knowable at E2 entry time.

And enforces this via `E2_EXCLUDED_FILTER_PREFIXES = ("VOL_RV", "ATR70_VOL")` and `E2_EXCLUDED_FILTER_SUBSTRINGS = ("_CONT", "_FAST", "NOMON_CONT")`, gated in `_e2_look_ahead_reason()` and consumed by `BreakSpeedFilter.describe()` (line 2163) and `BreakBarContinuesFilter.describe()` (line 2262). Registered E2 strategies using `BRK_FAST*`, `BRK_CONT`, `VOL_RV*`, or `ATR70_VOL` resolve to `NOT_APPLICABLE_ENTRY_MODEL` and are not deployed.

Meanwhile `.claude/rules/backtesting-methodology.md § RULE 6.1` listed:

> `orb_{s}_break_delay_min`, `orb_{s}_break_bar_continues`, `orb_{s}_break_bar_volume` (known at break-bar close, before E2 CB1 entry)

as SAFE features. A research author who read RULE 6.1 without reading `config.py:3540-3568` could write a scan that uses these columns directly on E2 outcomes (bypassing the filter framework) and believe they were lookahead-clean.

This PR narrows § 6.1 and expands § 6.3, cites the canonical source, and publishes a real-data corroboration.

---

## 1. Real-data measurement

Against canonical `gold.db`, 2026-04-21. Query: all `orb_outcomes` rows where
`symbol='MNQ' AND orb_label='EUROPE_FLOW' AND orb_minutes=5 AND entry_model='E2'
AND confirm_bars=1 AND rr_target=1.5 AND trading_day < '2026-01-01' AND entry_ts IS NOT NULL`,
joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.

| Year | N | entry_ts == break_ts | entry_ts < break_ts (LEAK) | entry_ts > break_ts |
|------|---|----------------------|-----------------------------|----------------------|
| 2019 | 171 | 107 | 64 (37.4%) | 0 |
| 2020 | 256 | 143 | 113 (44.1%) | 0 |
| 2021 | 259 | 158 | 101 (39.0%) | 0 |
| 2022 | 258 | 162 | 96 (37.2%) | 0 |
| 2023 | 258 | 153 | 105 (40.7%) | 0 |
| 2024 | 259 | 154 | 105 (40.5%) | 0 |
| 2025 | 257 | 132 | 125 (48.6%) | 0 |
| **TOTAL** | **1,718** | **1,009** | **709 (41.3%)** | **0** |

**Per-year leak rate is stable 37.4%-48.6%, consistent with the canonical comment's "~42-49%" range.** No year is an outlier. The 2026-04-07 postmortem measured 35.5% across all E2 lanes — the higher L1 figure is within expected noise for a single lane.

---

## 2. Root cause — two definitions of "break bar"

### 2.1 E2 entry (range-touch)

`trading_app/entry_rules.py:157-216 detect_break_touch()`:

```python
if break_dir == "long":
    touch_mask = candidate_bars["high"].values > orb_high
else:
    touch_mask = candidate_bars["low"].values < orb_low
# ...
return BreakTouchResult(touched=True, touch_bar_ts=ts, ...)
```

E2 (stop-market) fires the moment intra-bar range crosses the ORB boundary — **wick-only touches count, including fakeouts where the bar closes back inside**. `entry_ts` is the open time of the touch bar.

### 2.2 `daily_features.orb_{s}_break_*` (close-cross)

`pipeline/build_daily_features.py:285-340 detect_break()`:

```python
for bar in window_bars.itertuples():
    close = float(bar.close)
    if close > orb_high:
        delay = (bar_ts - orb_end).total_seconds() / 60.0
        return {"break_dir": "long", "break_ts": bar_ts, "break_delay_min": delay,
                "break_bar_continues": close > bar_open,
                "break_bar_volume": int(bar.volume)}
```

`break_ts`, `break_delay_min`, `break_bar_continues`, `break_bar_volume` are all defined on the first bar whose **close** is outside the ORB. For fakeout bars (range-touch, close back inside), this bar is a **later** bar than the E2 touch bar.

### 2.3 Consequence for E2 research

For ~41% of E2 entries, `break_*` columns in `daily_features` reflect a bar that arrived AFTER the E2 stop-market was already filled. Treating them as predictors = using post-entry information.

E1 and E3 (confirm-bar entries) enter AFTER the break bar closes and are unaffected — the break-bar columns are correctly pre-entry for those entry models.

---

## 3. What production enforces today (unchanged)

`trading_app/config.py`:

- Line 3560-3568: `E2_EXCLUDED_FILTER_PREFIXES = ("VOL_RV", "ATR70_VOL")`, `E2_EXCLUDED_FILTER_SUBSTRINGS = ("_CONT", "_FAST", "NOMON_CONT")`.
- Line 149-180: `_e2_look_ahead_reason(filter_type)` returns a human-readable reason string for any filter matching the prefix/substring lists.
- Line 2163-2185: `BreakSpeedFilter.describe()` — on E2, returns `is_not_applicable=True, not_applicable_reason=...`, so the strategy validator and deployment pipeline never activate a `BRK_FAST*` filter on an E2 lane.
- Line 2262-2290: `BreakBarContinuesFilter.describe()` — same pattern for `BRK_CONT`.
- Line 3585-3615: `E2_ORDER_TIMEOUT` comment block — explains the *correct* way to approximate "fast break" as an E2 strategy: an **execution-time order timeout** (cancel the E2 stop after N minutes if it hasn't filled), which is real-time-knowable rather than look-ahead. Dormant pending broker GTD support.

The production path is fail-closed. The doctrine doc drifted from it.

---

## 4. What this PR changes

### 4.1 `.claude/rules/backtesting-methodology.md § RULE 6.1`

Before:

```
- `orb_{s}_size`, `orb_{s}_high`, `orb_{s}_low`, `orb_{s}_break_dir` (known at ORB end, before entry for E2)
- `orb_{s}_break_delay_min`, `orb_{s}_break_bar_continues`, `orb_{s}_break_bar_volume` (known at break-bar close, before E2 CB1 entry)
```

After:

```
- `orb_{s}_size`, `orb_{s}_high`, `orb_{s}_low` (known at ORB end, before entry for E2)
```

(break-bar bullet deleted; `break_dir` removed from the safe clause because direction-as-predictor has the same E2 look-ahead shape for fakeout bars.)

### 4.2 `.claude/rules/backtesting-methodology.md § RULE 6.3`

Adds an E2 sub-section enumerating the now-banned columns with reasoning, the 41.3% real-data figure, canonical authority cites (`config.py:3540-3568`, `entry_rules.py:157-216`, `build_daily_features.py:285-340`), and a pointer to this postmortem.

### 4.3 No code change

`trading_app/config.py` is already correct. `mechanism_priors.md § 4` is already correct. `pipeline/check_drift.py` does not currently check the research-layer use of these columns; see § 5.4 follow-up.

---

## 5. Follow-up items (not in this PR)

These are surfaced, not resolved. Each is a discrete next task requiring its own audit.

### 5.1 Research-script retro-audit

Grep hits for `break_delay_min` / `break_bar_continues` / `break_bar_volume` in `research/**` (excluding canonical pipeline / test files / doc files / `chatgpt_bundle/`):

```
research/q1_h04_mechanism_shape_validation_v1.py
research/shadow_htf_mes_europe_flow_long_skip.py
research/volume_confluence_scan.py
research/t0_t8_audit_volume_cells.py
research/update_forward_gate_tracker.py
research/research_wf_stress_keepers.py
research/research_wide_non_leadlag_composite.py
research/research_universal_hypothesis_pool.py
research/research_shinies_overlay_stack_presets.py
research/research_shinies_universal_overlays.py
research/research_shinies_bqs_overlay_tests.py
research/research_round_number_proximity.py
research/research_mgc_e2_microstructure_pilot.py
research/research_false_breakout_bqs_tests.py
research/comprehensive_deployed_lane_scan.py
research/break_delay_nuggets.py
research/break_delay_filtered.py
(plus archive/)
```

Each one needs a per-script audit question: **does it segment or filter E2 outcomes by a break-bar column as a predictor?** If yes, any survivor it reported is suspect and needs a rerun gated on `entry_model != 'E2'` or `entry_ts >= break_ts`.

This postmortem does NOT claim any of the above scripts are wrong — some may only consume these columns for descriptive diagnostics (legitimate) or for E1/E3 lanes (safe). The retro-audit is a separate work item.

### 5.2 `orb_{s}_break_dir` as a predictor

`break_dir` is set by `detect_break()` on close-break, not range-touch. For range-touch-then-reverse bars where E2 filled long and the subsequent close-break was the short side, `daily_features.break_dir` disagrees with E2's fill direction. Needs its own real-data audit (how many trades? which direction does the disagreement resolve toward?). For now, `break_dir` is conservatively treated as E2-look-ahead in RULE 6.3 when used as a *predictor* for E2 trade selection. Segmenting an already-fired trade by its own fill direction (pulled from E2 fill metadata) is fine — that is a different operation.

### 5.3 Drift-check addition

Consider adding a `pipeline/check_drift.py` check that any `validated_setups` / `experimental_strategies` row whose `strategy_id` encodes `entry_model=E2` must NOT include `BRK_FAST*`, `BRK_CONT`, `VOL_RV*`, or `ATR70_VOL` in its filter composite. Today this is enforced at the deployment pipeline via `_e2_look_ahead_reason()`, but a terminal-state drift check would catch any future regressions in discovery/validation code. Out of scope for this PR.

### 5.4 Static check for research scripts

Consider adding `.pre-commit` hook that greps `research/**` for `break_delay_min` / `break_bar_continues` / `break_bar_volume` / `break_ts` references, requires a `# E2-SAFE:` or `# E1/E3-ONLY:` comment within 5 lines, and fails CI otherwise. Out of scope for this PR.

---

## 6. Why this matters

If a research author uses the deprecated RULE 6.1 wording to justify a scan, the scan will:

1. Pass trade-time-knowability RULE 6 review by self-citation (the doc said it was safe).
2. Produce E2 survivors that appear to have a real edge, because break-bar features ARE predictive of outcome (they encode post-entry direction).
3. Fail silently — neither `pipeline/check_drift.py` nor the filter framework catches it, because the scan bypassed `StrategyFilter` by reading `daily_features` columns directly in SQL or pandas.

The fix here prevents future occurrences. § 5.1 identifies the retro-audit surface.

---

## 7. Verification

- Real-data query: reproducible via the table in § 1 (single SQL statement, canonical `gold.db`).
- Code-level corroboration: `trading_app/config.py:3540-3568` comment matches my real-data measurement within tolerance.
- Drift check: `python pipeline/check_drift.py` passes on the branch post-edit (pre-existing `anthropic` import gap unchanged).

---

## 8. Binding decisions

1. `orb_{s}_break_delay_min`, `orb_{s}_break_bar_continues`, `orb_{s}_break_bar_volume` are **look-ahead for E2** and are banned as predictors on E2 entries. They remain valid for E1/E3 (where entry is post-break-bar-close by construction).
2. `orb_{s}_break_dir` and `orb_{s}_break_ts` are **look-ahead when used as predictors for E2 trade selection**. Using them as descriptive segmentation of an already-taken trade (e.g., "what was the eventual close-break direction on days when E2 filled long?") is fine as a diagnostic but not as a filter.
3. `.claude/rules/backtesting-methodology.md` is now aligned with `trading_app/config.py:3540-3568` and `docs/institutional/mechanism_priors.md § 4`.
4. Research scripts flagged in § 5.1 remain unmodified by this PR. Their retro-audit is a follow-up work item.

---

## 9. 2026-04-28 follow-up — § 5.1 retro-audit completed

The drift check `pipeline/check_drift.py::check_e2_lookahead_research_contamination` (landed 2026-04-28 commit `2c91d9d1`) replaces the proposed § 5.4 static check. On first run it surfaced **9 additional research scripts** beyond the original 18-file registry — these are the § 5.1 retro-audit closure list.

**Two doctrine drifts found and fixed in the closure pass:**

1. § 6.1 of `backtesting-methodology.md` still listed `rel_vol_{s}` as a safe predictor with the description "ORB volume vs session-avg historical — known at ORB end." The canonical computation at `pipeline/build_daily_features.py:1600-1660` is `rel_vol_{label} = break_bar_volume / median(prior 20 bars_1m at same UTC minute-of-day)` — the numerator is `break_bar_volume`, so the same 41.3% post-entry class bug applies. § 6.1 narrowed and § 6.3 extended on 2026-04-28 to cover this gap.
2. The `late-fill-only` annotation grammar permitted by the drift check is statistically unsafe for any signal-discovery purpose: filtering trades on `entry_ts >= break_ts` conditions the backtest universe on a future-bar timestamp (Chan Ch 1 p.4 — using future information). The grammar still allows the annotation, but per-script application now requires an inline reminder that the late-fill subset is selection-biased and not deployable.

**Closure verdicts** for the 9 candidates are catalogued in `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` § "Drift-check sweep additions (2026-04-28)". This closes the § 5.1 deferred audit.
