# 2026-04-19 Filter-Delegation Campaign Handover

**For:** next-session continuation in a fresh context terminal
**Sessions covered:** 2026-04-19 MAX-EV extraction campaign (filter-delegation sub-track)
**Last handover author:** jbot-bit session (context near-limit)

## Campaign goal

Harden research/ scripts by eliminating inline canonical-filter SQL and hardcoded commission constants. Per `.claude/rules/research-truth-protocol.md` § Canonical filter delegation (added 2026-04-18). Canonical rule grounded in `docs/institutional/literature/lopez_de_prado_bailey_2018_false_strategy.md:29` Theorem 1 + `:65` conclusion (parallel-model drift inflates effective K → amplifies false positives) and `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md:26` ("worthless without search-extent control").

## Open PRs (state as of handover)

| PR | Branch | Scope | Status | Commits |
|---|---|---|---|---|
| [#13](https://github.com/jbot-bit/canompx3/pull/13) | `chore/filter-delegation-comex-battery` | File 1 — `garch_comex_settle_institutional_battery.py` | **Ready to merge** — all 6 filter families parity-verified (ATR_P50 N=440, OVNRNG_100 N=276, ORB_G5 EUROPE N=708, ORB_G5 TOKYO N=716, VWAP_MID_ALIGNED long N=381 / short N=303) | `58b066b4` |
| [#14](https://github.com/jbot-bit/canompx3/pull/14) | `chore/filter-delegation-validated-role-exhaustion` | File 2 — `garch_validated_role_exhaustion.py` | **Ready to merge** — surfaces real COST_LT12 slippage-correction (693→691 = structural fix, not defect). Includes F2 dtype-safe enrichment + F3 regression guard (3 pytest assertions). | `9baa3b50` → `227aff87` → `58dd2bd4` |
| [#15](https://github.com/jbot-bit/canompx3/pull/15) | `chore/filter-delegation-broad-role-exhaustion` | File 3 — `garch_broad_exact_role_exhaustion.py` | **Ready to merge** — 15 filter families delegated. `exact_filter_sql` preserved DEPRECATED for 3 external callers. Parity inherited structurally from PRs #13/#14. | `59387b3b` → `840536d7` (scope-bleed correction) |
| [#12](https://github.com/jbot-bit/canompx3/pull/12) | user's | A2b-2 Shape E | CLEAN — user's work |
| [#11](https://github.com/jbot-bit/canompx3/pull/11) | user's | check_drift #37 canonical-delegation | user's work |
| [#10](https://github.com/jbot-bit/canompx3/pull/10) | user's | audit_behavioral SQL detection tighten | user's work |
| [#9](https://github.com/jbot-bit/canompx3/pull/9) | `chore/ruff-format-sweep` | 398-file ruff format sweep | Blocked on CI-DB fix (user's parallel terminal) |
| [#8](https://github.com/jbot-bit/canompx3/pull/8) | user's | A2b-2 Shape E diagnostic DSR | user's work |

## Main-branch commits this session

- `aad612e6` — SR monitor regression test (captures Phase 3.1 F-3 insight: `run_monitor` reports SR at alarm-trigger, not post-recovery stream end)
- `26751084` — Filter-delegation audit addendum (broader-regex findings: 3 new HIGH/MEDIUM offenders caught by original narrow sweep)
- `7694eb18` — Filter-delegation audit addendum 2 (hardcoded-commission sweep: 4 more files with stale `$2.74`)

## Remaining filter-delegation queue (priority order)

| Priority | File | Class | Notes |
|---|---|---|---|
| HIGH | `research/garch_validated_scope_honest_test.py` | Same COST_LT12 `(2.74 / risk_dollars) < 0.12` inline as File 2 before fix. Apply PR #14 pattern. | Will surface same ~2-row divergence on parity test (semantic correction via slippage inclusion). |
| HIGH (dep) | `research/carry_encoding_exploration.py` | Imports `broad.exact_filter_sql` | Migrate to `filter_signal` to enable eventual deletion of `exact_filter_sql` from File 3. |
| HIGH (dep) | `research/garch_additive_sizing_audit.py` | Same as above | Same migration. |
| HIGH (dep) | `research/garch_carry_collinearity_check.py` | Same as above | Same migration. |
| MED | `research/research_prop_firm_fit.py` | Hardcoded `{"MNQ": 2.74, ...}` commission dict at line 57 | Design-gate first: confirm whether file is still consumed by any CI / dashboards / other research scripts. If yes, rewrite to source from `pipeline.cost_model.COST_SPECS`. If no, deprecate with top-of-file note. |
| LOW | `research/tmp_dd_anatomy.py`, `research/tmp_dd_budget_configs.py` | `tmp_` scratch; pre-Rithmic FRICTION dict | Propose deletion — `tmp_` prefix is project convention for non-canonical. If outputs referenced anywhere, migrate first. |

After all above: delete `exact_filter_sql` from File 3, extend `DELEGATED_FILES` list in `tests/test_research/test_filter_delegation_guard.py` to include all migrated files.

## Review action items closed on-branch

Per the code-review pass (B+ grade → A- after these):
- **F1** (PR #13): VWAP_MID_ALIGNED parity gap closed. Both long (N=381) + short (N=303) match canonical. All 6 filter families now parity-verified in PR body.
- **F2** (PR #14): Dtype-safe cross-asset enrichment. Direct column-map replaces lossy dict round-trip. `cross_atr_MES_pct` now `float64` (was `object`). X_MES_ATR60 N=278 parity preserved.
- **F3** (PR #14): Regression pytest guard at `tests/test_research/test_filter_delegation_guard.py`. Static test (no DB). 3 assertions per delegated file: imports `filter_signal`, no inline filter SQL, no hardcoded `$2.74` commission. 3/3 pass.
- **F4** (deferred): py3.13 f-string nested-quote at File 2 line 385 — prevents py3.11 import. Out-of-scope for delegation PRs. Needs tiny standalone PR.

## Critical institutional findings surfaced this campaign

1. **Canonical `CostRatioFilter` formula** (trading_app/config.py:602-613):
   ```
   raw_risk = size * cost_spec.point_value
   cost_ratio_pct = 100 * total_friction / (raw_risk + total_friction)
   ```
   Uses TOTAL friction (commission + slippage). Inline `(commission / risk_dollars)` patterns in research scripts OMIT slippage → systematic under-count of cost.

2. **Pre-Rithmic commission drift:** Current canonical `COST_SPECS['MNQ'].commission_rt = $1.42`. Many research files still hardcode pre-Rithmic `$2.74`. Identified in audit addendum 2.

3. **`CrossAssetATRFilter` round-trip dtype loss:** `df.to_dict('records') → fn(...) → pd.DataFrame(feat_dicts)` collapses numpy dtypes to object. Fitness tracker path tolerates this (it iterates dicts). Research scan path breaks silently because subsequent `filter_signal` calls on other filters could receive object-dtype columns. Fix pattern: direct column-map assignment.

4. **SR monitor alarm-report semantic** (captured in `tests/test_trading_app/test_sr_monitor.py` commit `aad612e6`): `run_monitor` breaks at first-crossing, persists alarm-trigger `sr_stat`. Any path-walk reconstruction that continues through the whole stream reports a different (smaller) value. Both are mathematically correct; they answer different questions. Treat as reporting-mode distinction, not a bug.

5. **Mode A drift universal** (not documented here — on user's campaign branch commit `122af101`): all 38 active `validated_setups` lanes show material drift from stored `expectancy_r` to canonical Mode A IS. Allocator should not trust stored ExpR for allocation decisions until Phase 3.2 lands.

## Other unresolved items

- **CI drift 6-check failures** (checks 37, 61, 80, 92, 95, 96): user's parallel terminal working on DB-fixture decision. Do NOT engage here — risk of divergent work.
- **Local main 1 commit ahead of origin/main** (wait — re-verify with `git status`; state should be sync'd per session close-out).
- **Worktrees** (last verified):
  - `C:/Users/joshd/canompx3` → main
  - `C:/Users/joshd/canompx3/.worktrees/campaign-2026-04-19-phase-2` → research/campaign-2026-04-19-phase-2 (user's parallel work)
  - `C:/Users/joshd/canompx3/.worktrees/mnq-nyse-close-long-direction-locked-v1` → research/mnq-nyse-close-long-direction-locked-v1
  - `C:/Users/joshd/canompx3-f5` → chore/ruff-format-sweep
  - `C:/Users/joshd/canompx3-htf` → research/htf-path-a-design
  - `C:/Users/joshd/canompx3-phase-d` → phase-d-volume-pilot-d0

## Next session start checklist

1. `git fetch origin && git status` — confirm main synced
2. `git log --oneline origin/main~5..origin/main` — verify merges landed
3. `gh pr list --state open` — see which of #13/#14/#15 are still open
4. If PRs merged → extend `tests/test_research/test_filter_delegation_guard.py::DELEGATED_FILES` to include the merged files
5. If PRs not merged → resume here, identify blockers
6. Pick next queue item (`garch_validated_scope_honest_test.py` is top HIGH priority)

## Canonical patterns (copy-paste for next File)

### Pattern for load_trades fix

```python
from research.filter_utils import filter_signal
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter

def load_trades(con, row, direction, *, is_oos):
    filter_type = row["filter_type"]
    if filter_type not in ALL_FILTERS:
        print(f"  ERR unknown filter_type '{filter_type}' ...")
        return pd.DataFrame()

    # Load d.* so filter_signal sees every column any canonical filter may need
    df = con.execute(...).df()
    if len(df) == 0: return df

    # Cross-asset enrichment (dtype-safe)
    filt = ALL_FILTERS[filter_type]
    if isinstance(filt, CrossAssetATRFilter):
        src = filt.source_instrument
        atr_rows = con.execute(
            "SELECT trading_day, atr_20_pct FROM daily_features "
            "WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL",
            [src]
        ).fetchall()
        atr_map = {(td.date() if hasattr(td, "date") else td): pct for td, pct in atr_rows}
        df[f"cross_atr_{src}_pct"] = df["trading_day"].apply(
            lambda t: t.date() if hasattr(t, "date") else t
        ).map(atr_map)

    # Canonical filter application
    mask = np.asarray(filter_signal(df, filter_type, row["orb_label"])).astype(bool)
    df = df.loc[mask].reset_index(drop=True)
    ...
```

### Pattern for parity verification

```python
PYTHONPATH=. python -c "
import duckdb, numpy as np
from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
# Load df + apply filter_signal + compare N against inline SQL
...
"
```

---

**End of handover.** Good luck next session.
