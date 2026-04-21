# 2026-04-21 Reset Snapshot

Phase A only. Read-only audit snapshot for the 2026-04-21 deployment-first reset.

## Scope and evidence policy

- This snapshot verifies live claims against canonical repo sources in this session.
- `gold-db` MCP was requested by the directive but is **not available in this session**:
  - `list_mcp_resources -> []`
  - `list_mcp_resource_templates -> []`
- Therefore the live evidence below uses the repo's canonical local surfaces instead:
  - `gold.db` via `pipeline.paths.GOLD_DB_PATH` + `pipeline.db_config.configure_connection`
  - `trading_app.prop_profiles.ACCOUNT_PROFILES`
  - `trading_app.strategy_fitness`
  - `trading_app.sr_monitor`
  - `trading_app.live.broker_connections.connection_manager`
  - `pipeline/build_daily_features.py`
  - `trading_app/config.py`
  - `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`

## Headline truth-state

- The active live book is **6 lanes**, not 7.
- The active profile note is stale and overstates the current lane count.
- `pit_range_atr` is **not** a 0%-populated dead column. Live non-null rate is **42.54%** and it is still registered in `ALL_FILTERS`.
- The six live lanes are all currently **FIT** on rolling fitness, but this does **not** rescue their institutional status:
  - all 6 fail DSR > 0.95
  - all 6 fail MinBTL bounds by discovery-trial count
  - only 3/6 clear t >= 3.00, and 0/6 clear t >= 3.79
  - all 6 are holdout-contaminated by post-2026 discovery dates
- Stored Criterion 11/12 lifecycle state is **invalid** due DB identity mismatch.
- Fresh report-only SR monitoring shows **one current alarm**: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`.
- `rel_vol` research claims survive as **math**, but not as clean `E2` deployment framing:
  - PR #48 sizer numbers reproduce from canonical data
  - Q5 filter numbers reproduce from canonical data
  - canonical `VolumeFilter` explicitly marks `rel_vol` as `E2`-excluded because it depends on break-bar volume
- The old "sacred 124" object does **not** exist in the live DB state. Current `validated_setups` has **61 total rows** (`38 active`, `23 retired`).

## Evidence ledger

### E1. Session preflight / branch / clean state

Command:

```bash
python3 scripts/tools/session_preflight.py --context codex-wsl
git status --short --branch
git log --oneline -10
```

Output excerpt:

```text
Branch: research/pr48-sizer-rule-oos-backtest
HEAD: 9991695a5b537d769b4c9cf7e9824841ed352a9d
## research/pr48-sizer-rule-oos-backtest...origin/research/pr48-sizer-rule-oos-backtest
9991695a docs(handoff): record rel-vol filter-form outcome
96b9e358 audit(rel-vol): run locked filter-form validation
3df2acb1 audit(rel-vol): clear MGC sizer institutional stack
```

### E2. Active profile, live lane count, and stale profile note

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes
for pid, p in ACCOUNT_PROFILES.items():
    if p.active:
        print("PROFILE", pid, "lane_count=", len(effective_daily_lanes(p)))
        for lane in effective_daily_lanes(p):
            print(lane.strategy_id)
        print("NOTE:", p.notes)
PY
```

Output excerpt:

```text
PROFILE topstep_50k_mnq_auto lane_count= 6
MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5
MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15
MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5
MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12
MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12
MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15
NOTE: 7-lane MNQ auto profile — DYNAMIC ...
```

### E3. Current live-lane fitness states

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
import json
from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes
from trading_app.strategy_fitness import compute_fitness
for pid, p in ACCOUNT_PROFILES.items():
    if p.active:
        for lane in effective_daily_lanes(p):
            s = compute_fitness(lane.strategy_id)
            print(json.dumps({
                "strategy_id": s.strategy_id,
                "fitness_status": s.fitness_status,
                "rolling_exp_r": s.rolling_exp_r,
                "rolling_sample": s.rolling_sample,
                "recent_sharpe_30": s.recent_sharpe_30,
            }))
PY
```

Output excerpt:

```text
{"strategy_id": "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5", "fitness_status": "FIT", "rolling_exp_r": 0.1252, "rolling_sample": 374, "recent_sharpe_30": 0.5891824782260582}
{"strategy_id": "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15", "fitness_status": "FIT", "rolling_exp_r": 0.1689, "rolling_sample": 244, "recent_sharpe_30": 0.09859356937261894}
{"strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "fitness_status": "FIT", "rolling_exp_r": 0.2077, "rolling_sample": 359, "recent_sharpe_30": 0.16103730188330032}
{"strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "fitness_status": "FIT", "rolling_exp_r": 0.1546, "rolling_sample": 374, "recent_sharpe_30": 0.05049188510439462}
{"strategy_id": "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12", "fitness_status": "FIT", "rolling_exp_r": 0.1056, "rolling_sample": 317, "recent_sharpe_30": 0.22625883485474568}
{"strategy_id": "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "fitness_status": "FIT", "rolling_exp_r": 0.0037, "rolling_sample": 321, "recent_sharpe_30": 0.11851626607252314}
```

### E4. Filter fire rates, separated into discovery filter and execution overlay

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
import json
import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.db_config import configure_connection
from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes
from trading_app.config import ALL_FILTERS

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
configure_connection(con)
for pid, p in ACCOUNT_PROFILES.items():
    if not p.active:
        continue
    for lane in effective_daily_lanes(p):
        instrument, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, filter_type = con.execute(
            "SELECT instrument, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, filter_type FROM active_validated_setups WHERE strategy_id=?",
            [lane.strategy_id],
        ).fetchone()
        outcomes = con.execute(
            '''
            SELECT trading_day, entry_price, stop_price
            FROM orb_outcomes
            WHERE symbol=? AND orb_label=? AND orb_minutes=? AND entry_model=? AND rr_target=? AND confirm_bars=?
              AND outcome IN ('win','loss')
            ''',
            [instrument, orb_label, orb_minutes, entry_model, rr_target, confirm_bars],
        ).fetchall()
        feats = con.execute("SELECT * FROM daily_features WHERE symbol=? AND orb_minutes=?", [instrument, orb_minutes]).fetchall()
        cols = [d[0] for d in con.description]
        feat_map = {r[0]: dict(zip(cols, r, strict=False)) for r in feats}
        filt = ALL_FILTERS[filter_type]
        on = 0
        overlay_on = 0
        overlay_base = 0
        for td, entry_price, stop_price in outcomes:
            if filt.matches_row(feat_map[td], orb_label):
                on += 1
                if lane.max_orb_size_pts is not None:
                    overlay_base += 1
                    if abs(float(entry_price) - float(stop_price)) < lane.max_orb_size_pts:
                        overlay_on += 1
        print(json.dumps({
            "strategy_id": lane.strategy_id,
            "filter_type": filter_type,
            "discovery_universe_n": len(outcomes),
            "discovery_filter_on_n": on,
            "discovery_filter_fire_rate": round(on / len(outcomes), 4),
            "execution_overlay": "max_orb_size_pts",
            "execution_overlay_threshold": lane.max_orb_size_pts,
            "execution_overlay_pass_rate": round(overlay_on / overlay_base, 4),
        }))
PY
```

Output excerpt:

```text
{"strategy_id": "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5", "filter_type": "ORB_G5", "discovery_universe_n": 1790, "discovery_filter_on_n": 1655, "discovery_filter_fire_rate": 0.9246, "execution_overlay": "max_orb_size_pts", "execution_overlay_threshold": 39.0, "execution_overlay_pass_rate": 0.9619}
{"strategy_id": "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15", "filter_type": "ATR_P50", "discovery_universe_n": 1786, "discovery_filter_on_n": 967, "discovery_filter_fire_rate": 0.5414, "execution_overlay": "max_orb_size_pts", "execution_overlay_threshold": 37.8, "execution_overlay_pass_rate": 0.8469}
{"strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "filter_type": "ORB_G5", "discovery_universe_n": 1702, "discovery_filter_on_n": 1621, "discovery_filter_fire_rate": 0.9524, "execution_overlay": "max_orb_size_pts", "execution_overlay_threshold": 52.8, "execution_overlay_pass_rate": 0.9648}
{"strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "filter_type": "COST_LT12", "discovery_universe_n": 1764, "discovery_filter_on_n": 1740, "discovery_filter_fire_rate": 0.9864, "execution_overlay": "max_orb_size_pts", "execution_overlay_threshold": 117.8, "execution_overlay_pass_rate": 0.9741}
{"strategy_id": "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12", "filter_type": "COST_LT12", "discovery_universe_n": 1794, "discovery_filter_on_n": 1020, "discovery_filter_fire_rate": 0.5686, "execution_overlay": "max_orb_size_pts", "execution_overlay_threshold": 45.6, "execution_overlay_pass_rate": 0.9657}
{"strategy_id": "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "filter_type": "ORB_G5", "discovery_universe_n": 1550, "discovery_filter_on_n": 1546, "discovery_filter_fire_rate": 0.9974, "execution_overlay": "max_orb_size_pts", "execution_overlay_threshold": 94.9, "execution_overlay_pass_rate": 0.859}
```

### E5. `pit_range_atr` live null-rate and ALL_FILTERS registration

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.db_config import configure_connection
from trading_app.config import ALL_FILTERS
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
configure_connection(con)
print(con.execute("""
SELECT COUNT(*) AS total_rows,
       COUNT(pit_range_atr) AS non_null_rows,
       ROUND(100.0 * COUNT(pit_range_atr) / COUNT(*), 4) AS pct_non_null
FROM daily_features
""").fetchone())
print("PIT_MIN" in ALL_FILTERS, type(ALL_FILTERS["PIT_MIN"]).__name__)
PY
```

Output excerpt:

```text
(35112, 14938, 42.5439)
True PitRangeFilter
```

### E6. XFA / broker connection state

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
import json
from trading_app.live.broker_connections import connection_manager
connection_manager.load()
print(json.dumps(connection_manager.list_connections(), indent=2))
PY
```

Output excerpt:

```text
[
  {
    "broker_type": "projectx",
    "display_name": "TopStepX (from .env)",
    "enabled": true,
    "source": "env",
    "status": "disconnected",
    "connected_at": null,
    "account_count": 0
  },
  {
    "broker_type": "tradovate",
    "display_name": "Tradovate (from .env)",
    "enabled": false,
    "source": "env",
    "status": "disconnected",
    "connected_at": null,
    "account_count": 0
  }
]
```

### E7. Lifecycle-state validity and current SR monitor reality

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
from trading_app.lifecycle_state import read_lifecycle_state
state = read_lifecycle_state("topstep_50k_mnq_auto")
print(state["criterion11"])
print(state["criterion12"])
print(state["pauses"])
PY
./.venv-wsl/bin/python trading_app/sr_monitor.py
```

Output excerpt:

```text
{'available': True, 'valid': False, 'reason': 'db identity mismatch', 'gate_ok': False, ...}
{'valid': False, 'reason': 'db identity mismatch', ...}
{'paused_count': 0, 'paused_strategy_ids': []}

Lane                                        N         SR      Thr Status
L1 EUROPE_FLOW ORB_G5                      22       2.02    31.96 CONTINUE
L2 SINGAPORE_OPEN ATR_P50                   4       1.90    31.96 CONTINUE
L3 COMEX_SETTLE ORB_G5                     16       5.76    31.96 CONTINUE
L4 NYSE_OPEN COST_LT12                     21      33.27    31.96 ALARM
L5 TOKYO_OPEN COST_LT12                    20       6.63    31.96 CONTINUE
L6 US_DATA_1000 ORB_G5                      5       0.63    31.96 CONTINUE
```

### E8. Live `validated_setups` distribution

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
import duckdb, json, math
from pipeline.paths import GOLD_DB_PATH
from pipeline.db_config import configure_connection
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
configure_connection(con)
rows = con.execute("""
SELECT strategy_id, status, sample_size, sharpe_ratio, wfe, dsr_score, n_trials_at_discovery,
       wf_tested, wf_passed, oos_exp_r, all_years_positive
FROM validated_setups
""").fetchall()
summary = {
    "total_rows": len(rows),
    "active_rows": sum(1 for r in rows if r[1] == "active"),
    "retired_rows": sum(1 for r in rows if r[1] == "retired"),
    "wf_tested_true": sum(1 for r in rows if r[7] is True),
    "wf_passed_true": sum(1 for r in rows if r[8] is True),
    "oos_exp_nonnull": sum(1 for r in rows if r[9] is not None),
    "wfe_ge_05": sum(1 for r in rows if r[4] is not None and r[4] >= 0.5),
    "dsr_gt_095": sum(1 for r in rows if r[5] is not None and r[5] > 0.95),
    "trials_le_300": sum(1 for r in rows if r[6] is not None and r[6] <= 300),
    "trials_le_2000": sum(1 for r in rows if r[6] is not None and r[6] <= 2000),
    "all_years_positive_true": sum(1 for r in rows if r[10] is True),
}
print(json.dumps(summary, indent=2))
PY
```

Output excerpt:

```text
{
  "total_rows": 61,
  "active_rows": 38,
  "retired_rows": 23,
  "wf_tested_true": 61,
  "wf_passed_true": 61,
  "oos_exp_nonnull": 61,
  "wfe_ge_05": 61,
  "dsr_gt_095": 0,
  "trials_le_300": 0,
  "trials_le_2000": 0,
  "all_years_positive_true": 23
}
```

### E9. PR #48 sizer-rule live recheck

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
import duckdb, json
from pipeline.paths import GOLD_DB_PATH
from research.pr48_sizer_rule_oos_backtest_v1 import _run_instrument, _verdict
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
for inst in ["MNQ", "MES", "MGC"]:
    r = _run_instrument(con, inst)
    print(json.dumps({
        "instrument": inst,
        "n_oos": r.n_oos,
        "uniform_expr": round(r.mean_uniform, 5),
        "sizer_expr": round(r.mean_weighted, 5),
        "delta": round(r.delta, 5),
        "paired_t": round(r.paired_t, 3),
        "paired_p": round(r.paired_p, 4),
        "verdict": _verdict(r),
    }))
PY
```

Output excerpt:

```text
{"instrument": "MNQ", "n_oos": 771, "uniform_expr": 0.05895, "sizer_expr": 0.06522, "delta": 0.00627, "paired_t": 0.44, "paired_p": 0.3302, "verdict": "SIZER_WEAK"}
{"instrument": "MES", "n_oos": 702, "uniform_expr": -0.09022, "sizer_expr": -0.05997, "delta": 0.03025, "paired_t": 2.084, "paired_p": 0.0188, "verdict": "SIZER_ALIVE"}
{"instrument": "MGC", "n_oos": 601, "uniform_expr": 0.06955, "sizer_expr": 0.1013, "delta": 0.03175, "paired_t": 2.0, "paired_p": 0.023, "verdict": "SIZER_ALIVE"}
```

### E10. Q5 / rel_vol filter-form live recheck

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
import duckdb, json
from pipeline.paths import GOLD_DB_PATH
from research.rel_vol_filter_form_v1 import (
    _load_instrument, _train_thresholds, _apply_thresholds, _evaluate_form,
    FORMS, VAL_START, VAL_END, SEMI_OOS_START, SEMI_OOS_END, FRESH_OOS_START
)
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
for inst in ["MGC", "MES"]:
    full = _load_instrument(con, inst)
    thresholds, _ = _train_thresholds(full)
    kept = _apply_thresholds(full, thresholds)
    val_df = kept[(kept["trading_day"] >= VAL_START) & (kept["trading_day"] < VAL_END)]
    semi_df = kept[(kept["trading_day"] >= SEMI_OOS_START) & (kept["trading_day"] < SEMI_OOS_END)]
    fresh_df = kept[kept["trading_day"] >= FRESH_OOS_START]
    for form in FORMS:
        fr = _evaluate_form(val_df, semi_df, fresh_df, form)
        print(json.dumps({
            "instrument": inst,
            "form": form.key,
            "val_n": fr.val.n_total,
            "fire_n": fr.val.n_fire,
            "fire_rate": round(fr.val.fire_rate, 4),
            "filter_expr": round(fr.val.filter_expr, 5),
            "uniform_expr": round(fr.val.uniform_expr, 5),
            "filter_sr": round(fr.val.filter_sr, 4),
            "uniform_sr": round(fr.val.uniform_sr, 4),
            "delta_sr_ci_lo": None if fr.val.ci_lo is None else round(fr.val.ci_lo, 4),
            "delta_sr_ci_hi": None if fr.val.ci_hi is None else round(fr.val.ci_hi, 4),
            "verdict": fr.verdict,
        }))
PY
```

Output excerpt:

```text
{"instrument": "MGC", "form": "F1_Q5_only", "val_n": 4014, "fire_n": 673, "fire_rate": 0.1677, "filter_expr": 0.0812, "uniform_expr": -0.08987, "filter_sr": 0.0749, "uniform_sr": -0.0866, "delta_sr_ci_lo": 0.0954, "delta_sr_ci_hi": 0.2338, "verdict": "FILTER_ALIVE_IS_F1"}
{"instrument": "MES", "form": "F1_Q5_only", "val_n": 4841, "fire_n": 910, "fire_rate": 0.188, "filter_expr": 0.06352, "uniform_expr": -0.10122, "filter_sr": 0.0576, "uniform_sr": -0.0963, "delta_sr_ci_lo": 0.0984, "delta_sr_ci_hi": 0.2137, "verdict": "FILTER_ALIVE_IS_F1"}
```

### E11. rel_vol timing / execution validity

Command:

```bash
nl -ba pipeline/build_daily_features.py | sed -n '1537,1622p'
nl -ba trading_app/config.py | sed -n '682,754p'
```

Output excerpt:

```text
1539  # rel_vol_{label} = break_bar_volume / median(prior 20 bars_1m at same UTC minute-of-day).
1560  bvol_col = f"orb_{label}_break_bar_volume"
1592  bvol = row.get(bvol_col)
1621  row[rel_col] = round(bvol / baseline, 4)

730  """Relative-volume gate. Intra-session, resolves at BREAK_DETECTED.
732  E2-excluded: rel_vol includes break-bar volume, unknown at E2 entry.
750  "Relative-volume gate does not apply to E2 entries — break-bar volume is unknown at E2 order placement."
```

### E12. Live-lane timing-validity facts

Command:

```bash
nl -ba trading_app/config.py | sed -n '511,678p'
nl -ba trading_app/config.py | sed -n '1134,1200p'
nl -ba pipeline/build_daily_features.py | sed -n '1381,1394p'
```

Output excerpt:

```text
517  def matches_row(self, row: dict, orb_label: str) -> bool:
518      size = row.get(f"orb_{orb_label}_size")
548  ORB size is intra-session: it resolves at ORB_FORMATION

602  def matches_row(self, row: dict, orb_label: str) -> bool:
603      size = row.get(f"orb_{orb_label}_size")
640  Intra-session: resolves at ORB_FORMATION

1138 Reads atr_20_pct from daily_features (rolling 252d percentile, pre-computed).
1176 Own ATR-20 percentile gate. Pre-session, resolves at STARTUP.

1381 # ATR percentile: rank of today's ATR_20 among prior 252 trading days.
1383 # Prior-only window [i-252:i], no look-ahead.
1393 rows[i]["atr_20_pct"] = round(rank / len(sorted_prior) * 100, 2)
```

### E13. Holdout boundary and live-lane discovery dates

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
print(HOLDOUT_SACRED_FROM)
PY

./.venv-wsl/bin/python - <<'PY'
import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.db_config import configure_connection
from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
configure_connection(con)
for pid, p in ACCOUNT_PROFILES.items():
    if p.active:
        for lane in effective_daily_lanes(p):
            print(con.execute(
                "SELECT strategy_id, discovery_date, wf_tested, wf_passed, oos_exp_r FROM active_validated_setups WHERE strategy_id=?",
                [lane.strategy_id],
            ).fetchone())
PY
```

Output excerpt:

```text
2026-01-01

('MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5', date(2026, 4, 11), True, True, 0.1001)
('MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15', date(2026, 4, 13), True, True, 0.1236)
('MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5', date(2026, 4, 11), True, True, 0.1099)
('MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12', date(2026, 4, 11), True, True, 0.1003)
('MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', date(2026, 4, 11), True, True, 0.1188)
('MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15', date(2026, 4, 13), True, True, 0.0922)
```

### E14. COMEX_SETTLE x EUROPE_FLOW duplication

Command:

```bash
./.venv-wsl/bin/python - <<'PY'
import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.db_config import configure_connection
from trading_app.config import ALL_FILTERS
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
configure_connection(con)
def fired_days(strategy_id):
    inst, session, om, em, rr, cb, ft = con.execute(
        "SELECT instrument, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, filter_type FROM active_validated_setups WHERE strategy_id=?",
        [strategy_id],
    ).fetchone()
    feats = con.execute("SELECT * FROM daily_features WHERE symbol=? AND orb_minutes=?", [inst, om]).fetchall()
    cols = [d[0] for d in con.description]
    feat_map = {r[0]: dict(zip(cols, r, strict=False)) for r in feats}
    rows = con.execute(
        "SELECT trading_day, pnl_r FROM orb_outcomes WHERE symbol=? AND orb_label=? AND orb_minutes=? AND entry_model=? AND rr_target=? AND confirm_bars=? AND outcome IN ('win','loss')",
        [inst, session, om, em, rr, cb],
    ).fetchall()
    filt = ALL_FILTERS[ft]
    return {td: pnl for td, pnl in rows if filt.matches_row(feat_map[td], session)}
a = fired_days('MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5')
b = fired_days('MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5')
sa, sb = set(a), set(b)
inter = sa & sb
union = sa | sb
print({
    "comex_n": len(sa),
    "europe_n": len(sb),
    "overlap_n": len(inter),
    "overlap_vs_comex": round(len(inter)/len(sa), 4),
    "overlap_vs_europe": round(len(inter)/len(sb), 4),
    "jaccard": round(len(inter)/len(union), 4),
})
PY
```

Output excerpt:

```text
{'comex_n': 1621, 'europe_n': 1655, 'overlap_n': 1542, 'overlap_vs_comex': 0.9513, 'overlap_vs_europe': 0.9317, 'jaccard': 0.8893}
```

## A2 claim verdicts

| Claim | Status | Live audit result |
|---|---|---|
| Live 6-lane fitness states exist and should be checked per lane | CONFIRMED | There are 6 effective live lanes and all 6 are currently `FIT` on rolling fitness (`E2`, `O5/O15`, current DB). |
| Current live profile is a 7-lane dynamic book | CONTRADICTED | Effective live lane count is 6, while the profile note still says 7. The note is stale. |
| Filter fire rates per deployed strategy need discovery-vs-overlay separation | CONFIRMED | Live discovery fire rates range from `54.14%` to `99.74%`. Operational overlay (`max_orb_size_pts`) pass rates range from `84.69%` to `97.41%`. |
| `pit_range_atr` is 0%-populated but still registered | CONTRADICTED | Live `daily_features.pit_range_atr` is non-null on `14938 / 35112` rows = `42.5439%`. `PIT_MIN` still exists in `ALL_FILTERS`. |
| XFA / ProjectX is live-connected | CONTRADICTED | ProjectX credentials exist in env-backed broker config, but live connection state is `disconnected`, `connected_at=null`, `account_count=0`. Current repo/runtime state is dormant, not live-connected. |
| PR #48 participation-shape latest state still holds numerically | CONFIRMED | The sizer recheck reproduces `MNQ +0.00627`, `MES +0.03025`, `MGC +0.03175` with the same verdicts as the committed result doc. |
| PR #48 participation-shape is timing-valid as an `E2` pre-entry filter | CONTRADICTED | Canonical `rel_vol` uses `orb_*_break_bar_volume` and resolves at `BREAK_DETECTED`. That is post-break information and not valid as an `E2` pre-entry filter. |
| Q5 rel_vol latest state holds numerically | CONFIRMED | Live recheck reproduces `F1_Q5_only` as `FILTER_ALIVE_IS_F1` on both `MGC` and `MES` on the locked 2024-2025 validation window. |
| Q5 rel_vol is currently deployable as an `E2` filter | CONTRADICTED | Canonical `VolumeFilter` marks `rel_vol` as `E2`-excluded. Current framing is execution-invalid. |
| Sacred 124 distribution can be audited live in the current DB | CONTRADICTED | The live DB does not currently contain a 124-row `validated_setups` cohort. Current state is `61 total`, `38 active`, `23 retired`. Any "124" distribution is historical, not live. |
| ORB_G5 is degenerate on the active book | CONFIRMED | Active ORB_G5 lanes fire at `92.46%`, `95.24%`, and `99.74%` of eligible trades. |
| COMEX_SETTLE x EUROPE_FLOW are de facto duplicates | CONDITIONAL | They are near-duplicates in **fire calendar** (`88.93%` Jaccard, `95.13%` of COMEX days overlap) but not proven duplicates in return stream from this check alone. |

## A3 retroactive compliance matrix — six live lanes

The table below uses live `active_validated_setups` fields plus live fitness and SR-monitor outputs. Two important gaps are explicit:

- Exact `G3` Eq.9 `N̂` certificate is **not implemented canonically**. `trading_app/dsr.py` explicitly says ONC / proper `N_eff` is not yet implemented, and DSR remains informational.
- Exact per-lane `BHY` hurdle is **not canonically implemented** in the repo for these trade-level rows; any number here would require extra assumptions about monthly aggregation and volatility scaling. That is marked `UNVERIFIABLE`, not guessed.

| Lane | Fitness | t≈SR*sqrt(N) | t>=3.00 | t>=3.79 | WFE | WFE>=0.5 | DSR | DSR>0.95 | n_trials | <=300 | <=2000 | Holdout-clean | SR state | Verdict |
|---|---|---:|---|---|---:|---|---:|---|---:|---|---|---|---|---|
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | FIT | 2.528 | FAIL | FAIL | 2.8551 | PASS | 0.0000000004 | FAIL | 35616 | FAIL | FAIL | FAIL | CONTINUE | RESEARCH-PROVISIONAL |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | FIT | 2.928 | FAIL | FAIL | 1.4222 | PASS | 0.042151 | FAIL | 35700 | FAIL | FAIL | FAIL | CONTINUE | RESEARCH-PROVISIONAL |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | FIT | 3.717 | PASS | FAIL | 2.6151 | PASS | 0.0000005135 | FAIL | 35616 | FAIL | FAIL | FAIL | CONTINUE | RESEARCH-PROVISIONAL |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | FIT | 3.511 | PASS | FAIL | 1.9835 | PASS | 0.0000001230 | FAIL | 35616 | FAIL | FAIL | FAIL | ALARM | RESEARCH-PROVISIONAL + SR REVIEW |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | FIT | 3.400 | PASS | FAIL | 0.8225 | PASS | 0.000315 | FAIL | 35616 | FAIL | FAIL | FAIL | CONTINUE | RESEARCH-PROVISIONAL |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | FIT | 2.831 | FAIL | FAIL | 0.7008 | PASS | 0.002599 | FAIL | 35700 | FAIL | FAIL | FAIL | CONTINUE | RESEARCH-PROVISIONAL |

### A3 live summary

- `WFE >= 0.5`: **6 / 6 pass**
- `DSR > 0.95`: **0 / 6 pass**
- `t >= 3.00`: **3 / 6 pass**
- `t >= 3.79`: **0 / 6 pass**
- `MinBTL` discovery-budget bounds: **0 / 6 pass** the repo's current `<=300 clean / <=2000 proxy` operational ceilings
- `Holdout integrity`: **0 / 6 pass**
- `SR monitor`: **1 / 6 currently in ALARM** (`MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`)

## A4 timing-validity audit

### Live six-lane book

| Filter type | Active lanes | Source fields | Source timing | Timing validity for current framing |
|---|---|---|---|---|
| `ORB_G5` | EUROPE_FLOW, COMEX_SETTLE, US_DATA_1000(O15) | `daily_features.orb_{session}_size` | resolves at `ORB_FORMATION` | VALID |
| `COST_LT12` | NYSE_OPEN, TOKYO_OPEN | `orb_{session}_size` + `pipeline.cost_model.COST_SPECS` | resolves at `ORB_FORMATION` | VALID |
| `ATR_P50` | SINGAPORE_OPEN(O15) | `daily_features.atr_20_pct` from prior 252-day ATR rank | `STARTUP`, prior-only window | VALID |

### rel_vol candidate lineages

| Candidate | Source fields | Source timing | Audit result |
|---|---|---|---|
| PR #48 / PR #59 rel_vol sizer | `orb_{session}_break_bar_volume`, `orb_{session}_break_ts`, `rel_vol_{session}` | `BREAK_DETECTED` | INVALID as `E2` pre-entry filter framing; only valid as post-break role |
| Q5 rel_vol filter-form | same as above | `BREAK_DETECTED` | INVALID as `E2` pre-entry filter framing; valid only if reframed to post-break role |

## A5 SR / drift audit

- Stored lifecycle Criterion 11/12 state cannot be trusted right now because the state envelope is invalid (`db identity mismatch`).
- Live report-only SR monitor on current canonical DB shows:
  - 5 lanes `CONTINUE`
  - 1 lane `ALARM`: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
- Stored pause state is still empty (`paused_count=0`), so the current repo state is inconsistent:
  - the persisted lifecycle envelope is invalid
  - the live SR computation shows an alarm
  - no pause has been applied yet

## A6 holdout-integrity audit

### Six live lanes

- Sacred holdout boundary is live-canonical: `2026-01-01`.
- All 6 active live lanes have `discovery_date` after that boundary:
  - 4 lanes discovered `2026-04-11`
  - 2 lanes discovered `2026-04-13`
- All 6 also show `wf_tested = True`, `wf_passed = True`, and non-null `oos_exp_r`.
- Conclusion: the live six are **not Mode-A holdout-clean discoveries**. They are research-provisional / grandfathered style rows in current repo terms, not clean 2026-holdout evidence.

### PR #48 sizer lineage

- The pre-reg is locked and commit-pinned (`commit_sha: d227f8ed...`).
- The script recheck reproduces the committed numbers from canonical data.
- But the pre-reg itself declares upstream OOS provenance from the earlier 2026 OOS β₁ replication. So the sizer run is not a "never-seen-2026" first look at the participation theme; it is a bounded follow-on on a lineage that had already touched the 2026 OOS pattern.
- Result: **math confirmed, pure untouched-holdout status not confirmed**.

### Q5 rel_vol filter-form lineage

- The pre-reg explicitly declares `oos_peeked_window: 2026-01-01 to 2026-04-19`.
- It explicitly gates confirmatory status on fresh OOS from `2026-04-22+`.
- Current repo data max day is `2026-04-16`, so fresh OOS accrual is still `0`.
- Result: **timing honesty confirmed, deploy-confirmation not yet available**.

## Claims that live data contradicted

- `pit_range_atr` is rotten / 0%-populated.
- The live book is 7 lanes.
- There is a live-connected XFA / ProjectX runtime in this snapshot.
- A live "sacred 124" cohort still exists in the current DB.
- `rel_vol` can be treated as an `E2` pre-entry filter.

## Claims that live data confirmed

- There are currently 6 effective live lanes, and they can be audited per lane.
- ORB_G5 is degenerate on the active book.
- PR #48 sizer numbers reproduce on canonical data.
- Q5 filter-form numbers reproduce on canonical data.
- Current live lanes are timing-valid on their own filter inputs.

## Claims that remain unverified in Phase A

- Exact `G3` Eq.9 `N̂` certificates for the live lanes. Repo-canonical implementation gap remains.
- Exact per-lane `BHY` hurdle certificates for the live lanes. Repo-canonical implementation gap remains.
- A live 124-row criterion distribution. The object no longer exists in current `validated_setups`.

## Phase A stop point

Phase A is complete enough to proceed to Phase B, but per the directive this session should stop here and wait for explicit confirmation before issuing per-lane KEEP / DEGRADE / RETIRE decisions.
