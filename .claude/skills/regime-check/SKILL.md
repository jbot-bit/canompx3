---
name: regime-check
description: Check portfolio fitness and regime health across all instruments
allowed-tools: Read, Grep, Glob, Bash
---
Check portfolio fitness and regime health: $ARGUMENTS

Use when: "fitness", "regime", "decay", "how's the portfolio", "strategy health"

## Step 1: Query Fitness

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

print('=== REGIME SUMMARY ===')
# Family linkage is via family_hash (every member of a family shares the
# same hash). Joining on ef.head_strategy_id only returns the one head per
# family, so non-head family members are mis-labeled NO_FAMILY. That bug
# was caught on 2026-04-19 when an audit reported "17 MNQ unlinked" that
# were actually all classified family members at non-winning RRs. Always
# join on family_hash for linkage counts; use head_strategy_id only when
# specifically isolating the head.
print(con.sql('''
    SELECT vs.instrument, COALESCE(ef.robustness_status, 'NO_FAMILY') as regime_status,
           COALESCE(ef.trade_tier, 'NONE') as tier, COUNT(*) as count
    FROM validated_setups vs
    LEFT JOIN edge_families ef ON vs.family_hash = ef.family_hash
    WHERE LOWER(vs.status) = 'active'
    GROUP BY vs.instrument, regime_status, tier ORDER BY vs.instrument, regime_status
''').fetchdf().to_string(index=False))

print('\n=== FAMILY HEADS ONLY (one row per family; useful for allocator debug) ===')
print(con.sql('''
    SELECT vs.instrument, vs.orb_label, vs.orb_minutes, vs.rr_target,
           vs.entry_model, vs.filter_type,
           ef.trade_tier, ef.robustness_status, ef.member_count,
           ROUND(ef.head_expectancy_r, 4) AS head_expr
    FROM edge_families ef
    JOIN validated_setups vs ON vs.strategy_id = ef.head_strategy_id
    WHERE LOWER(vs.status) = 'active'
    ORDER BY vs.instrument, ef.head_expectancy_r DESC
''').fetchdf().to_string(index=False))

print('\n=== EDGE FAMILIES ===')
print(con.sql('''
    SELECT instrument, trade_tier, COUNT(*) as families,
           ROUND(AVG(head_expectancy_r), 4) as avg_expr, MIN(min_member_trades) as min_trades
    FROM edge_families GROUP BY instrument, trade_tier ORDER BY instrument, trade_tier
''').fetchdf().to_string(index=False))

last_built = con.sql('SELECT MAX(created_at) FROM edge_families').fetchone()[0]
print(f'\nEdge families last built: {last_built}')
import datetime
if last_built:
    age_days = (datetime.datetime.now(datetime.timezone.utc) - last_built.replace(tzinfo=datetime.timezone.utc)).days if hasattr(last_built, 'replace') else None
    if age_days and age_days > 30:
        print(f'  WARNING: {age_days} days old — STALE')
    elif age_days is not None:
        print(f'  ({age_days} days old — FRESH)')

print('\n=== INSTRUMENT HEALTH SUMMARY ===')
print(con.sql('''
    SELECT instrument,
           SUM(CASE WHEN trade_tier='CORE' THEN 1 ELSE 0 END) as core_families,
           SUM(CASE WHEN trade_tier='REGIME' THEN 1 ELSE 0 END) as regime_families,
           COUNT(*) as total_families,
           ROUND(AVG(head_expectancy_r), 4) as avg_expr
    FROM edge_families GROUP BY instrument ORDER BY instrument
''').fetchdf().to_string(index=False))

print('\n=== ORB SIZE TREND (all MNQ sessions, recent 6mo vs prior 6mo) ===')
# Recent ORB size expansion flagged in 2026-04-19 session: 5 of 6 MNQ sessions
# up 18-45% vs prior 6mo. Keep all 6 visible, not just CME_PRECLOSE + NYSE_OPEN.
print(con.sql('''
    SELECT CASE WHEN trading_day >= CURRENT_DATE - 180 THEN 'recent_6mo' ELSE 'prior_6mo' END as period,
           ROUND(AVG(orb_TOKYO_OPEN_size),2) as tokyo,
           ROUND(AVG(orb_EUROPE_FLOW_size),2) as europe,
           ROUND(AVG(orb_US_DATA_830_size),2) as us830,
           ROUND(AVG(orb_NYSE_OPEN_size),2) as nyse_o,
           ROUND(AVG(orb_COMEX_SETTLE_size),2) as comex,
           ROUND(AVG(orb_CME_PRECLOSE_size),2) as cme_pre
    FROM daily_features WHERE symbol = 'MNQ' AND orb_minutes = 5 AND trading_day >= CURRENT_DATE - 360
    GROUP BY period ORDER BY period DESC
''').fetchdf().to_string(index=False))

print('\n=== MODE-B GRANDFATHER CONTAMINATION FLAG ===')
# Per research-truth-protocol.md § Mode B grandfathered, any validated_setups
# row with last_trade_day >= 2026-01-01 has an ExpR computed partly on data
# that is now sacred Mode A OOS. See 2026-04-19 re-validation at
# docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md
print(con.sql('''
    SELECT instrument,
           SUM(CASE WHEN last_trade_day >= DATE '2026-01-01' THEN 1 ELSE 0 END) AS mode_b_contaminated,
           COUNT(*) AS total_active
    FROM validated_setups
    WHERE LOWER(status) = 'active'
    GROUP BY instrument ORDER BY instrument
''').fetchdf().to_string(index=False))

print('\n=== FRESH-OOS WINDOW LENGTH (days since sacred boundary) ===')
import datetime
sacred_start = datetime.date(2026, 1, 1)
today = datetime.date.today()
days_oos = (today - sacred_start).days
print(f'  Days since HOLDOUT_SACRED_FROM (2026-01-01): {days_oos}')
print(f'  Days of Mode A fresh-OOS data available: {days_oos}')

con.close()
"
```

## Step 1b: DSR diagnostic (Shape E, A2b-2)

If `docs/runtime/lane_allocation.json` was generated post-2026-04-18 it now records DSR diagnostic per lane + per-rebalance globals. Surface it:

```bash
python -c "
import json
from pathlib import Path
data = json.loads(Path('docs/runtime/lane_allocation.json').read_text())
diag = data.get('dsr_diagnostics')
if diag:
    print('=== DSR DIAGNOSTICS (informational, not consumed by selection) ===')
    print(f\"  N_eff_raw (validator): {diag['n_eff_raw']}\")
    print(f\"  N_hat_eq9 (Bailey-LdP): {diag['n_hat_eq9']}  (avg_rho_hat = {diag['avg_rho_hat']})\")
    print(f\"  var_sr_em: {diag['var_sr_em']}\")
    print('  --- per-lane DSR ---')
    print(f\"  {'lane':38} {'DSR_raw':>8} {'DSR_eq9':>8} {'sr0':>6}\")
    for L in data['lanes']:
        if 'dsr_score' in L:
            print(f\"  {L['strategy_id'][:38]:38} {L['dsr_at_n_eff_raw']:>8.4f} {L['dsr_at_n_hat_eq9']:>8.4f} {L['sr0_at_rebalance']:>6.3f}\")
else:
    print('(no DSR diagnostics in lane_allocation.json — pre-Shape-E rebalance)')
"
```

Read DSR **alongside** trailing_expr. Per `trading_app/dsr.py:35` DSR is INFORMATIONAL — it does NOT pause/deploy on its own. Use it to spot deployed lanes whose ranker selection is on thin Bailey-LdP ground (DSR < 0.10 means observed Sharpe is consistent with "best of N noise" outcome) AND cross-check whether they're showing forward decay (combined signal stronger than either alone).

## Step 2: Flags

- **0 CORE families** for any instrument → RED
- **Edge families > 30 days old** → STALE (triggers rebuild recommendation)
- **ORB sizes trending up >20%** on 3+ sessions → regime-shift flag (may lift vol; may increase cost-risk at small accounts)
- **ORB sizes trending down >20%** on 3+ sessions → edge weakening
- **Mode-B contaminated count >0** → stored ExpR values for those lanes are NOT Mode A canonical; treat stored numbers as indicative only, cite the 2026-04-19 re-validation doc for Mode A baselines
- **Fresh-OOS days < 90** → too short for WFE OOS validation under strict Mode A (Criterion 8 N_OOS>=30 requires ~3 months at typical trade frequency)
- **Deployed lane has DSR_raw < 0.10 AND recent_3mo_expr < 0** → ESCALATE (combined Bailey-LdP + decay signal; A2b-2 Shape E diagnostic)

## Step 3: Present

One-liner per instrument plus Mode-B flag:
```
MGC: X CORE, Y REGIME [HEALTHY/CONCERN/CRITICAL], Z Mode-B contaminated
```

For deployed-lane-specific decisions, route to `docs/audit/results/2026-04-19-mnq-mode-a-committee-review-pack.md` (or the newest equivalent) for per-lane action recommendations.

## Rules

- NEVER cite counts from memory — always query fresh
- Column is `instrument` not `symbol` in validated_setups
- Fitness is in `edge_families` (robustness_status, trade_tier)
- **Family linkage: join on `family_hash`, not `head_strategy_id`.** `edge_families` stores one row per family (the head); every `validated_setups` row carries the same `family_hash` as its family. A head-only join mis-labels all non-head members as NO_FAMILY — fixed 2026-04-19 after an audit over-reported "17 MNQ unlinked" (they were all correctly classified members at non-winning RRs). Use `ef.head_strategy_id` only in queries that explicitly scope to heads (e.g., the `FAMILY HEADS ONLY` block above).
- **Mode-B grandfather contamination is a real, not theoretical flag.** On 2026-04-19 all 38 active lanes had material drift vs stored values under strict Mode A. The Mode-B flag in the query above identifies lanes whose `last_trade_day >= 2026-01-01`, meaning stored ExpR includes data now sacred under Mode A. Canonical Mode A baselines live in `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`. Always cite the Mode A recomputed value going forward, NOT the stored value.
- **ORB size regime drift is an informational signal, not an edge claim.** Big up/down moves on 3+ sessions signal a regime shift affecting risk/cost dynamics but not directly the edge. Use alongside allocator capital-efficiency review rather than as a standalone deploy/retire trigger.
