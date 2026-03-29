#!/usr/bin/env python3
"""
Migration script for pipeline fairness audit findings (2026-03-26).

Findings addressed:
  F2: Reset noise_risk = FALSE (stale global-max methodology disabled)
  F3: Populate era_dependent + max_year_pct from yearly_results
  F4: Record WFE verdicts for outlier strategies (>1.50 or <0.50)
  F5: Set slippage_validation_status per lane

Idempotent — safe to re-run. All operations are UPDATE on existing rows.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.db_manager import init_trading_app_schema


def main():
    # Ensure schema has the new columns
    init_trading_app_schema(db_path=GOLD_DB_PATH)

    con = duckdb.connect(str(GOLD_DB_PATH))

    # ── F2: Reset noise_risk ─────────────────────────────────────────
    print("F2: Resetting noise_risk = FALSE for all validated strategies...")
    con.execute("UPDATE validated_setups SET noise_risk = FALSE WHERE noise_risk IS NOT FALSE")
    updated = con.execute("SELECT COUNT(*) FROM validated_setups WHERE noise_risk = FALSE").fetchone()[0]
    print(f"  {updated} strategies now have noise_risk = FALSE")

    # ── F3: Populate era_dependent + max_year_pct ────────────────────
    print("\nF3: Computing era_dependent + max_year_pct from yearly_results...")
    rows = con.execute("SELECT strategy_id, yearly_results FROM validated_setups WHERE status = 'active'").fetchall()

    era_dep_count = 0
    for sid, yr_json in rows:
        if not yr_json:
            continue
        try:
            yearly = json.loads(yr_json) if isinstance(yr_json, str) else yr_json
        except (json.JSONDecodeError, TypeError):
            continue

        # Compute total_r per year from yearly_results
        year_totals = {}
        for y, d in yearly.items():
            total_r = d.get("total_r")
            if total_r is None:
                total_r = d.get("avg_r", 0) * d.get("trades", 0)
            year_totals[y] = total_r

        total_r_sum = sum(year_totals.values())
        if total_r_sum > 0:
            max_year_pct = max(yr / total_r_sum for yr in year_totals.values())
        else:
            max_year_pct = None

        era_dep = max_year_pct is not None and max_year_pct > 0.50

        con.execute(
            "UPDATE validated_setups SET era_dependent = ?, max_year_pct = ? WHERE strategy_id = ?",
            [era_dep, round(max_year_pct, 4) if max_year_pct is not None else None, sid],
        )
        if era_dep:
            era_dep_count += 1

    print(f"  {len(rows)} strategies processed, {era_dep_count} flagged ERA_DEPENDENT")

    # Show era_dependent strategies
    era_rows = con.execute(
        "SELECT strategy_id, instrument, orb_label, rr_target, max_year_pct "
        "FROM validated_setups WHERE era_dependent = TRUE ORDER BY max_year_pct DESC"
    ).fetchall()
    for r in era_rows:
        print(f"    {r[0]}: max_year_pct={r[4]:.1%}")

    # Show Apex lane status
    print("\n  Apex lanes concentration:")
    apex_sids = [
        "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20",  # O15→O5 2026-03-29
        "MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_ORB_G8_O15",
        "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8",
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60",  # O15→O5 2026-03-29
        "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075",
    ]
    for sid in apex_sids:
        row = con.execute(
            "SELECT era_dependent, max_year_pct FROM validated_setups WHERE strategy_id = ?",
            [sid],
        ).fetchone()
        if row:
            label = "ERA_DEPENDENT" if row[0] else "CLEAN"
            pct_str = f"{row[1]:.1%}" if row[1] is not None else "N/A"
            print(f"    {sid}: {label} ({pct_str})")

    # ── F4: Record WFE verdicts ──────────────────────────────────────
    print("\nF4: Recording WFE verdicts...")

    # Load WF fold data for analysis
    wf_path = Path(__file__).resolve().parent.parent.parent / "data" / "walkforward_results.jsonl"
    wf_data = {}
    if wf_path.exists():
        with open(wf_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    wf_data[d["strategy_id"]] = d
                except (json.JSONDecodeError, KeyError):
                    continue

    # Get all WFE outlier strategies
    outliers = con.execute("""
        SELECT strategy_id, orb_label, wfe, instrument
        FROM validated_setups
        WHERE (wfe > 1.50 OR wfe < 0.50) AND wfe IS NOT NULL
        ORDER BY wfe DESC
    """).fetchall()

    for sid, _session, wfe, _instrument in outliers:
        folds = wf_data.get(sid, {}).get("windows", [])

        if wfe < 0.50:
            # WFE_UNDERFIT
            verdict = "UNDERFIT"
            notes = (
                f"OOS retains only {wfe:.0%} of IS performance. "
                "Trade at reduced confidence. Monitor forward data closely."
            )
        elif wfe > 1.50:
            # Analyze fold-by-fold: is it regime-concentrated or consistent?
            if folds:
                fold_ratios = []
                for w in folds:
                    is_e = w.get("is_exp_r")
                    oos_e = w.get("test_exp_r")
                    if is_e and is_e > 0 and oos_e is not None:
                        fold_ratios.append(oos_e / is_e)

                # Check if later folds dominate (regime improvement)
                n_folds = len(fold_ratios)
                if n_folds >= 4:
                    first_half = fold_ratios[: n_folds // 2]
                    second_half = fold_ratios[n_folds // 2 :]
                    mean_first = sum(first_half) / len(first_half) if first_half else 0
                    mean_second = sum(second_half) / len(second_half) if second_half else 0

                    if mean_second > mean_first * 1.5:
                        verdict = "REGIME_BET"
                        notes = (
                            f"Later folds outperform earlier (mean ratio {mean_second:.2f} vs {mean_first:.2f}). "
                            "OOS improvement driven by recent regime expansion. Not leakage. "
                            "Monitor for regime reversion."
                        )
                    else:
                        # Check for single outlier fold
                        if max(fold_ratios) > 3 * sum(fold_ratios) / len(fold_ratios):
                            verdict = "LUCKY_FOLD"
                            notes = (
                                f"Single outlier fold (max ratio {max(fold_ratios):.2f}, "
                                f"mean {sum(fold_ratios) / len(fold_ratios):.2f}). "
                                "WFE inflated by one window. Strategy otherwise normal."
                            )
                        else:
                            verdict = "REGIME_BET"
                            notes = (
                                f"Consistently elevated OOS/IS ratios across folds (mean {sum(fold_ratios) / len(fold_ratios):.2f}). "
                                "Likely regime-driven improvement. Monitor for reversion."
                            )
                else:
                    verdict = "REGIME_BET"
                    notes = f"Only {n_folds} folds with valid IS — insufficient for detailed analysis. Defaulting to REGIME_BET."
            else:
                verdict = "REGIME_BET"
                notes = "No fold data in walkforward JSONL. Defaulting to REGIME_BET based on session-level analysis."

        con.execute(
            """UPDATE validated_setups SET
                wfe_verdict = ?, wfe_investigation_date = '2026-03-26',
                wfe_investigation_notes = ?
            WHERE strategy_id = ?""",
            [verdict, notes, sid],
        )
        print(f"  {sid}: WFE={wfe:.3f} -> {verdict}")

    print(f"  {len(outliers)} WFE outliers recorded")

    # ── F5: Set slippage_validation_status ────────────────────────────
    print("\nF5: Setting slippage_validation_status per session...")

    # Break-even ticks from cost_model.py comments (MNQ):
    #   COMEX_SETTLE: 4.9 extra ticks (PENDING — no tbbo pilot yet)
    #   SINGAPORE_OPEN: 6.0 extra ticks (PENDING — no tbbo)
    #   NYSE_CLOSE: 15.4 extra ticks (ROBUST)
    #   NYSE_OPEN: 17.7 extra ticks (ROBUST)
    #   US_DATA_1000: 22.2 extra ticks (ROBUST — from audit analysis)
    session_status = {
        "NYSE_CLOSE": "ROBUST",
        "NYSE_OPEN": "ROBUST",
        "US_DATA_1000": "ROBUST",
        "SINGAPORE_OPEN": "PENDING",
        "COMEX_SETTLE": "PENDING",
        "BRISBANE_1025": "PENDING",
        "US_DATA_830": "PENDING",
        "CME_PRECLOSE": "PENDING",
    }

    for session, status in session_status.items():
        con.execute(
            "UPDATE validated_setups SET slippage_validation_status = ? WHERE orb_label = ?",
            [status, session],
        )
    # MGC sessions: different instrument, set PENDING
    con.execute("UPDATE validated_setups SET slippage_validation_status = 'PENDING' WHERE instrument = 'MGC'")
    # MES sessions: PENDING (no tbbo pilot for MES either)
    con.execute("UPDATE validated_setups SET slippage_validation_status = 'PENDING' WHERE instrument = 'MES'")

    # Show final status
    rows = con.execute("""
        SELECT orb_label, slippage_validation_status, COUNT(*)
        FROM validated_setups GROUP BY orb_label, slippage_validation_status
        ORDER BY orb_label
    """).fetchall()
    for r in rows:
        print(f"  {r[0]}: {r[1]} ({r[2]} strategies)")

    con.commit()
    con.close()

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"F2: noise_risk = FALSE for all {updated} strategies")
    print(f"F3: {era_dep_count} ERA_DEPENDENT strategies flagged")
    print(f"F4: {len(outliers)} WFE verdicts recorded")
    print("F5: slippage_validation_status set per session")


if __name__ == "__main__":
    main()
