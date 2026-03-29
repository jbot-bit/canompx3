"""GAP 5: WFE inspection with IS/OOS breakdown."""

import duckdb

from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

print("=" * 100)
print("GAP 5: WFE INSPECTION - IS/OOS BREAKDOWN")
print("=" * 100)

all_ids = [
    # Current lanes
    "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20_O15",
    "MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_ORB_G8_O15",
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8",
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075",
    # ATR70 candidates (same-params)
    "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR70_VOL",
    "MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_ATR70_VOL_O15",
    # ATR70 candidates (best-ExpR from Config B)
    "MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR70_VOL",
    "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR70_VOL",
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR70_VOL",
    "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR70_VOL_O15_S075",
]

header = f"  {'Strategy':<58s} {'N':>5s} {'WFE':>7s} {'ExpR':>8s} {'OOS_ExpR':>9s} {'WF_wins':>8s} {'Flags'}"
print(header)
print("-" * 120)

for sid in all_ids:
    r = con.execute(
        "SELECT strategy_id, sample_size, wfe, expectancy_r, oos_exp_r, "
        "wf_windows, wf_passed, sharpe_ratio "
        f"FROM validated_setups WHERE strategy_id = '{sid}'"
    ).fetchone()

    if r is None:
        print(f"  {sid:<58s}   NOT IN VALIDATED")
        continue

    sid_s, n, wfe, expr, oos_expr, wf_wins, wf_passed, sharpe = r

    flags = []
    if wfe is not None:
        if wfe > 1.5:
            flags.append("WFE>1.5_INSPECT")
        elif wfe > 0.95 and n < 200:
            flags.append("LEAKAGE(N<200)")
        elif wfe > 0.95:
            flags.append("WFE>0.95")
        if wfe > 1.0:
            flags.append("OOS>IS")
        if wfe < 0.5:
            flags.append("OVERFIT")

    oos_str = f"{oos_expr:+.4f}" if oos_expr is not None else "    N/A"
    wfe_str = f"{wfe:.4f}" if wfe is not None else "  N/A"
    wf_str = str(wf_wins) if wf_wins is not None else "N/A"
    flag_str = ", ".join(flags) if flags else "CLEAN"

    print(f"  {sid_s:<58s} {n:5d} {wfe_str:>7s} {expr:+8.4f} {oos_str:>9s} {wf_str:>8s} {flag_str}")

con.close()
