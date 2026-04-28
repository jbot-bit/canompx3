"""Audit script for the AI-studio external review claims on D4.

Verifies (or refutes) each claim against canonical layers only:
- bars_1m / daily_features / orb_outcomes
No derived layers, no metadata trust.

Run: python research/audit_aistudio_claims_d4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import COST_SPECS  # noqa: E402
from pipeline.dst import SESSION_CATALOG  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


def banner(msg: str) -> None:
    print()
    print("=" * 76)
    print(msg)
    print("=" * 76)


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # ============================================================
    # CLAIM 5: MGC at COMEX_SETTLE — auditor says GARCH should work better there
    # ============================================================
    banner("CLAIM 5 — MGC vs MNQ at COMEX_SETTLE (the Gold-during-Gold-settlement test)")
    for apt in [5, 15, 30]:
        for inst in ["MGC", "MNQ"]:
            q = f"""
            WITH lane AS (
                SELECT o.pnl_r,
                       NTILE(5) OVER (ORDER BY d.garch_forecast_vol_pct) AS q
                FROM orb_outcomes o
                JOIN daily_features d ON o.trading_day=d.trading_day
                                       AND o.symbol=d.symbol
                                       AND o.orb_minutes=d.orb_minutes
                WHERE o.symbol = ?
                  AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes={apt}
                  AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
                  AND o.outcome IN ('win','loss','scratch')
                  AND d.orb_COMEX_SETTLE_break_dir='long'
                  AND o.trading_day < ?
                  AND d.garch_forecast_vol_pct IS NOT NULL
            )
            SELECT q, COUNT(*),
                   AVG(pnl_r),
                   SUM(CASE WHEN pnl_r>0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS wr
            FROM lane GROUP BY q ORDER BY q
            """
            rs = con.execute(q, [inst, HOLDOUT_SACRED_FROM]).fetchall()
            if not rs:
                continue
            q1 = next((r for r in rs if r[0] == 1), None)
            q5 = next((r for r in rs if r[0] == 5), None)
            if q1 and q5:
                spread = q5[2] - q1[2]
                print(
                    f"  {inst} {apt:>2}m: Q1 N={q1[1]:>3} mu={q1[2]:+.4f} | "
                    f"Q5 N={q5[1]:>3} mu={q5[2]:+.4f} | Q5-Q1={spread:+.4f}"
                )

    # ============================================================
    # CLAIM 6: Slippage scaling — auditor claims 3x higher on garch>70 days
    # Project's COST_SPECS uses fixed dollar slippage. So the question is:
    # do REALIZED loss-trades get worse pnl_r on garch>70 (because the stop slipped)?
    # If pnl_r on losses < -1.0R systematically on 'on' cell → slippage IS scaling.
    # ============================================================
    banner("CLAIM 6 — slippage scaling check on losing trades")
    mnq = COST_SPECS["MNQ"]
    print(
        f"COST_SPECS MNQ: commission_rt=${mnq.commission_rt}, "
        f"spread_doubled=${mnq.spread_doubled}, slippage=${mnq.slippage}"
    )
    print(f"  Total fixed friction = ${mnq.commission_rt + mnq.spread_doubled + mnq.slippage:.2f}")
    print("  Project models slippage as FIXED DOLLAR; it does NOT scale with vol.")
    print()
    q = """
    SELECT
        CASE WHEN d.garch_forecast_vol_pct > 70 THEN 'on' ELSE 'off' END AS cell,
        COUNT(*) FILTER (WHERE o.outcome = 'loss') AS n_loss,
        AVG(o.pnl_r) FILTER (WHERE o.outcome = 'loss') AS mean_loss_r,
        MIN(o.pnl_r) FILTER (WHERE o.outcome = 'loss') AS min_loss_r,
        AVG(o.risk_dollars) AS avg_risk_d
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day
                          AND o.symbol=d.symbol
                          AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
      AND o.outcome IN ('win','loss','scratch')
      AND d.orb_COMEX_SETTLE_break_dir='long'
      AND o.trading_day < ?
      AND d.orb_COMEX_SETTLE_size >= 5
    GROUP BY cell ORDER BY cell
    """
    print("If realized losses are systematically <-1.0R on 'on' → slippage is scaling")
    print("cell | n_loss | mean_loss_R | min_loss_R | avg_risk_$")
    for r in con.execute(q, [HOLDOUT_SACRED_FROM]).fetchall():
        print(f"  {r[0]} | {r[1]:>4} | {r[2]:+.4f} | {r[3]:+.4f} | ${r[4]:.2f}")

    # ============================================================
    # CLAIM 7 — alternative filters within ORB_G5
    # ============================================================
    banner("CLAIM 7 — alternative filters within ORB_G5, MNQ COMEX_SETTLE 5m RR1.0 long")
    q = """
    SELECT o.pnl_r,
           d.garch_forecast_vol_pct,
           d.overnight_range_pct,
           d.atr_20_pct,
           d.gap_open_points,
           d.daily_open, d.prev_day_high, d.prev_day_low, d.prev_day_close
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day
                          AND o.symbol=d.symbol
                          AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
      AND o.outcome IN ('win','loss','scratch')
      AND d.orb_COMEX_SETTLE_break_dir='long'
      AND o.trading_day < ?
      AND d.orb_COMEX_SETTLE_size >= 5
    """
    df = con.execute(q, [HOLDOUT_SACRED_FROM]).df()
    print(f"Within-ORB_G5 N: {len(df)}")
    print()
    print("Filter | N_on | mu_on | N_off | mu_off | delta | t | p")
    filters = {
        "garch_pct>70": df["garch_forecast_vol_pct"] > 70,
        "garch_pct>80 (Q5)": df["garch_forecast_vol_pct"] > 80,
        "garch_pct<20 (Q1 low-vol)": df["garch_forecast_vol_pct"] < 20,
        "ovn_range_pct>80": df["overnight_range_pct"] > 80,
        "atr_20_pct>70": df["atr_20_pct"] > 70,
        "atr_20_pct<30 (low ATR)": df["atr_20_pct"] < 30,
        "gap_pts>0 (gap up)": df["gap_open_points"] > 0,
        "gap_pts<0 (gap down)": df["gap_open_points"] < 0,
        "open>PDH": df["daily_open"] > df["prev_day_high"],
        "open<PDL": df["daily_open"] < df["prev_day_low"],
        "open>=mid_PD": df["daily_open"] >= (df["prev_day_high"] + df["prev_day_low"]) / 2,
    }
    for name, mask in filters.items():
        m = mask.fillna(False)
        on = df.loc[m, "pnl_r"].values
        off = df.loc[~m, "pnl_r"].values
        if len(on) >= 10 and len(off) >= 10:
            delta = on.mean() - off.mean()
            t, p = sp.ttest_ind(on, off, equal_var=False)
            print(
                f"  {name:<32s} | {len(on):>3} | {on.mean():+.4f} | "
                f"{len(off):>3} | {off.mean():+.4f} | {delta:+.4f} | "
                f"{t:+.2f} | {p:.4f}"
            )
        else:
            print(f"  {name:<32s} | low N (on={len(on)}, off={len(off)})")

    # ============================================================
    # CLAIM 8 — Momentum vs reversion identity (MFE / MAE profile)
    # ============================================================
    banner("CLAIM 8 — momentum vs reversion identity (MFE/MAE)")
    q = """
    SELECT
        CASE WHEN d.garch_forecast_vol_pct > 70 THEN 'on' ELSE 'off' END AS cell,
        COUNT(*),
        AVG(o.pnl_r),
        AVG(o.mfe_r),
        AVG(o.mae_r),
        AVG(CASE WHEN o.outcome='win' THEN o.pnl_r END) AS avg_win,
        AVG(CASE WHEN o.outcome='loss' THEN o.pnl_r END) AS avg_loss
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day
                          AND o.symbol=d.symbol
                          AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
      AND o.outcome IN ('win','loss','scratch')
      AND d.orb_COMEX_SETTLE_break_dir='long'
      AND o.trading_day < ?
      AND d.orb_COMEX_SETTLE_size >= 5
    GROUP BY cell ORDER BY cell
    """
    print("MFE_on > MFE_off → momentum extends on garch>70 days (auditor's claim test)")
    print("If win_R is similar but mean_R rises → entries are cleaner (less noise)")
    print("cell | N | mean_R | MFE | MAE | avg_win | avg_loss")
    for r in con.execute(q, [HOLDOUT_SACRED_FROM]).fetchall():
        print(
            f"  {r[0]} | {r[1]:>3} | {r[2]:+.4f} | {r[3]:+.4f} | {r[4]:+.4f} | "
            f"{r[5]:+.4f} | {r[6]:+.4f}"
        )

    # ============================================================
    # CLAIM 9 — Q1 (low-vol) edge probe — both directions
    # ============================================================
    banner("CLAIM 9 — Q1 (low-vol) edge probe, within ORB_G5")
    for direction in ["long", "short"]:
        q = f"""
        WITH binned AS (
            SELECT o.pnl_r,
                   NTILE(5) OVER (ORDER BY d.garch_forecast_vol_pct) AS q
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day=d.trading_day
                                  AND o.symbol=d.symbol
                                  AND o.orb_minutes=d.orb_minutes
            WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
              AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
              AND o.outcome IN ('win','loss','scratch')
              AND d.orb_COMEX_SETTLE_break_dir = ?
              AND o.trading_day < ?
              AND d.garch_forecast_vol_pct IS NOT NULL
              AND d.orb_COMEX_SETTLE_size >= 5
        )
        SELECT q, COUNT(*),
               AVG(pnl_r),
               SUM(CASE WHEN pnl_r>0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS wr
        FROM binned GROUP BY q ORDER BY q
        """
        print(f"--- {direction} side, within ORB_G5 ---")
        print("  Q | N | mean_R | WR")
        for r in con.execute(q, [direction, HOLDOUT_SACRED_FROM]).fetchall():
            print(f"  Q{r[0]} | {r[1]:>3} | {r[2]:+.4f} | {r[3]:.1f}%")

    # ============================================================
    # CLAIM 10 — RR1.0 vs RR1.5 cleanness — auditor calls RR1.0 a vanity metric
    # because the system "must" run at RR1.5 to cover overhead.
    # Test: at deployed RR1.5, what is the absolute Sharpe of HIGH-only vs ALL on
    # the deployed lane. Already shown earlier; this is a recap with both RRs.
    # ============================================================
    banner("CLAIM 10 — RR1.0 vs RR1.5 ExpR per session (deployed RR is the truth test)")
    for rr in [1.0, 1.5]:
        q = """
        SELECT
            CASE WHEN d.garch_forecast_vol_pct > 70 THEN 'on' ELSE 'off' END AS cell,
            COUNT(*),
            AVG(o.pnl_r),
            STDDEV(o.pnl_r)
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day=d.trading_day
                              AND o.symbol=d.symbol
                              AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
          AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=?
          AND o.outcome IN ('win','loss','scratch')
          AND d.orb_COMEX_SETTLE_break_dir='long'
          AND o.trading_day < ?
          AND d.orb_COMEX_SETTLE_size >= 5
        GROUP BY cell ORDER BY cell
        """
        print(f"--- RR={rr} long, within ORB_G5 ---")
        print("cell | N | mean_R | sd | sharpe_pt")
        for r in con.execute(q, [rr, HOLDOUT_SACRED_FROM]).fetchall():
            sr = r[2] / r[3] if r[3] and r[3] > 0 else 0
            print(f"  {r[0]} | {r[1]:>3} | {r[2]:+.4f} | {r[3]:.4f} | {sr:+.4f}")

    # ============================================================
    # CLAIM 11 — RR1.0 vs RR1.5 — cost realism on $-net
    # The 'cleaner' RR1.0 t-stat must still cover real friction.
    # 1R = avg_risk = $69.74 on 'on' cell. Friction = $2.92 fixed.
    # → friction = 4.2% of 1R. Both RRs cover friction easily.
    # The auditor's slippage-scaling concern is empirically refuted by the
    # avg_loss_R numbers in CLAIM 6 above.
    # ============================================================
    banner("CLAIM 11 — net-of-cost ExpR check (RR1.0 long within ORB_G5)")
    q = """
    SELECT
        CASE WHEN d.garch_forecast_vol_pct > 70 THEN 'on' ELSE 'off' END AS cell,
        COUNT(*),
        AVG(o.pnl_r) AS expr,
        AVG(o.pnl_dollars) AS expr_d,
        AVG(o.risk_dollars) AS risk_d
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day
                          AND o.symbol=d.symbol
                          AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
      AND o.outcome IN ('win','loss','scratch')
      AND d.orb_COMEX_SETTLE_break_dir='long'
      AND o.trading_day < ?
      AND d.orb_COMEX_SETTLE_size >= 5
    GROUP BY cell ORDER BY cell
    """
    print("cell | N | ExpR | ExpR_$ | avg_risk_$ | implied_friction_%1R")
    for r in con.execute(q, [HOLDOUT_SACRED_FROM]).fetchall():
        # implied friction = COST/1R if we assume pnl_r is friction-net
        cost = mnq.commission_rt + mnq.spread_doubled + mnq.slippage
        pct = 100 * cost / r[4] if r[4] else float("nan")
        print(
            f"  {r[0]} | {r[1]:>3} | {r[2]:+.4f} | ${r[3]:+.2f} | ${r[4]:.2f} | "
            f"{pct:.2f}%"
        )

    # ============================================================
    # CLAIM 12 — Aperture mystery — is it micro-momentum or a fitness artifact?
    # Test: does the SAME garch>P80 (top quintile) work on 15m and 30m?
    # If yes → not microstructure. If no → 5m-specific.
    # ============================================================
    banner("CLAIM 12 — top-quintile (P80) garch across apertures (the hard test)")
    for apt in [5, 15, 30]:
        # P80 is computed per-aperture from IS distribution
        q = f"""
        WITH dist AS (
            SELECT d.trading_day, d.garch_forecast_vol_pct
            FROM daily_features d
            WHERE d.symbol='MNQ' AND d.orb_minutes={apt}
              AND d.trading_day < ?
              AND d.garch_forecast_vol_pct IS NOT NULL
        )
        SELECT PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY garch_forecast_vol_pct)
        FROM dist
        """
        p80 = con.execute(q, [HOLDOUT_SACRED_FROM]).fetchone()[0]
        q2 = f"""
        SELECT
            CASE WHEN d.garch_forecast_vol_pct >= ? THEN 'top20' ELSE 'rest' END AS cell,
            COUNT(*),
            AVG(o.pnl_r) AS expr,
            SUM(CASE WHEN o.pnl_r>0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS wr
        FROM orb_outcomes o
        JOIN daily_features d ON o.trading_day=d.trading_day
                              AND o.symbol=d.symbol
                              AND o.orb_minutes=d.orb_minutes
        WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes={apt}
          AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.0
          AND o.outcome IN ('win','loss','scratch')
          AND d.orb_COMEX_SETTLE_break_dir='long'
          AND o.trading_day < ?
        GROUP BY cell ORDER BY cell
        """
        print(f"--- {apt}m aperture (P80 = {p80:.2f}) ---")
        print("cell | N | mean_R | WR")
        rs = con.execute(q2, [p80, HOLDOUT_SACRED_FROM]).fetchall()
        top = next((r for r in rs if r[0] == "top20"), None)
        rest = next((r for r in rs if r[0] == "rest"), None)
        if top and rest:
            spread = top[2] - rest[2]
            for r in rs:
                print(f"  {r[0]} | {r[1]:>3} | {r[2]:+.4f} | {r[3]:.1f}%")
            print(f"  → top20 - rest = {spread:+.4f} R")

    # ============================================================
    # CLAIM 13 — RR1.5 sanity: would D-0 v2 + Q5 sizing pass the floor?
    # Auditor suggested re-running sizing on Q5 ONLY (P80-P100), ignoring P70-P80
    # Test: simulate this on the canonical lane
    # ============================================================
    banner("CLAIM 13 — D-0 v2 sizing redux: Q5-only (P80) sizing on deployed RR1.5")
    # Note: D-0 v2 lane uses OVNRNG_100 filter, not raw ORB_G5 cohort. Let's match exactly.
    from trading_app.config import ALL_FILTERS
    min_range = getattr(ALL_FILTERS["OVNRNG_100"], "min_range", None)
    q = """
    SELECT o.pnl_r, d.garch_forecast_vol_pct
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day
                          AND o.symbol=d.symbol
                          AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.5
      AND o.outcome IN ('win','loss','scratch')
      AND o.trading_day < ?
      AND d.overnight_range IS NOT NULL
      AND d.overnight_range >= ?
      AND d.garch_forecast_vol_pct IS NOT NULL
    """
    df_d0 = con.execute(q, [HOLDOUT_SACRED_FROM, min_range]).df()
    print(f"D-0 v2 IS cohort N: {len(df_d0)}")
    p80 = float(np.percentile(df_d0["garch_forecast_vol_pct"], 80))
    print(f"P80 garch_pct = {p80:.2f}")
    # Baseline (uniform 1.0x)
    baseline_rs = df_d0["pnl_r"].values
    bl_mu = float(baseline_rs.mean())
    bl_sd = float(baseline_rs.std(ddof=1))
    bl_sr = bl_mu / bl_sd if bl_sd > 0 else 0

    # Q5-only-sized: keep top 20% at 1.5x, rest at 1.0x
    sized_q5 = np.where(
        df_d0["garch_forecast_vol_pct"].values >= p80,
        df_d0["pnl_r"].values * 1.5,
        df_d0["pnl_r"].values * 1.0,
    )
    sz_mu = float(sized_q5.mean())
    sz_sd = float(sized_q5.std(ddof=1))
    sz_sr = sz_mu / sz_sd if sz_sd > 0 else 0
    abs_diff = sz_sr - bl_sr
    rel_uplift = 100 * abs_diff / abs(bl_sr) if bl_sr else float("nan")
    print(f"Baseline (1.0x flat):    mu={bl_mu:+.4f}, sd={bl_sd:.4f}, SR_pt={bl_sr:.4f}")
    print(f"Q5-only-sized (1.5x P80): mu={sz_mu:+.4f}, sd={sz_sd:.4f}, SR_pt={sz_sr:.4f}")
    print(f"  abs Sharpe diff: {abs_diff:+.4f}  (D-0 v2 floor: 0.05)")
    print(f"  rel Sharpe uplift: {rel_uplift:+.2f}%  (D-0 v2 floor: 15%)")

    # And try a Q5-with-skip (skip Q1, hold Q2-Q4 at 1.0x, Q5 at 1.5x)
    p20 = float(np.percentile(df_d0["garch_forecast_vol_pct"], 20))
    sized_skipQ1_q5 = np.where(
        df_d0["garch_forecast_vol_pct"].values >= p80,
        df_d0["pnl_r"].values * 1.5,
        np.where(
            df_d0["garch_forecast_vol_pct"].values < p20,
            0.0,  # SKIP Q1
            df_d0["pnl_r"].values * 1.0,
        ),
    )
    # Sharpe using only the kept trades (sized_skipQ1_q5 != 0 OR signal=high)
    nz_mask = ~((df_d0["garch_forecast_vol_pct"].values < p20))
    sz2_mu = float(sized_skipQ1_q5.mean())
    sz2_sd = float(sized_skipQ1_q5.std(ddof=1))
    sz2_sr = sz2_mu / sz2_sd if sz2_sd > 0 else 0
    print()
    print("Q5-up + Q1-skip variant (P20=skip, P80=1.5x, mid=1.0x):")
    print(f"  mu={sz2_mu:+.4f}, sd={sz2_sd:.4f}, SR_pt={sz2_sr:.4f}")
    print(f"  abs Sharpe diff vs baseline: {sz2_sr - bl_sr:+.4f}")

    # Q1-only short (auditor claim "Q1 is low-vol = high WR grind")
    q_short = """
    SELECT o.pnl_r, d.garch_forecast_vol_pct
    FROM orb_outcomes o
    JOIN daily_features d ON o.trading_day=d.trading_day
                          AND o.symbol=d.symbol
                          AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.orb_label='COMEX_SETTLE' AND o.orb_minutes=5
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=1.5
      AND o.outcome IN ('win','loss','scratch')
      AND o.trading_day < ?
      AND d.overnight_range IS NOT NULL
      AND d.overnight_range >= ?
      AND d.garch_forecast_vol_pct IS NOT NULL
    """
    print()

    # ============================================================
    # CLAIM 14 — auditor claims "GARCH is whole-day forecast" → mismatch with intra-session ORB
    # The local extract grounds garch_forecast_vol_pct as a percentile-rank applied
    # at session start (prior-day-close based). It IS used as a daily/whole-period forecast
    # input. The auditor's framing is technically correct but the project does
    # use it correctly per docs. Re-confirm timing alignment (RULE 1.2).
    # ============================================================
    banner("CLAIM 14 — GARCH timing alignment review")
    print("Project doctrine: garch_forecast_vol_pct is computed from rolling 252-day")
    print("prior-day daily closes. Forecast is fixed at prior-day close, applied once")
    print("per trading_day before any ORB session. RULE 1.2 valid-domain: any session.")
    print(f"Sample SESSION_CATALOG.COMEX_SETTLE: {SESSION_CATALOG['COMEX_SETTLE']}")
    print()
    print("Auditor's 'whole-day forecast' framing is correct in the sense that it's")
    print("a forecast OF the day's vol, not a per-session real-time vol gauge. But its")
    print("use as a SESSION-LEVEL gate (one bit per trading day) is consistent with the")
    print("Carver Ch9-10 vol-targeting framework, where daily vol forecasts size each")
    print("intraday position. Not a structural mismatch — but DOES underscore that the")
    print("signal is a daily regime gate, not an intraday timing tool.")

    con.close()


if __name__ == "__main__":
    main()
