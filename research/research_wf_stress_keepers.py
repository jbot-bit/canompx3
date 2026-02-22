#!/usr/bin/env python3
"""Walk-forward + stress validation for current KEEP strategies.

Purpose:
- Validate frozen KEEP set before live trading.
- Run expanding walk-forward yearly tests and stress scenarios.

Outputs:
- research/output/wf_stress_summary.csv
- research/output/wf_stress_splits.csv
- research/output/wf_stress_report.md
"""

from __future__ import annotations

from pathlib import Path
import re
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"
REG_PATH = ROOT / "research" / "output" / "shinies_registry.csv"

# Frozen presets from forward-gate config
PRESET = {
    "A0": "base_plus_both",
    "A1": "base",
    "A2": "base_plus_both",
    "A3": "base_plus_both",
    "B1": "base_plus_both",
    "B2": "base_plus_vol60",
}


def parse_tag(tag: str):
    if not isinstance(tag, str) or "_" not in tag:
        return None
    a, b = tag.split("_", 1)
    if re.fullmatch(r"[A-Za-z0-9]+", a) is None:
        return None
    if re.fullmatch(r"[A-Za-z0-9_]+", b) is None:
        return None
    return a, b


def safe_label(lbl: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_]+", lbl) is None:
        raise ValueError(f"Unsafe label: {lbl}")
    return lbl


def max_dd(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    c = s.cumsum()
    p = c.cummax()
    d = p - c
    return float(d.max())


def load_strategy_df(con: duckdb.DuckDBPyConnection, row: pd.Series) -> pd.DataFrame:
    follower = parse_tag(str(row["follower"]))
    if follower is None:
        return pd.DataFrame()
    fsym, fsess = follower
    fs = safe_label(fsess)

    leader_tag = str(row["leader"])
    if str(row.get("id", "")) == "B2" or leader_tag.endswith("fast_le_15"):
        leader = None
    else:
        leader = parse_tag(leader_tag)

    if leader is not None:
        lsym, lsess = leader
        ls = safe_label(lsess)
        q = f"""
        SELECT o.trading_day,o.pnl_r,o.entry_ts,
               d_f.orb_{fs}_break_dir AS f_dir,
               d_f.orb_{fs}_break_delay_min AS f_delay,
               d_f.orb_{fs}_break_bar_continues AS f_cont,
               d_f.orb_{fs}_size AS f_size,
               d_f.orb_{fs}_volume AS f_vol,
               d_f.orb_{fs}_break_bar_volume AS f_bvol,
               d_f.atr_20 AS f_atr,
               d_l.orb_{ls}_break_dir AS l_dir,
               d_l.orb_{ls}_break_ts  AS l_ts
        FROM orb_outcomes o
        JOIN daily_features d_f ON d_f.symbol=o.symbol AND d_f.trading_day=o.trading_day AND d_f.orb_minutes=o.orb_minutes
        JOIN daily_features d_l ON d_l.symbol='{lsym}' AND d_l.trading_day=o.trading_day AND d_l.orb_minutes=o.orb_minutes
        WHERE o.orb_minutes=5
          AND o.symbol='{fsym}' AND o.orb_label='{fsess}'
          AND o.entry_model='{row['entry_model']}'
          AND o.confirm_bars={int(row['confirm_bars'])}
          AND o.rr_target={float(row['rr_target'])}
          AND o.pnl_r IS NOT NULL
          AND o.entry_ts IS NOT NULL
        """
    else:
        q = f"""
        SELECT o.trading_day,o.pnl_r,o.entry_ts,
               d_f.orb_{fs}_break_dir AS f_dir,
               d_f.orb_{fs}_break_delay_min AS f_delay,
               d_f.orb_{fs}_break_bar_continues AS f_cont,
               d_f.orb_{fs}_size AS f_size,
               d_f.orb_{fs}_volume AS f_vol,
               d_f.orb_{fs}_break_bar_volume AS f_bvol,
               d_f.atr_20 AS f_atr,
               NULL::VARCHAR AS l_dir,
               NULL::TIMESTAMPTZ AS l_ts
        FROM orb_outcomes o
        JOIN daily_features d_f ON d_f.symbol=o.symbol AND d_f.trading_day=o.trading_day AND d_f.orb_minutes=o.orb_minutes
        WHERE o.orb_minutes=5
          AND o.symbol='{fsym}' AND o.orb_label='{fsess}'
          AND o.entry_model='{row['entry_model']}'
          AND o.confirm_bars={int(row['confirm_bars'])}
          AND o.rr_target={float(row['rr_target'])}
          AND o.pnl_r IS NOT NULL
          AND o.entry_ts IS NOT NULL
        """

    df = con.execute(q).fetchdf()
    if df.empty:
        return df

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    if "l_ts" in df.columns:
        df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    df["f_vol_imp"] = np.where((df["f_vol"].notna()) & (df["f_vol"] > 0), df["f_bvol"] / (df["f_vol"] / 5.0), np.nan)
    return df


def base_mask(df: pd.DataFrame, sid: str, leader_tag: str) -> pd.Series:
    if sid == "B2" or str(leader_tag).endswith("fast_le_15"):
        return df["f_delay"].notna() & (df["f_delay"] <= 15)

    return (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & (df["f_dir"] == df["l_dir"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )


def preset_mask(dfb: pd.DataFrame, preset: str) -> pd.Series:
    vq = dfb["f_vol_imp"].quantile(0.60)
    if preset == "base":
        return pd.Series(True, index=dfb.index)
    if preset == "base_plus_fast15":
        return dfb["f_delay"].notna() & (dfb["f_delay"] <= 15)
    if preset == "base_plus_vol60":
        return dfb["f_vol_imp"].notna() & (dfb["f_vol_imp"] >= vq)
    if preset == "base_plus_both":
        return (
            dfb["f_delay"].notna() & (dfb["f_delay"] <= 15)
            & dfb["f_vol_imp"].notna() & (dfb["f_vol_imp"] >= vq)
        )
    return pd.Series(True, index=dfb.index)


def bootstrap_prob_positive(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    if len(values) < 20:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(sample.mean()))
    arr = np.array(means)
    return float((arr > 0).mean()), float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95))


def main() -> int:
    reg = pd.read_csv(REG_PATH)
    reg = reg[reg["status"] == "KEEP"].copy()
    reg = reg[reg["id"].isin(PRESET.keys())].copy()

    con = duckdb.connect(str(DB_PATH), read_only=True)

    summary_rows = []
    split_rows = []

    for _, r in reg.iterrows():
        sid = str(r["id"])
        preset = PRESET[sid]

        df = load_strategy_df(con, r)
        if df.empty:
            continue

        bm = base_mask(df, sid, str(r["leader"]))
        dfb = df[bm].copy().sort_values(["trading_day", "entry_ts"])
        if len(dfb) < 120:
            continue

        pm = preset_mask(dfb, preset)
        sel = dfb[pm].copy()
        if len(sel) < 60:
            continue

        # Walk-forward yearly expanding tests
        years = sorted(int(y) for y in dfb["year"].dropna().unique())
        wf_test_avgs = []
        wf_test_uplifts = []

        for ty in years:
            tr = dfb[dfb["year"] < ty]
            te = dfb[dfb["year"] == ty]
            if tr.empty or te.empty:
                continue

            p_tr = preset_mask(tr, preset)
            p_te = preset_mask(te, preset)

            tr_on = tr[p_tr]["pnl_r"]
            te_on = te[p_te]["pnl_r"]
            te_base = te["pnl_r"]

            if len(tr_on) < 40 or len(te_on) < 20 or len(te_base) < 20:
                continue

            test_avg = float(te_on.mean())
            test_uplift = float(te_on.mean() - te_base.mean())

            wf_test_avgs.append(test_avg)
            wf_test_uplifts.append(test_uplift)

            split_rows.append(
                {
                    "id": sid,
                    "test_year": ty,
                    "n_test_on": int(len(te_on)),
                    "test_avg_r": test_avg,
                    "test_uplift_vs_base": test_uplift,
                }
            )

        if not wf_test_avgs:
            continue

        # Stress tests on full selected sample
        s = sel["pnl_r"].astype(float)
        base_avg = float(dfb["pnl_r"].mean())
        sel_avg = float(s.mean())
        sel_wr = float((s > 0).mean())
        sig_years = max(1, int(sel["year"].nunique()))
        sig_per_year = float(len(sel) / sig_years)

        slip_002 = sel_avg - 0.02
        slip_005 = sel_avg - 0.05
        slip_010 = sel_avg - 0.10

        # Trim top winners stress
        def trim_top(series: pd.Series, frac: float) -> float:
            if series.empty:
                return np.nan
            n = len(series)
            k = max(1, int(round(n * frac)))
            srt = series.sort_values()
            kept = srt.iloc[: max(1, n - k)]
            return float(kept.mean())

        trim1 = trim_top(s, 0.01)
        trim5 = trim_top(s, 0.05)

        boot_prob, boot_q05, boot_q95 = bootstrap_prob_positive(s.values, n_boot=1000, seed=123)

        wf_years = len(wf_test_avgs)
        pos_avg_ratio = float(np.mean(np.array(wf_test_avgs) > 0))
        pos_uplift_ratio = float(np.mean(np.array(wf_test_uplifts) > 0))
        worst_test_avg = float(np.min(wf_test_avgs))
        median_test_avg = float(np.median(wf_test_avgs))

        # strict verdict
        strict_pass = (
            wf_years >= 3
            and pos_avg_ratio >= 0.67
            and pos_uplift_ratio >= 0.67
            and median_test_avg > 0
            and worst_test_avg > -0.10
            and slip_005 > 0
            and trim5 > 0
            and pd.notna(boot_prob) and boot_prob >= 0.80
        )

        verdict = "PROMOTE" if strict_pass else "KILL"

        summary_rows.append(
            {
                "id": sid,
                "strategy": f"{r['leader']} -> {r['follower']} {r['entry_model']}/CB{int(r['confirm_bars'])}/RR{float(r['rr_target'])}",
                "preset": preset,
                "n_selected": int(len(sel)),
                "signals_per_year": sig_per_year,
                "avg_r_selected": sel_avg,
                "wr_selected": sel_wr,
                "avg_r_base": base_avg,
                "uplift_selected_vs_base": sel_avg - base_avg,
                "wf_years": wf_years,
                "wf_pos_avg_ratio": pos_avg_ratio,
                "wf_pos_uplift_ratio": pos_uplift_ratio,
                "wf_median_test_avg": median_test_avg,
                "wf_worst_test_avg": worst_test_avg,
                "stress_slip_0_02": slip_002,
                "stress_slip_0_05": slip_005,
                "stress_slip_0_10": slip_010,
                "stress_trim_top1_avg": trim1,
                "stress_trim_top5_avg": trim5,
                "boot_prob_avg_gt0": boot_prob,
                "boot_q05": boot_q05,
                "boot_q95": boot_q95,
                "verdict": verdict,
            }
        )

    con.close()

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_sum = out_dir / "wf_stress_summary.csv"
    p_split = out_dir / "wf_stress_splits.csv"
    p_md = out_dir / "wf_stress_report.md"

    summary = pd.DataFrame(summary_rows)
    splits = pd.DataFrame(split_rows)

    if summary.empty:
        p_md.write_text("# Walk-forward stress report\n\nNo strategies passed minimum data requirements.", encoding="utf-8")
        print("No strategies passed minimum requirements.")
        return 0

    summary = summary.sort_values(["verdict", "avg_r_selected", "uplift_selected_vs_base"], ascending=[True, False, False])
    summary.to_csv(p_sum, index=False)
    splits.to_csv(p_split, index=False)

    lines = [
        "# Walk-Forward + Stress Validation Report",
        "",
        "Strict verdict gates: PROMOTE only if all anti-overfit walk-forward and stress checks pass.",
        "",
    ]

    for r in summary.itertuples(index=False):
        lines.append(
            f"- {r.id} [{r.verdict}] {r.preset}: avg={r.avg_r_selected:+.4f}, uplift={r.uplift_selected_vs_base:+.4f}, "
            f"sig/yr={r.signals_per_year:.1f}, wfYears={r.wf_years}, wf+avg={r.wf_pos_avg_ratio:.2f}, wf+uplift={r.wf_pos_uplift_ratio:.2f}, "
            f"slip0.05={r.stress_slip_0_05:+.4f}, trim5={r.stress_trim_top5_avg:+.4f}, bootP={r.boot_prob_avg_gt0:.2f}"
        )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_sum}")
    print(f"Saved: {p_split}")
    print(f"Saved: {p_md}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
