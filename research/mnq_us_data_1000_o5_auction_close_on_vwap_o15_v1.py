"""MNQ US_DATA_1000 O5 auction-close confluence on VWAP O15 lane v1.

Scope is locked by:
  docs/audit/hypotheses/2026-04-22-mnq-us-data-1000-o5-auction-close-on-vwap-o15-v1.yaml

This runner evaluates exactly one nested same-session ORB-state claim:
the first 5 minutes of US_DATA_1000 as a confluence overlay on the fixed
MNQ US_DATA_1000 O15 E2 RR1.5 CB1 VWAP_MID_ALIGNED lane.

Canonical truth sources only:
  - bars_1m
  - daily_features
  - orb_outcomes

Outputs:
  - docs/audit/results/2026-04-22-mnq-us-data-1000-o5-auction-close-on-vwap-o15-v1.md
  - docs/audit/results/2026-04-22-mnq-us-data-1000-o5-auction-close-on-vwap-o15-v1-rows.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from pipeline.dst import orb_utc_window
from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-22-mnq-us-data-1000-o5-auction-close-on-vwap-o15-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-22-mnq-us-data-1000-o5-auction-close-on-vwap-o15-v1.md")
ROW_CSV = Path("docs/audit/results/2026-04-22-mnq-us-data-1000-o5-auction-close-on-vwap-o15-v1-rows.csv")
REPO_ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260422)
INSTRUMENT = "MNQ"
SESSION = "US_DATA_1000"
TARGET_APERTURE = 15
FEATURE_APERTURE = 5
RR_TARGET = 1.5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
FILTER_KEY = "VWAP_MID_ALIGNED"
OOS_START = HOLDOUT_SACRED_FROM
OOS_END = pd.Timestamp("2026-04-17").date()


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def _load_prereg(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_header(prereg: dict) -> list[str]:
    commit_sha = (((prereg.get("reproducibility") or {}).get("commit_sha")) or "").strip()
    if not commit_sha or commit_sha == "TO_FILL_AFTER_COMMIT":
        raise ValueError("pre-reg commit_sha must be stamped before running the harness")
    rel_script = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    return [
        f"# {prereg['title']}",
        "",
        f"**Pre-reg:** `{PREREG_PATH}` (LOCKED, commit_sha={commit_sha})",
        f"**Script:** `{rel_script}`",
        f"**DB:** `{GOLD_DB_PATH}`",
        "",
    ]


def _compute_clv(open_5: float, high_5: float, low_5: float, close_5: float) -> float | None:
    _ = open_5
    rng = high_5 - low_5
    if not np.isfinite(rng) or rng <= 0:
        return None
    clv = (close_5 - low_5) / rng
    return float(clv)


def _feature_fire_for_row(break_dir: str, o5_clv: float | None) -> int:
    if o5_clv is None or not np.isfinite(o5_clv):
        return 0
    if break_dir == "long":
        return int(o5_clv >= 0.75)
    if break_dir == "short":
        return int(o5_clv <= 0.25)
    return 0


def load_target_lane() -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        sql = f"""
        SELECT
            o.trading_day,
            o.pnl_r,
            o.outcome,
            d.atr_20,
            d.atr_20_pct,
            d.overnight_range_pct,
            d.gap_open_points,
            d.orb_{SESSION}_high,
            d.orb_{SESSION}_low,
            d.orb_{SESSION}_size,
            d.orb_{SESSION}_break_dir,
            d.orb_{SESSION}_vwap
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{INSTRUMENT}'
          AND o.orb_label = '{SESSION}'
          AND o.orb_minutes = {TARGET_APERTURE}
          AND o.entry_model = '{ENTRY_MODEL}'
          AND o.confirm_bars = {CONFIRM_BARS}
          AND o.rr_target = {RR_TARGET}
          AND o.trading_day < DATE '{OOS_END.isoformat()}'
          AND o.pnl_r IS NOT NULL
          AND d.atr_20 IS NOT NULL
          AND d.atr_20 > 0
          AND d.orb_{SESSION}_break_dir IN ('long', 'short')
        ORDER BY o.trading_day
        """
        df = con.execute(sql).df()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError("target lane query returned zero rows")
    lane_sig = filter_signal(df, FILTER_KEY, orb_label=SESSION)
    df["lane_fire"] = lane_sig.astype(int)
    df = df.loc[df["lane_fire"] == 1].copy()
    if df.empty:
        raise RuntimeError("canonical base lane filter produced zero rows")
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["is_is"] = df["trading_day"].dt.date < OOS_START
    df["is_oos"] = (df["trading_day"].dt.date >= OOS_START) & (df["trading_day"].dt.date < OOS_END)
    df["is_scratch"] = df["pnl_r"].isna().astype(int)
    df["pnl_r0"] = df["pnl_r"].fillna(0.0).astype(float)
    return df


def load_o5_microstructure(trading_days: list[pd.Timestamp]) -> pd.DataFrame:
    windows = []
    for td in trading_days:
        start_utc, end_utc = orb_utc_window(td.date(), SESSION, FEATURE_APERTURE)
        windows.append(
            {
                "trading_day": td.date(),
                "window_start_utc": start_utc,
                "window_end_utc": end_utc,
            }
        )
    windows_df = pd.DataFrame(windows)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        con.register("o5_windows", windows_df)
        bars = con.execute(
            f"""
            SELECT
                w.trading_day,
                b.ts_utc,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume
            FROM bars_1m b
            JOIN o5_windows w
              ON b.ts_utc >= w.window_start_utc
             AND b.ts_utc < w.window_end_utc
            WHERE b.symbol = '{INSTRUMENT}'
            ORDER BY w.trading_day, b.ts_utc
            """
        ).df()
    finally:
        con.close()
    if bars.empty:
        raise RuntimeError("bars_1m microstructure query returned zero rows")

    rows: list[dict] = []
    for trading_day, grp in bars.groupby("trading_day", sort=True):
        grp = grp.sort_values("ts_utc")
        open_5 = float(grp.iloc[0]["open"])
        close_5 = float(grp.iloc[-1]["close"])
        high_5 = float(grp["high"].max())
        low_5 = float(grp["low"].min())
        volume_5 = float(grp["volume"].sum())
        o5_size = high_5 - low_5
        o5_clv = _compute_clv(open_5, high_5, low_5, close_5)
        rows.append(
            {
                "trading_day": pd.Timestamp(trading_day),
                "o5_open": open_5,
                "o5_close": close_5,
                "o5_high": high_5,
                "o5_low": low_5,
                "o5_size": o5_size,
                "o5_volume": volume_5,
                "o5_clv": o5_clv,
            }
        )
    return pd.DataFrame(rows)


def build_panel() -> pd.DataFrame:
    lane_df = load_target_lane()
    micro_df = load_o5_microstructure(sorted(lane_df["trading_day"].tolist()))
    df = lane_df.merge(micro_df, on="trading_day", how="left", validate="one_to_one")
    if df["o5_clv"].isna().all():
        raise RuntimeError("all O5 microstructure rows are missing")
    df["feature_fire"] = [
        _feature_fire_for_row(bd, clv) for bd, clv in zip(df[f"orb_{SESSION}_break_dir"], df["o5_clv"], strict=False)
    ]
    df["feature_fire"] = df["feature_fire"].astype(int)
    df["o15_size_q3_proxy"] = 0
    df["o5_size_q3_proxy"] = 0
    return df


def _welch(on: pd.Series, off: pd.Series) -> tuple[float, float]:
    if len(on) < 2 or len(off) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(on.astype(float), off.astype(float), equal_var=False)
    return float(t), float(p)


def _t0_tautology(df: pd.DataFrame) -> tuple[float, str, dict[str, float]]:
    is_df = df.loc[df["is_is"]].copy()
    q67_o5 = float(np.nanpercentile(is_df["o5_size"].astype(float), 67))
    q67_o15 = float(np.nanpercentile(is_df[f"orb_{SESSION}_size"].astype(float), 67))
    proxies = pd.DataFrame(
        {
            "feature_fire": is_df["feature_fire"].astype(float),
            "o5_size_q3": (is_df["o5_size"].astype(float) >= q67_o5).astype(float),
            "o15_size_q3": (is_df[f"orb_{SESSION}_size"].astype(float) >= q67_o15).astype(float),
            "atr70_fire": (is_df["atr_20_pct"].astype(float) >= 70).astype(float),
            "ovn80_fire": (is_df["overnight_range_pct"].astype(float) >= 80).astype(float),
        }
    )
    corrs: dict[str, float] = {}
    for col in ["o5_size_q3", "o15_size_q3", "atr70_fire", "ovn80_fire"]:
        corr = proxies["feature_fire"].corr(proxies[col])
        corrs[col] = float(corr) if not pd.isna(corr) else 0.0
    max_filter = max(corrs, key=lambda k: abs(corrs[k]))
    max_corr = abs(corrs[max_filter])
    return max_corr, max_filter, corrs


def _t1_accounting_consistency(df: pd.DataFrame) -> tuple[float, float]:
    is_df = df.loc[df["is_is"]].copy()
    on_resolved = is_df.loc[(is_df["feature_fire"] == 1) & (is_df["is_scratch"] == 0), "pnl_r"].astype(float)
    off_resolved = is_df.loc[(is_df["feature_fire"] == 0) & (is_df["is_scratch"] == 0), "pnl_r"].astype(float)
    resolved_delta = float(on_resolved.mean() - off_resolved.mean())
    on_all = is_df.loc[is_df["feature_fire"] == 1, "pnl_r0"].astype(float)
    off_all = is_df.loc[is_df["feature_fire"] == 0, "pnl_r0"].astype(float)
    scratch_inclusive_delta = float(on_all.mean() - off_all.mean())
    return resolved_delta, scratch_inclusive_delta


def _t2_parent_lane_quality(df: pd.DataFrame) -> tuple[float, float]:
    is_df = df.loc[df["is_is"]].copy()
    lane_resolved = is_df.loc[is_df["is_scratch"] == 0, "pnl_r"].astype(float)
    lane_all = is_df["pnl_r0"].astype(float)
    return float(lane_resolved.mean()), float(lane_all.mean())


def _t3_oos_direction(df: pd.DataFrame) -> tuple[int, float | None, float | None]:
    oos_df = df.loc[df["is_oos"]].copy()
    on = oos_df.loc[(oos_df["feature_fire"] == 1) & (oos_df["is_scratch"] == 0), "pnl_r"].astype(float)
    off = oos_df.loc[(oos_df["feature_fire"] == 0) & (oos_df["is_scratch"] == 0), "pnl_r"].astype(float)
    if len(on) == 0 or len(off) == 0:
        return int(len(on)), None, None
    return int(len(on)), float(on.mean()), float(off.mean())


def _t6_null_floor(df: pd.DataFrame, B: int = 1000) -> tuple[float, float]:
    is_df = df.loc[df["is_is"] & (df["is_scratch"] == 0)].copy().reset_index(drop=True)
    observed = float(
        is_df.loc[is_df["feature_fire"] == 1, "pnl_r"].astype(float).mean()
        - is_df.loc[is_df["feature_fire"] == 0, "pnl_r"].astype(float).mean()
    )
    values = is_df["pnl_r"].astype(float).to_numpy()
    feature = is_df["feature_fire"].to_numpy()
    n_on = int(feature.sum())
    if n_on < 30 or (len(values) - n_on) < 30:
        return float("nan"), observed
    more_extreme = 0
    for _ in range(B):
        shuffled = RNG.permutation(values)
        sample_on = shuffled[:n_on]
        sample_off = shuffled[n_on:]
        delta = float(sample_on.mean() - sample_off.mean())
        if delta >= observed:
            more_extreme += 1
    p_val = (more_extreme + 1) / (B + 1)
    return float(p_val), observed


def _t7_per_year_delta(df: pd.DataFrame) -> tuple[int, int, dict[int, float]]:
    is_df = df.loc[df["is_is"] & (df["is_scratch"] == 0)].copy()
    positive = 0
    testable = 0
    year_delta: dict[int, float] = {}
    for year in sorted(is_df["year"].unique()):
        yr = is_df.loc[is_df["year"] == year].copy()
        on = yr.loc[yr["feature_fire"] == 1, "pnl_r"].astype(float)
        off = yr.loc[yr["feature_fire"] == 0, "pnl_r"].astype(float)
        if len(on) < 5 or len(off) < 5:
            year_delta[int(year)] = float("nan")
            continue
        value = float(on.mean() - off.mean())
        year_delta[int(year)] = value
        testable += 1
        if value > 0:
            positive += 1
    return positive, testable, year_delta


def _t8_scratch_bias(df: pd.DataFrame) -> tuple[float, float]:
    is_df = df.loc[df["is_is"]].copy()
    on = is_df.loc[is_df["feature_fire"] == 1, "is_scratch"].astype(float)
    off = is_df.loc[is_df["feature_fire"] == 0, "is_scratch"].astype(float)
    return float(on.mean()), float(off.mean())


def _build_verdict(
    resolved_delta: float,
    t_is: float,
    p_is: float,
    n_on_oos: int,
    expr_on_oos: float | None,
    expr_off_oos: float | None,
    taut_corr: float,
    scratch_delta: float,
) -> str:
    delta_oos = None if expr_on_oos is None or expr_off_oos is None else (expr_on_oos - expr_off_oos)
    if taut_corr > 0.70:
        return "KILL"
    if not np.isfinite(t_is) or abs(t_is) < 3.79 or p_is >= 0.05:
        return "KILL"
    if resolved_delta <= 0 or scratch_delta <= 0:
        return "KILL"
    if n_on_oos < 15:
        return "PARK"
    if delta_oos is None or delta_oos <= 0:
        return "KILL"
    return "KEEP"


def main() -> int:
    prereg = _load_prereg(PREREG_PATH)
    df = build_panel()

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    ROW_CSV.parent.mkdir(parents=True, exist_ok=True)

    row_df = df[
        [
            "trading_day",
            "pnl_r",
            "pnl_r0",
            "outcome",
            "is_is",
            "is_oos",
            "is_scratch",
            f"orb_{SESSION}_break_dir",
            f"orb_{SESSION}_size",
            "o5_open",
            "o5_close",
            "o5_high",
            "o5_low",
            "o5_size",
            "o5_volume",
            "o5_clv",
            "feature_fire",
            "lane_fire",
        ]
    ].copy()
    row_df.to_csv(REPO_ROOT / ROW_CSV, index=False)

    is_df = df.loc[df["is_is"]].copy()
    oos_df = df.loc[df["is_oos"]].copy()
    on_is = is_df.loc[(is_df["feature_fire"] == 1) & (is_df["is_scratch"] == 0), "pnl_r"].astype(float)
    off_is = is_df.loc[(is_df["feature_fire"] == 0) & (is_df["is_scratch"] == 0), "pnl_r"].astype(float)
    t_is, p_is = _welch(on_is, off_is)
    resolved_delta, scratch_delta = _t1_accounting_consistency(df)
    lane_resolved, lane_all = _t2_parent_lane_quality(df)
    n_on_oos, expr_on_oos, expr_off_oos = _t3_oos_direction(df)
    taut_corr, taut_against, taut_corrs = _t0_tautology(df)
    null_p, observed_delta = _t6_null_floor(df)
    pos_years, total_years, year_delta = _t7_per_year_delta(df)
    scratch_on, scratch_off = _t8_scratch_bias(df)
    verdict = _build_verdict(
        resolved_delta=resolved_delta,
        t_is=t_is,
        p_is=p_is,
        n_on_oos=n_on_oos,
        expr_on_oos=expr_on_oos,
        expr_off_oos=expr_off_oos,
        taut_corr=taut_corr,
        scratch_delta=scratch_delta,
    )

    expr_on_is = float(on_is.mean()) if len(on_is) else float("nan")
    expr_off_is = float(off_is.mean()) if len(off_is) else float("nan")
    oos_delta = None if expr_on_oos is None or expr_off_oos is None else (expr_on_oos - expr_off_oos)
    headers = _build_header(prereg)
    lines = headers + [
        f"**Verdict:** `{verdict}`",
        "",
        "## Scope",
        "",
        f"- instrument: `{INSTRUMENT}`",
        f"- session: `{SESSION}`",
        f"- target lane: `O{TARGET_APERTURE} E2 RR{RR_TARGET} CB{CONFIRM_BARS} {FILTER_KEY}`",
        f"- confluence feature: `O{FEATURE_APERTURE}` close-location-in-range aligned to O15 break direction",
        f"- OOS window: `{OOS_START}` through `{OOS_END - pd.Timedelta(days=1)}`",
        "",
        "## Truth / Calculation Check",
        "",
        "- canonical target lane membership delegated through `research.filter_utils.filter_signal` -> `trading_app.config.ALL_FILTERS['VWAP_MID_ALIGNED']`",
        "- O5 confluence computed from raw `bars_1m` over canonical `pipeline.dst.orb_utc_window(trading_day, 'US_DATA_1000', 5)`",
        "- no O5/O15 break-delay, break-bar, outcome, MAE, or MFE columns used in the feature",
        "- result is local to this lane and does NOT claim anything about O15 as a class",
        "",
        "## Results",
        "",
        f"- lane rows total: `{len(df)}` | IS: `{int(is_df.shape[0])}` | OOS: `{int(oos_df.shape[0])}`",
        f"- feature fire rows IS resolved: `{len(on_is)}` | off rows IS resolved: `{len(off_is)}`",
        f"- lane ExpR IS resolved: `{_fmt(lane_resolved)}` | scratch-inclusive: `{_fmt(lane_all)}`",
        f"- on-signal ExpR IS resolved: `{_fmt(expr_on_is)}`",
        f"- off-signal ExpR IS resolved: `{_fmt(expr_off_is)}`",
        f"- delta IS resolved: `{_fmt(resolved_delta)}` | scratch-inclusive: `{_fmt(scratch_delta)}`",
        f"- Welch t/p: `t={_fmt(t_is, 3)}` `p={_fmt(p_is, 4)}`",
        f"- OOS on-signal N resolved: `{n_on_oos}` | delta OOS resolved: `{_fmt(oos_delta)}`",
        f"- T0 tautology max |corr|: `{_fmt(taut_corr, 3)}` vs `{taut_against}`",
        f"- T6 null-floor p: `{_fmt(null_p, 4)}` | observed delta: `{_fmt(observed_delta)}`",
        f"- T7 positive IS years: `{pos_years}/{total_years}`",
        f"- T8 scratch rate on/off: `{_fmt(scratch_on, 3)}` / `{_fmt(scratch_off, 3)}`",
        "",
        "## Year Delta",
        "",
        "| Year | Delta |",
        "|---|---:|",
    ]
    for year in sorted(year_delta):
        lines.append(f"| {year} | {_fmt(year_delta[year])} |")
    lines += [
        "",
        "## T0 Proxy Correlations",
        "",
        "| Proxy | corr |",
        "|---|---:|",
    ]
    for key, val in taut_corrs.items():
        lines.append(f"| {key} | {_fmt(val, 3)} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- This is a nested same-session ORB-state test, not an aperture-family reopen.",
        "- A failure here does not kill other sessions, assets, O15 lanes, or other O5-derived features.",
        "- A pass here would still only support this exact lane/feature/role combination.",
    ]
    (REPO_ROOT / RESULT_DOC).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"result_doc={RESULT_DOC}")
    print(f"row_csv={ROW_CSV}")
    print(f"verdict={verdict}")
    print(f"is_delta={resolved_delta:.6f}")
    print(f"oos_delta={(float('nan') if oos_delta is None else oos_delta):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
