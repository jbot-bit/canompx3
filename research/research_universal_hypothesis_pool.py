#!/usr/bin/env python3
"""Universal pooled hypothesis test (single hypothesis, massive N).

Hypothesis U1 (pre-registered):
Trades are higher quality when breakout is both:
1) fast (break_delay_min <= 30), and
2) continuation-type (break_bar_continues == True).

This is tested pooled across all symbols/sessions/entry models in orb_outcomes (orb_minutes=5),
with no lookahead features only.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"

N_PERM = 1000
SEED = 123


def make_case(stem: str, labels: list[str]) -> str:
    parts = ["CASE o.orb_label"]
    for lbl in labels:
        safe = lbl.replace("'", "''")
        parts.append(f" WHEN '{safe}' THEN d.orb_{safe}_{stem}")
    parts.append(" ELSE NULL END")
    return "\n".join(parts)


def perm_p(mask: np.ndarray, years: np.ndarray, pnl: np.ndarray, obs: float, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    ge = 0
    valid = 0
    uniq = np.unique(years)

    for _ in range(n_perm):
        sh = mask.copy()
        for y in uniq:
            idx = np.where(years == y)[0]
            sh[idx] = rng.permutation(sh[idx])

        n_on = int(sh.sum())
        n_off = len(sh) - n_on
        if n_on < 200 or n_off < 200:
            continue

        up = float(pnl[sh].mean() - pnl[~sh].mean())
        valid += 1
        if up >= obs:
            ge += 1

    if valid == 0:
        return np.nan
    return float((ge + 1) / (valid + 1))


def main() -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    labels_df = con.execute(
        """
        SELECT DISTINCT orb_label
        FROM orb_outcomes
        WHERE orb_minutes=5
        ORDER BY 1
        """
    ).fetchdf()
    labels = labels_df["orb_label"].tolist()

    c_delay = make_case("break_delay_min", labels)
    c_cont = make_case("break_bar_continues", labels)

    q = f"""
    SELECT
      o.symbol,
      o.trading_day,
      o.orb_label,
      o.entry_model,
      o.confirm_bars,
      o.rr_target,
      o.pnl_r,
      {c_delay} AS break_delay,
      {c_cont}  AS break_cont
    FROM orb_outcomes o
    JOIN daily_features d
      ON d.symbol=o.symbol
     AND d.trading_day=o.trading_day
     AND d.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.pnl_r IS NOT NULL
    """

    df = con.execute(q).fetchdf()
    con.close()

    if df.empty:
        print("No rows.")
        return 0

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year

    # usable rows only
    df = df[df["break_delay"].notna() & df["break_cont"].notna()].copy()
    if len(df) < 1000:
        print("Too few usable rows.")
        return 0

    # U1 condition (fixed, no tuning)
    cond = (df["break_cont"] == True) & (df["break_delay"] <= 30)

    on = df.loc[cond, "pnl_r"]
    off = df.loc[~cond, "pnl_r"]

    if len(on) < 200 or len(off) < 200:
        print("Insufficient ON/OFF sample for pooled test.")
        return 0

    avg_on = float(on.mean())
    avg_off = float(off.mean())
    uplift = avg_on - avg_off
    wr_on = float((on > 0).mean())
    wr_off = float((off > 0).mean())

    years = df["year"].values
    pnl = df["pnl_r"].values
    m = cond.values.astype(bool)

    pval = perm_p(m.copy(), years, pnl, uplift, n_perm=N_PERM, seed=SEED)

    # consistency by symbol/session/model
    sym = (
        df.groupby("symbol")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "n_on": int(cond.loc[g.index].sum()),
            "avg_on": float(g.loc[cond.loc[g.index], "pnl_r"].mean()) if cond.loc[g.index].sum() > 0 else np.nan,
            "avg_off": float(g.loc[~cond.loc[g.index], "pnl_r"].mean()) if (~cond.loc[g.index]).sum() > 0 else np.nan,
        }))
        .reset_index()
    )
    sym["uplift"] = sym["avg_on"] - sym["avg_off"]

    ses = (
        df.groupby("orb_label")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "n_on": int(cond.loc[g.index].sum()),
            "avg_on": float(g.loc[cond.loc[g.index], "pnl_r"].mean()) if cond.loc[g.index].sum() > 0 else np.nan,
            "avg_off": float(g.loc[~cond.loc[g.index], "pnl_r"].mean()) if (~cond.loc[g.index]).sum() > 0 else np.nan,
        }))
        .reset_index()
    )
    ses["uplift"] = ses["avg_on"] - ses["avg_off"]

    mdl = (
        df.groupby(["entry_model", "confirm_bars", "rr_target"])
        .apply(lambda g: pd.Series({
            "n": len(g),
            "n_on": int(cond.loc[g.index].sum()),
            "avg_on": float(g.loc[cond.loc[g.index], "pnl_r"].mean()) if cond.loc[g.index].sum() > 0 else np.nan,
            "avg_off": float(g.loc[~cond.loc[g.index], "pnl_r"].mean()) if (~cond.loc[g.index]).sum() > 0 else np.nan,
        }))
        .reset_index()
    )
    mdl["uplift"] = mdl["avg_on"] - mdl["avg_off"]

    # positivity ratios (with min support)
    sym_use = sym[(sym["n_on"] >= 200) & ((sym["n"] - sym["n_on"]) >= 200)]
    ses_use = ses[(ses["n_on"] >= 200) & ((ses["n"] - ses["n_on"]) >= 200)]
    mdl_use = mdl[(mdl["n_on"] >= 200) & ((mdl["n"] - mdl["n_on"]) >= 200)]

    sym_pos = float((sym_use["uplift"] > 0).mean()) if len(sym_use) else np.nan
    ses_pos = float((ses_use["uplift"] > 0).mean()) if len(ses_use) else np.nan
    mdl_pos = float((mdl_use["uplift"] > 0).mean()) if len(mdl_use) else np.nan

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([
        {
            "hypothesis": "U1_fast30_and_continuation",
            "n_total": int(len(df)),
            "n_on": int(len(on)),
            "on_rate": float(len(on) / len(df)),
            "avg_on": avg_on,
            "avg_off": avg_off,
            "uplift": uplift,
            "wr_on": wr_on,
            "wr_off": wr_off,
            "perm_p": pval,
            "sym_pos_ratio": sym_pos,
            "session_pos_ratio": ses_pos,
            "model_pos_ratio": mdl_pos,
        }
    ])

    p_sum = out_dir / "universal_hypothesis_u1_summary.csv"
    p_sym = out_dir / "universal_hypothesis_u1_by_symbol.csv"
    p_ses = out_dir / "universal_hypothesis_u1_by_session.csv"
    p_mdl = out_dir / "universal_hypothesis_u1_by_model.csv"
    p_md = out_dir / "universal_hypothesis_u1_report.md"

    summary.to_csv(p_sum, index=False)
    sym.sort_values("uplift", ascending=False).to_csv(p_sym, index=False)
    ses.sort_values("uplift", ascending=False).to_csv(p_ses, index=False)
    mdl.sort_values("uplift", ascending=False).to_csv(p_mdl, index=False)

    # strict verdict for this hypothesis
    verdict = "KILL"
    if (
        avg_on >= 0.20 and uplift >= 0.20 and
        pd.notna(pval) and pval <= 0.01 and
        pd.notna(sym_pos) and sym_pos >= 0.60 and
        pd.notna(ses_pos) and ses_pos >= 0.60 and
        pd.notna(mdl_pos) and mdl_pos >= 0.60
    ):
        verdict = "PROMOTE"

    lines = [
        "# Universal Hypothesis U1 Report",
        "",
        "Hypothesis: fast breakout (<=30m) + continuation bar implies better trade quality.",
        "Pooled across all symbols/sessions/models (orb_minutes=5).",
        "",
        f"N total: {len(df)}",
        f"N on: {len(on)} ({len(on)/len(df):.1%})",
        f"avg_on: {avg_on:+.4f}",
        f"avg_off: {avg_off:+.4f}",
        f"uplift: {uplift:+.4f}",
        f"WR on/off: {wr_on:.1%}/{wr_off:.1%}",
        f"Permutation p-value: {pval:.6f}",
        f"Positive-ratio by symbol/session/model: {sym_pos:.2f}/{ses_pos:.2f}/{mdl_pos:.2f}",
        "",
        f"Verdict: {verdict}",
    ]

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_sum}")
    print(f"Saved: {p_sym}")
    print(f"Saved: {p_ses}")
    print(f"Saved: {p_mdl}")
    print(f"Saved: {p_md}")
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
