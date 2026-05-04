#!/usr/bin/env python3
"""Extended fast lead-lag scan including Russell (M2K) and FX (M6E).

Universe: MES, MNQ, M2K, M6E
Slice: E1 / CB2 / RR2.5 (orb_minutes=5)

Condition ON:
- leader break_dir == follower break_dir (long/short)
- leader_break_ts <= follower entry_ts (no-lookahead)

Outputs:
- research/output/fast_lead_lag_extended_summary.csv
- research/output/fast_lead_lag_extended_notes.md
"""

from __future__ import annotations

import re
from pathlib import Path
import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"
UNIVERSE = ["MES", "MNQ", "M2K", "M6E"]
ENTRY_MODEL = "E1"
CONFIRM_BARS = 2
RR_TARGET = 2.5
MIN_LABEL_N = 800
MIN_ON = 80


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe label: {label}")
    return label


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)

    # session labels with enough data for this slice
    lab = con.execute(
        """
        SELECT symbol, orb_label, COUNT(*) AS n
        FROM orb_outcomes
        WHERE orb_minutes=5
          AND entry_model=?
          AND confirm_bars=?
          AND rr_target=?
          AND symbol IN ('MES','MNQ','M2K','M6E')
        GROUP BY 1,2
        HAVING COUNT(*) >= ?
        ORDER BY symbol, n DESC
        """,
        [ENTRY_MODEL, CONFIRM_BARS, RR_TARGET, MIN_LABEL_N],
    ).fetchdf()

    if lab.empty:
        print("No labels met minimum count.")
        return 0

    # only labels with matching daily_features columns
    cols = {r[1] for r in con.execute("PRAGMA table_info('daily_features')").fetchall()}
    sess_by_sym: dict[str, list[str]] = {}
    for r in lab.itertuples(index=False):
        lbl = safe_label(r.orb_label)
        if f"orb_{lbl}_break_dir" in cols and f"orb_{lbl}_break_ts" in cols:
            sess_by_sym.setdefault(r.symbol, []).append(lbl)

    rows = []

    for lsym in UNIVERSE:
        for fsym in UNIVERSE:
            for lsess in sess_by_sym.get(lsym, []):
                for fsess in sess_by_sym.get(fsym, []):
                    lcol_dir = f"orb_{safe_label(lsess)}_break_dir"
                    lcol_ts = f"orb_{safe_label(lsess)}_break_ts"
                    fcol_dir = f"orb_{safe_label(fsess)}_break_dir"

                    q = f"""
                    WITH base AS (
                      SELECT
                        o.pnl_r,
                        o.entry_ts,
                        EXTRACT(YEAR FROM o.trading_day) AS y,
                        df_f.{fcol_dir} AS f_dir,
                        df_l.{lcol_dir} AS l_dir,
                        df_l.{lcol_ts}  AS l_ts
                      FROM orb_outcomes o
                      JOIN daily_features df_f
                        ON df_f.symbol=o.symbol
                       AND df_f.trading_day=o.trading_day
                       AND df_f.orb_minutes=o.orb_minutes
                      JOIN daily_features df_l
                        ON df_l.symbol='{lsym}'
                       AND df_l.trading_day=o.trading_day
                       AND df_l.orb_minutes=o.orb_minutes
                      WHERE o.orb_minutes=5
                        AND o.symbol='{fsym}'
                        AND o.orb_label='{fsess}'
                        AND o.entry_model='{ENTRY_MODEL}'
                        AND o.confirm_bars={CONFIRM_BARS}
                        AND o.rr_target={RR_TARGET}
                        AND o.pnl_r IS NOT NULL
                        AND o.entry_ts IS NOT NULL
                    )
                    SELECT
                      COUNT(*) AS n_base,
                      SUM(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN 1 ELSE 0 END) AS n_on,
                      AVG(pnl_r) AS avg_base,
                      AVG(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN pnl_r END) AS avg_on,
                      AVG(CASE WHEN NOT (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN pnl_r END) AS avg_off,
                      AVG(CASE WHEN (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN CASE WHEN pnl_r>0 THEN 1.0 ELSE 0.0 END END) AS wr_on,
                      AVG(CASE WHEN NOT (l_dir IN ('long','short') AND f_dir IN ('long','short') AND l_ts IS NOT NULL AND l_ts<=entry_ts AND l_dir=f_dir) THEN CASE WHEN pnl_r>0 THEN 1.0 ELSE 0.0 END END) AS wr_off
                    FROM base
                    """

                    r = con.execute(q).fetchone()
                    if r is None:
                        continue
                    n_base, n_on, avg_base, avg_on, avg_off, wr_on, wr_off = r
                    n_base = int(n_base or 0)
                    n_on = int(n_on or 0)
                    n_off = n_base - n_on

                    if n_on < MIN_ON or n_off < MIN_ON:
                        continue
                    if avg_on is None or avg_off is None:
                        continue

                    rows.append(
                        {
                            "leader": f"{lsym}_{lsess}",
                            "follower": f"{fsym}_{fsess}",
                            "n_base": n_base,
                            "n_on": n_on,
                            "on_rate": n_on / n_base if n_base else None,
                            "avg_r_base": float(avg_base),
                            "avg_r_on": float(avg_on),
                            "avg_r_off": float(avg_off),
                            "uplift_on_vs_off": float(avg_on - avg_off),
                            "wr_on": float(wr_on) if wr_on is not None else None,
                            "wr_off": float(wr_off) if wr_off is not None else None,
                        }
                    )

    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "fast_lead_lag_extended_summary.csv"
    p_md = out_dir / "fast_lead_lag_extended_notes.md"

    if not rows:
        p_md.write_text("# Fast Lead-Lag Extended\n\nNo rows met thresholds.", encoding="utf-8")
        print("No rows met thresholds.")
        return 0

    df = pd.DataFrame(rows).sort_values(["uplift_on_vs_off", "avg_r_on"], ascending=False)
    df.to_csv(p_csv, index=False)

    lines = [
        "# Fast Lead-Lag Extended (MES/MNQ/M2K/M6E)",
        "",
        f"- Slice: {ENTRY_MODEL}/CB{CONFIRM_BARS}/RR{RR_TARGET}",
        f"- Min label N: {MIN_LABEL_N}",
        f"- Min ON/OFF: {MIN_ON}",
        "- No-lookahead: leader_break_ts <= follower entry_ts",
        "",
        "## Top rows",
    ]

    for r in df.head(20).itertuples(index=False):
        lines.append(
            f"- {r.leader} -> {r.follower}: N_on={r.n_on}/{r.n_base}, avgR on/off {r.avg_r_on:+.4f}/{r.avg_r_off:+.4f}, Δ={r.uplift_on_vs_off:+.4f}, WR on/off {r.wr_on:.1%}/{r.wr_off:.1%}"
        )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(df.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
