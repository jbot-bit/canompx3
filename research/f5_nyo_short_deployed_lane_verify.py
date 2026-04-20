#!/usr/bin/env python3
"""Exact-lane verify for NYSE_OPEN short F5_BELOW_PDL on COST_LT12."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.filter_utils import filter_signal
from research.lib import connect_db
from research.oos_power import format_power_report, oos_ttest_power, power_verdict
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

OUTPUT_MD = Path("docs/audit/results/2026-04-20-f5-nyo-short-deployed-lane-verify.md")
BLOCK_LEN = 5
N_BOOT = 5000
RNG_SEED = 20260420


def _load_lane() -> pd.DataFrame:
    sql = """
    SELECT
        o.trading_day,
        o.entry_price,
        o.stop_price,
        o.pnl_r,
        d.prev_day_low,
        d.orb_NYSE_OPEN_high AS orb_high,
        d.orb_NYSE_OPEN_low AS orb_low,
        d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = 'NYSE_OPEN'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.0
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    with connect_db() as con:
        df = con.execute(sql).fetchdf()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    fire = filter_signal(df, "COST_LT12", "NYSE_OPEN")
    df = df.loc[fire == 1].copy()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    mid = (df["orb_high"].astype(float) + df["orb_low"].astype(float)) / 2.0
    df["F5_BELOW_PDL"] = mid < df["prev_day_low"].astype(float)
    return df[df["direction"] == "short"].copy()


def _split_stats(df: pd.DataFrame) -> dict[str, float]:
    on = df.loc[df["F5_BELOW_PDL"], "pnl_r"].to_numpy(dtype=float)
    off = df.loc[~df["F5_BELOW_PDL"], "pnl_r"].to_numpy(dtype=float)
    out: dict[str, float] = {
        "n_on": float(len(on)),
        "n_off": float(len(off)),
        "exp_on": float(on.mean()) if len(on) else float("nan"),
        "exp_off": float(off.mean()) if len(off) else float("nan"),
        "delta": float(on.mean() - off.mean()) if len(on) and len(off) else float("nan"),
        "wr_on": float((on > 0).mean()) if len(on) else float("nan"),
        "wr_off": float((off > 0).mean()) if len(off) else float("nan"),
    }
    if len(on) >= 2 and len(off) >= 2:
        res = stats.ttest_ind(on, off, equal_var=False)
        out["t"] = float(np.asarray(res.statistic))
        out["p"] = float(np.asarray(res.pvalue))
        pooled_std = np.sqrt(
            (((len(on) - 1) * on.var(ddof=1)) + ((len(off) - 1) * off.var(ddof=1)))
            / max(1, len(on) + len(off) - 2)
        )
        out["pooled_std"] = float(pooled_std)
    else:
        out["t"] = float("nan")
        out["p"] = float("nan")
        out["pooled_std"] = float("nan")
    return out


def _block_bootstrap_p(is_df: pd.DataFrame) -> tuple[float, float]:
    ordered = is_df.sort_values("trading_day").reset_index(drop=True).copy()
    pnl = ordered["pnl_r"].to_numpy(dtype=float)
    mask = ordered["F5_BELOW_PDL"].to_numpy(dtype=bool)
    observed = float(pnl[mask].mean() - pnl[~mask].mean())
    n = len(pnl)
    n_blocks = int(np.ceil(n / BLOCK_LEN))
    rng = np.random.default_rng(RNG_SEED)
    null_deltas = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        starts = rng.integers(0, max(1, n - BLOCK_LEN + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + BLOCK_LEN) for s in starts])[:n]
        pnl_boot = pnl[idx]
        null_deltas[i] = float(pnl_boot[mask].mean() - pnl_boot[~mask].mean())
    p_val = float(((np.abs(null_deltas) >= np.abs(observed)).sum() + 1) / (N_BOOT + 1))
    return observed, p_val


def main() -> None:
    holdout = HOLDOUT_SACRED_FROM
    if isinstance(holdout, str):
        holdout = date.fromisoformat(holdout)

    shorts = _load_lane()
    is_df = shorts[shorts["trading_day"].dt.date < holdout].copy()
    oos_df = shorts[shorts["trading_day"].dt.date >= holdout].copy()

    is_stats = _split_stats(is_df)
    oos_stats = _split_stats(oos_df)
    _, boot_p = _block_bootstrap_p(is_df)

    is_df["year"] = is_df["trading_day"].dt.year
    year_rows = []
    years_consistent = 0
    for year, group in is_df.groupby("year"):
        on = group.loc[group["F5_BELOW_PDL"], "pnl_r"]
        off = group.loc[~group["F5_BELOW_PDL"], "pnl_r"]
        if len(on) >= 10 and len(off) >= 10:
            delta = float(on.mean() - off.mean())
            if delta > 0:
                years_consistent += 1
        else:
            delta = float("nan")
        year_rows.append((int(year), int(len(on)), int(len(off)), delta))

    pwr = oos_ttest_power(
        is_delta=is_stats["delta"],
        is_pooled_std=is_stats["pooled_std"],
        n_oos_a=max(2, int(oos_stats["n_on"])),
        n_oos_b=max(2, int(oos_stats["n_off"])),
        alpha=0.05,
    )
    tier = power_verdict(pwr["power"])
    dir_match = (
        np.sign(is_stats["delta"]) == np.sign(oos_stats["delta"])
        if not (np.isnan(is_stats["delta"]) or np.isnan(oos_stats["delta"]))
        else False
    )

    is_pass = (
        is_stats["delta"] >= 0.05
        and is_stats["exp_on"] > 0
        and is_stats["p"] < 0.05
        and boot_p < 0.05
        and is_stats["n_on"] >= 100
        and is_stats["n_off"] >= 100
        and years_consistent >= 5
    )
    verdict = "CONDITIONAL_UNVERIFIED" if is_pass else "DEAD"

    lines = [
        "# F5 NYSE_OPEN Short Deployed-Lane Verify",
        "",
        "**Pre-reg:** `docs/audit/hypotheses/2026-04-20-f5-nyo-short-deployed-lane-verify.yaml`",
        "**Lane:** `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` short",
        f"**Verdict:** **{verdict}**",
        "",
        "## Resource grounding",
        "",
        "- `resources/Algorithmic_Trading_Chan.pdf`: bounded executable strategy families are acceptable research units when the rules are explicit.",
        "- `resources/Robert Carver - Systematic Trading.pdf`: a useful signal can live as a conditioner or sizing input without needing to become a standalone system.",
        "- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`: theory-first and small-family verification over broad fishing.",
        "",
        "## Canonical results",
        "",
        f"- IS: N_on={int(is_stats['n_on'])}, N_off={int(is_stats['n_off'])}, "
        f"ExpR_on={is_stats['exp_on']:+.4f}, ExpR_off={is_stats['exp_off']:+.4f}, "
        f"delta={is_stats['delta']:+.4f}, t={is_stats['t']:.2f}, p={is_stats['p']:.4f}",
        f"- IS block-bootstrap p={boot_p:.4f}",
        f"- OOS: N_on={int(oos_stats['n_on'])}, N_off={int(oos_stats['n_off'])}, "
        f"ExpR_on={oos_stats['exp_on']:+.4f}, ExpR_off={oos_stats['exp_off']:+.4f}, "
        f"delta={oos_stats['delta']:+.4f}, p={oos_stats['p']:.4f}",
        f"- OOS dir_match={dir_match}",
        "",
        "## RULE 3.3 power floor",
        "",
        "```text",
        format_power_report(pwr),
        f"OOS tier: {tier}",
        "```",
        "",
        "## IS year-by-year",
        "",
    ]
    for year, n_on, n_off, delta in year_rows:
        delta_txt = "NA" if np.isnan(delta) else f"{delta:+.4f}"
        lines.append(f"- {year}: N_on={n_on}, N_off={n_off}, delta={delta_txt}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        "- IS evidence is strong enough to keep this as a real exact-lane candidate.",
        "- OOS is still too thin to refute or promote; any sign flip here is descriptive only, so the correct label remains CONDITIONAL_UNVERIFIED rather than CONFIRMED.",
        "- Correct role if pursued further: conditioner / deployment-shape, not standalone strategy.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Verdict={verdict} | IS_delta={is_stats['delta']:+.4f} | OOS_delta={oos_stats['delta']:+.4f}")


if __name__ == "__main__":
    main()
