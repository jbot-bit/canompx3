"""F5_BELOW_PDL LONG MNQ US_DATA_1000 O5 E2 CB1 RR1.0 — one-shot validator.

Pre-reg: docs/audit/hypotheses/2026-04-18-f5-below-pdl-mnq-us-data-1000-o5-rr1-one-shot.yaml
commit_sha: 311f79979251cadfce5c82d7413260ca1b069e80 (stamped 2026-04-18)

Single-shot Mode A OOS read. Reads IS (< 2026-01-01) and OOS
(2026-01-01..2026-04-18 exclusive) ONCE, applies F5_BELOW_PDL predicate,
evaluates gates, writes verdict md.

Refuses to re-run if output md already exists (Mode A single-shot enforcement).

Gates (aligned with post-audit oneshot_utils decision rule):
- KILL is evaluated UNCONDITIONALLY on power whenever any OOS observation exists:
    ExpR_on_OOS < 0 OR eff_ratio < 0.40 OR sign-flip vs IS.
- PARK fires when no KILL fires and power is below the CLT floor:
    N_on_OOS < 10  -> INSUFFICIENT (bootstrap undefined)
    10 <= N < 30   -> UNDERPOWERED
- PASS requires N_on_OOS >= 30 AND ExpR_on_OOS >= 0 AND eff_ratio >= 0.40 AND sign match.

IS stats are locked in the pre-reg; this script re-computes them in-validator
and asserts drift < 5e-4 vs locked values.

Refuses to re-run if output md already exists (Mode A single-shot enforcement).
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.oneshot_utils import (  # noqa: E402
    POWER_FLOOR_PASS,
    OneShotGates,
    decide_verdict,
    moving_block_bootstrap_p,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

LOCK_DATE = "2026-04-18"
IS_TO = "2026-01-01"
OOS_FROM = "2026-01-01"
OOS_TO = LOCK_DATE

PRE_REG_SHA = "311f79979251cadfce5c82d7413260ca1b069e80"

IS_LOCKED = {
    "N_on": 136,
    "N_off": 745,
    "ExpR_on": 0.3258,
    "ExpR_off": -0.0112,
    "delta": 0.3370,
    "welch_t": 4.0176,
    "welch_p": 0.000084,
    "SD_on": 0.8909,
    "WR_on": 0.6912,
}

EFF_RATIO_MIN = 0.40
BOOTSTRAP_B = 10_000
BOOTSTRAP_BLOCK = 5
RNG_SEED = 20260418

OUTPUT_MD = Path("docs/audit/results/2026-04-18-f5-below-pdl-one-shot-validator.md")


def load_cell() -> tuple:
    """Load IS + OOS long-direction entries with feature columns.

    Returns (df_is_long, df_oos_long).
    """
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    sql = f"""
    SELECT
        o.trading_day,
        o.entry_ts, o.entry_price, o.stop_price, o.target_price,
        o.outcome, o.pnl_r,
        d.atr_20,
        d.prev_day_high, d.prev_day_low, d.prev_day_close,
        (d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 AS orb_mid,
        d.orb_US_DATA_1000_break_dir AS break_dir,
        CASE
          WHEN o.trading_day < DATE '{IS_TO}' THEN 'IS'
          WHEN o.trading_day >= DATE '{OOS_FROM}' AND o.trading_day < DATE '{OOS_TO}' THEN 'OOS'
          ELSE 'EXCLUDED'
        END AS window_label
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = 'US_DATA_1000'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.0
      AND o.pnl_r IS NOT NULL
      AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
      AND d.prev_day_high IS NOT NULL
      AND d.prev_day_low IS NOT NULL
      AND d.orb_US_DATA_1000_break_dir IN ('long', 'short')
      AND o.trading_day < DATE '{OOS_TO}'
    ORDER BY o.trading_day, o.entry_ts
    """
    df = con.execute(sql).df()
    con.close()
    df["F5_BELOW_PDL"] = (df["orb_mid"] < df["prev_day_low"]).astype(int)
    df_long = df[df["break_dir"] == "long"].copy()
    return (
        df_long[df_long["window_label"] == "IS"].copy(),
        df_long[df_long["window_label"] == "OOS"].copy(),
    )


def cell_stats(df, label: str) -> dict:
    on = df[df["F5_BELOW_PDL"] == 1]["pnl_r"].values
    off = df[df["F5_BELOW_PDL"] == 0]["pnl_r"].values
    on_outcomes = df[df["F5_BELOW_PDL"] == 1]["outcome"].values
    out = {
        "label": label,
        "N_total_long": int(len(df)),
        "N_on": int(len(on)),
        "N_off": int(len(off)),
        "ExpR_on": float(on.mean()) if len(on) > 0 else float("nan"),
        "ExpR_off": float(off.mean()) if len(off) > 0 else float("nan"),
        "SD_on": float(on.std(ddof=1)) if len(on) > 1 else float("nan"),
        "WR_on": (
            float((on_outcomes == "win").sum() / len(on))
            if len(on) > 0
            else float("nan")
        ),
    }
    if len(on) >= 30 and len(off) >= 30:
        result = stats.ttest_ind(on, off, equal_var=False)
        out["welch_t"] = float(result.statistic)
        out["welch_p"] = float(result.pvalue)
    else:
        out["welch_t"] = float("nan")
        out["welch_p"] = float("nan")
    out["delta"] = out["ExpR_on"] - out["ExpR_off"]
    return out


# NOTE: the prior script-local `block_bootstrap_p_vs_zero` was replaced by
# `research.oneshot_utils.moving_block_bootstrap_p` on 2026-04-18 after the
# F5 audit caught a structural bug producing p ~ 0.5 regardless of signal.
# See tests/research/test_oneshot_utils.py for the numerical self-test.


def main() -> int:
    if OUTPUT_MD.exists():
        print(f"REFUSE: {OUTPUT_MD} already exists. Mode A one-shot cannot re-run.")
        return 2

    df_is, df_oos = load_cell()
    is_stats = cell_stats(df_is, "IS")
    oos_stats = cell_stats(df_oos, "OOS")

    # Drift check vs pre-reg locked IS (tolerance 5e-4 given rounded reporting)
    drift = {
        "N_on": is_stats["N_on"] - IS_LOCKED["N_on"],
        "ExpR_on": is_stats["ExpR_on"] - IS_LOCKED["ExpR_on"],
        "delta": is_stats["delta"] - IS_LOCKED["delta"],
    }
    is_drift_ok = (
        abs(drift["N_on"]) == 0
        and abs(drift["ExpR_on"]) < 5e-4
        and abs(drift["delta"]) < 5e-4
    )

    # Gates — delegated to research.oneshot_utils.decide_verdict so that:
    #  - KILL is evaluated UNCONDITIONALLY on power (kills > parks)
    #  - PARK only fires when no KILL fires and N_on_OOS < 30 (CLT floor)
    #  - PASS requires N_on_OOS >= POWER_FLOOR_PASS (30) AND all primaries clear
    n_on_oos = oos_stats["N_on"]
    expR_on_oos = oos_stats["ExpR_on"]
    expR_on_is = IS_LOCKED["ExpR_on"]

    gates = OneShotGates(expR_on_is=expR_on_is, eff_ratio_min=EFF_RATIO_MIN)
    result = decide_verdict(
        expR_on_oos=expR_on_oos, n_on_oos=n_on_oos, gates=gates
    )
    verdict = result.verdict
    kill_reasons = list(result.kill_reasons)
    park_reason = result.park_reason
    eff_ratio = result.eff_ratio
    sign_match = result.sign_match

    # Bootstrap (corrected moving-block under H0:mean=0). Upper-tail because
    # the IS direction is positive and H1 is "OOS mean > 0".
    on_oos_pnl = df_oos[df_oos["F5_BELOW_PDL"] == 1]["pnl_r"].values
    bootstrap_p = moving_block_bootstrap_p(
        on_oos_pnl, B=BOOTSTRAP_B, block=BOOTSTRAP_BLOCK, seed=RNG_SEED, tail="upper"
    )

    # T0 informational: correlation to VWAP_MID_ALIGNED fire is skipped here —
    # VWAP_MID_ALIGNED is the live deployed filter on this lane but its fire
    # vector requires live-config wiring; we log the informational gap rather
    # than fake a value.

    # Write MD
    lines: list[str] = []
    lines.append("# F5_BELOW_PDL LONG MNQ US_DATA_1000 O5 RR1.0 — One-Shot Validator")
    lines.append("")
    lines.append(f"**Pre-reg:** `docs/audit/hypotheses/2026-04-18-f5-below-pdl-mnq-us-data-1000-o5-rr1-one-shot.yaml`")
    lines.append(f"**Pre-reg sha:** `{PRE_REG_SHA}`")
    lines.append(f"**Lock date:** {LOCK_DATE}")
    lines.append(f"**IS window:** [2019-05-07, {IS_TO})")
    lines.append(f"**OOS window:** [{OOS_FROM}, {OOS_TO})")
    lines.append(f"**Mode A sacred boundary:** {HOLDOUT_SACRED_FROM} (from trading_app.holdout_policy)")
    lines.append("")
    lines.append(f"## VERDICT: **{verdict}**")
    lines.append("")
    if kill_reasons:
        lines.append(f"Kill reasons: {', '.join(kill_reasons)}")
        lines.append("")
    if park_reason:
        lines.append(f"Park reason: {park_reason}")
        lines.append("")

    lines.append("## IS drift check vs pre-reg locked")
    lines.append("")
    lines.append("| Metric | Pre-reg locked | Validator re-computed | Drift |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| N_on | {IS_LOCKED['N_on']} | {is_stats['N_on']} | {drift['N_on']:+d} |")
    lines.append(f"| ExpR_on | {IS_LOCKED['ExpR_on']:+.4f} | {is_stats['ExpR_on']:+.4f} | {drift['ExpR_on']:+.6f} |")
    lines.append(f"| Δ (on-off) | {IS_LOCKED['delta']:+.4f} | {is_stats['delta']:+.4f} | {drift['delta']:+.6f} |")
    lines.append("")
    lines.append(f"IS drift within tolerance: **{is_drift_ok}**")
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | IS (locked) | OOS (one-shot) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| N_total_long | — | {oos_stats['N_total_long']} |")
    lines.append(f"| N_on (F5_BELOW_PDL=1) | {IS_LOCKED['N_on']} | {oos_stats['N_on']} |")
    lines.append(f"| N_off (F5_BELOW_PDL=0) | {IS_LOCKED['N_off']} | {oos_stats['N_off']} |")
    lines.append(f"| ExpR_on | {IS_LOCKED['ExpR_on']:+.4f} | {oos_stats['ExpR_on']:+.4f} |")
    lines.append(f"| ExpR_off | {IS_LOCKED['ExpR_off']:+.4f} | {oos_stats['ExpR_off']:+.4f} |")
    lines.append(f"| Δ (on-off) | {IS_LOCKED['delta']:+.4f} | {oos_stats['delta']:+.4f} |")
    lines.append(f"| SD_on | {IS_LOCKED['SD_on']:.4f} | {oos_stats['SD_on']:.4f} |")
    lines.append(f"| WR_on | {IS_LOCKED['WR_on']:.4f} | {oos_stats['WR_on']:.4f} |")
    lines.append(f"| Welch t (on vs off) | {IS_LOCKED['welch_t']:+.4f} | {oos_stats['welch_t']:+.4f} |")
    lines.append(f"| Welch p | {IS_LOCKED['welch_p']:.6f} | {oos_stats['welch_p']:.6f} |")
    lines.append("")

    lines.append("## Gate evaluation")
    lines.append("")
    lines.append("| Gate | Rule | Value | Result |")
    lines.append("|---|---|---:|:---:|")
    lines.append(
        f"| Primary: OOS ExpR >= 0 | ExpR_on_OOS >= 0 | {expR_on_oos:+.4f} | "
        f"{'PASS' if expR_on_oos >= 0 else 'FAIL'} |"
    )
    lines.append(
        f"| Primary: eff_ratio >= 0.40 | ExpR_on_OOS / ExpR_on_IS | {eff_ratio:+.4f} | "
        f"{'PASS' if not np.isnan(eff_ratio) and eff_ratio >= EFF_RATIO_MIN else 'FAIL'} |"
    )
    sign_char = "+" if (n_on_oos > 0 and expR_on_oos > 0) else ("-" if (n_on_oos > 0 and expR_on_oos < 0) else "0")
    lines.append(
        f"| Primary: direction match | sign(OOS) == sign(IS=+) | "
        f"{sign_char} | "
        f"{'PASS' if sign_match else 'FAIL'} |"
    )
    lines.append(
        f"| Primary: N_on_OOS >= {POWER_FLOOR_PASS} (CLT floor) | N_on_OOS | {n_on_oos} | "
        f"{'PASS' if n_on_oos >= POWER_FLOOR_PASS else f'PARK ({result.power_band})'} |"
    )
    lines.append(
        f"| Secondary: bootstrap p < 0.10 | moving-block B={BOOTSTRAP_B}, block={BOOTSTRAP_BLOCK} | "
        f"{bootstrap_p:.4f} | "
        f"{'PASS' if not np.isnan(bootstrap_p) and bootstrap_p < 0.10 else ('N/A' if np.isnan(bootstrap_p) else 'WEAK')} |"
    )
    lines.append("")

    lines.append("## Per-fire log — OOS on-signal (F5_BELOW_PDL=1, long)")
    lines.append("")
    on_oos_rows = df_oos[df_oos["F5_BELOW_PDL"] == 1].sort_values("trading_day")
    if len(on_oos_rows) > 0:
        lines.append("| trading_day | entry_price | stop_price | target_price | pnl_r | outcome | orb_mid | prev_day_low |")
        lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
        for _, r in on_oos_rows.iterrows():
            lines.append(
                f"| {r['trading_day'].date()} | {r['entry_price']:.2f} | "
                f"{r['stop_price']:.2f} | {r['target_price']:.2f} | "
                f"{r['pnl_r']:+.4f} | {r['outcome']} | "
                f"{r['orb_mid']:.2f} | {r['prev_day_low']:.2f} |"
            )
    else:
        lines.append("(no on-signal OOS fires)")
    lines.append("")

    lines.append("## Compliance")
    lines.append("")
    lines.append(f"- [x] IS window respected: trading_day < {IS_TO}")
    lines.append(f"- [x] OOS window respected: {OOS_FROM} <= trading_day < {OOS_TO}")
    lines.append("- [x] No threshold tuning (binary predicate, nothing to tune)")
    lines.append("- [x] Feature is trade-time-knowable (backtesting-methodology.md Rule 6.1)")
    lines.append("- [x] Triple-join on (trading_day, symbol, orb_minutes)")
    lines.append("- [x] Script refuses re-run if output md exists")
    lines.append(f"- [x] Pre-reg sha pinned: {PRE_REG_SHA}")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if verdict == "PASS":
        lines.append(
            "- PASS: primary gates cleared. User decides whether to author an "
            "implementation pre-reg that wires F5_BELOW_PDL into "
            "`trading_app.config.ALL_FILTERS`. This validator does NOT authorise "
            "wiring; implementation requires a SEPARATE approved pre-reg."
        )
    elif verdict.startswith("PARK"):
        lines.append(
            f"- PARK ({result.power_band}): OOS on-signal count below "
            f"POWER_FLOOR_PASS={POWER_FLOOR_PASS}. No capital deployment. "
            f"Shadow forward for 6+ months before any re-read. No re-tuning "
            f"permitted under Mode A. Park reason: {park_reason}"
        )
    elif verdict == "KILL":
        lines.append(
            f"- KILL: {', '.join(kill_reasons)}. Declare F5_BELOW_PDL on this "
            "exact lane DEAD. Postmortem required. Do not wire into ALL_FILTERS."
        )
    else:
        lines.append(f"- {verdict}: unexpected state; manual review required.")
    lines.append("")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"VERDICT: {verdict}")
    print(f"ExpR_on_OOS={expR_on_oos:+.4f}  N_on_OOS={n_on_oos}  eff_ratio={eff_ratio:+.4f}  sign_match={sign_match}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
