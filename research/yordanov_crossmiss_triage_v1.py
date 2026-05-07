"""Yordanov 2026 § 3.8 Cross+Miss veto-signal triage on 3 deployed MNQ lanes.

Pre-registered: docs/audit/hypotheses/2026-05-07-mnq-yordanov-crossmiss-triage-v1.yaml
Commit lock: fd7c5073

This is a Pathway B K=1 individual-mechanism IS-only TRIAGE probe. It tests
whether the Cross+Miss bucket pattern Yordanov 2026 § 3.8 reported on NQ
(n=44, 22.7% hit-rate at 0.5x dev vs 71.1% baseline; 48.4pp gap) is even
present on MNQ before any further grounding work.

Locked numeric kill criteria (echoed from pre-reg, NOT recomputed):
- K1 FALSIFIED: pooled gap < 20pp
- K2 SURVIVES:  pooled gap >= 35pp AND >= 2/3 lanes show per-lane gap >= 25pp
- K3 AMBIGUOUS: 20-35pp pooled OR (>=35 BUT only 1/3 lanes confirms)
- K4 UNDERPOWERED: any bucket N<30 on any lane (precedence: fires first)

Canonical anchors (Rule: integrity-guardian.md § 2 — never re-encode):
- DB path           : pipeline.paths.GOLD_DB_PATH
- IS boundary       : trading_app.holdout_policy.HOLDOUT_SACRED_FROM
- Filter delegation : research.filter_utils.filter_signal
- Triple-join key   : (trading_day, symbol, orb_minutes)  per daily-features-joins.md

Look-ahead boundary: bars_1m.ts_utc > entry_ts (strict). entry_ts is bar-CLOSE
per trading_app/outcome_builder.py:277-282; first post-entry bar is therefore
strictly later. Asserted in self_review() on a sampled trade.

# scratch-policy: drop
# Rationale: confirm_bars=1 stop-market lanes; matches deployed-lane convention.
# Per pre-reg field scratch_policy.policy = "drop". Bailey-LdP 2014 caveat
# acknowledged in pre-reg.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# ---------------------------------------------------------------------------
# Lane scope — verbatim from docs/runtime/lane_allocation.json:7-65 (deployed
# MNQ lanes as of rebalance_date 2026-05-03). Mirrors pre-reg scope.lanes.
# ---------------------------------------------------------------------------

LANES: list[dict] = [
    {
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        "symbol": "MNQ",
        "orb_label": "COMEX_SETTLE",
        "orb_minutes": 5,
        "rr_target": 1.5,
        "filter_key": "OVNRNG_100",
    },
    {
        "strategy_id": "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
        "symbol": "MNQ",
        "orb_label": "US_DATA_1000",
        "orb_minutes": 15,
        "rr_target": 1.5,
        "filter_key": "VWAP_MID_ALIGNED",
    },
    {
        "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        "symbol": "MNQ",
        "orb_label": "NYSE_OPEN",
        "orb_minutes": 5,
        "rr_target": 1.0,
        "filter_key": "COST_LT12",
    },
]
# @canonical-source: docs/runtime/lane_allocation.json:7-65


PRE_REG_PATH = Path("docs/audit/hypotheses/2026-05-07-mnq-yordanov-crossmiss-triage-v1.yaml")
RESULT_MD_PATH = Path("docs/audit/results/2026-05-07-mnq-yordanov-crossmiss-triage-v1.md")
RESULT_CSV_PATH = Path("docs/audit/results/2026-05-07-mnq-yordanov-crossmiss-triage-v1.csv")
DECISION_LEDGER_PATH = Path("docs/runtime/decision-ledger.md")
LOCKED_COMMIT_SHA = "fd7c5073"

Bucket = Literal["NO_CROSS", "CROSS_HIT", "CROSS_MISS"]


# ---------------------------------------------------------------------------
# (1) load_lane_outcomes — canonical SQL with triple-join + filter columns.
# ---------------------------------------------------------------------------


def load_lane_outcomes(con: duckdb.DuckDBPyConnection, lane: dict) -> pd.DataFrame:
    """Load IS orb_outcomes JOIN daily_features (triple-join) for one lane.

    Returns one row per (trading_day, entry_ts) trade. Includes the full
    daily_features row so research.filter_utils.filter_signal can read whatever
    columns the canonical filter requires.

    # scratch-policy: drop  (WHERE pnl_r IS NOT NULL)
    """
    holdout = HOLDOUT_SACRED_FROM
    high_col = f"orb_{lane['orb_label']}_high"
    low_col = f"orb_{lane['orb_label']}_low"

    sql = f"""
        SELECT
            o.trading_day,
            o.entry_ts,
            o.exit_ts,
            o.entry_price,
            o.pnl_r,
            o.outcome,
            d.{high_col} AS orb_high,
            d.{low_col}  AS orb_low,
            d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol      = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol       = ?
          AND o.orb_label    = ?
          AND o.orb_minutes  = ?
          AND o.entry_model  = 'E2'
          AND o.confirm_bars = 1
          AND o.rr_target    = ?
          AND o.pnl_r IS NOT NULL
          AND o.trading_day  < ?
        ORDER BY o.entry_ts
    """
    df = con.execute(
        sql,
        [
            lane["symbol"],
            lane["orb_label"],
            lane["orb_minutes"],
            lane["rr_target"],
            holdout,
        ],
    ).fetchdf()

    # Drop duplicated 'trading_day' column injected by `d.*` (orb_outcomes has
    # one too). Keep o.trading_day (first occurrence).
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# (2) apply_canonical_filter — wraps research.filter_utils.filter_signal.
# ---------------------------------------------------------------------------


def apply_canonical_filter(df: pd.DataFrame, filter_key: str, orb_label: str) -> pd.DataFrame:
    """Apply canonical lane filter. Returns rows where filter fires (signal==1)."""
    if df.empty:
        return df
    sig = filter_signal(df, filter_key, orb_label)
    return df.loc[sig == 1].reset_index(drop=True).copy()


# ---------------------------------------------------------------------------
# (3) load_post_entry_bars — strict ts_utc > entry_ts.
# ---------------------------------------------------------------------------


def load_post_entry_bars(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    entry_ts_utc: pd.Timestamp,
    upper_bound_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Return all bars STRICTLY after entry_ts up to upper_bound_utc inclusive.

    upper_bound_utc per pre-reg bucket_definitions: exit_ts (the trade's actual
    resolution timestamp). Bounding by exit_ts measures structural
    follow-through during the trade's lifetime — NOT into next session — which
    is the trade-relevant analogue of Yordanov's session-bounded "Episode 1"
    measurement. Without this bound, post-stop intraday trends inflate
    hit-rates to ~99% (caught in self-review on first run; pre-reg locked
    correct bound, initial code used 23-hour day-end which was an
    implementation bug, fixed pre-emit).
    """
    sql = """
        SELECT ts_utc, open, high, low, close
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc > ?
          AND ts_utc <= ?
        ORDER BY ts_utc
    """
    return con.execute(sql, [symbol, entry_ts_utc, upper_bound_utc]).fetchdf()


# ---------------------------------------------------------------------------
# (4) classify_trade — first-event-wins bucket assignment.
# ---------------------------------------------------------------------------


def classify_trade(
    orb_high: float,
    orb_low: float,
    entry_price: float,
    post_entry_bars: pd.DataFrame,
) -> tuple[Bucket, float]:
    """Bucket assignment per pre-reg bucket_definitions.

    Returns (bucket, max_favourable_excursion_pts).

    Side: long if entry_price > orb_mid; short if entry_price < orb_mid.
    Mid: (orb_high + orb_low) / 2.
    Dev target: 0.5 * (orb_high - orb_low).
    Mid-cross detected on bar.close (Yordanov § 2.7 verbatim: "5-minute close
    back through the Filter Mid"). Dev target met on bar high/low (favourable
    excursion can be reached intra-bar).

    Bucket assignment is presence-based (order-agnostic), per Yordanov § 3.8:
      NO_CROSS   = no opposite-direction close-cross of orb_mid in window
      CROSS_HIT  = cross occurred AND favourable excursion reached 0.5x dev
      CROSS_MISS = cross occurred AND favourable excursion never reached 0.5x
    Returned `max_favourable_excursion_pts` is the true session max across
    the full post-entry window (not truncated to the bucket-assignment bar).
    """
    orb_mid = (orb_high + orb_low) / 2.0
    dev_target = 0.5 * (orb_high - orb_low)

    if entry_price > orb_mid:
        side = "long"
    elif entry_price < orb_mid:
        side = "short"
    else:
        # Edge case: entry exactly on mid. Pre-reg side_resolution: warned and
        # excluded (caller decides; we return a sentinel by raising).
        raise ValueError(f"entry_price == orb_mid ({entry_price}) — undecidable side")

    if post_entry_bars.empty:
        return "NO_CROSS", 0.0

    cross_seen = False
    max_excursion = 0.0
    for bar in post_entry_bars.itertuples(index=False):
        if side == "long":
            excursion = float(bar.high) - entry_price
            if not cross_seen and float(bar.close) < orb_mid:
                cross_seen = True
        else:
            excursion = entry_price - float(bar.low)
            if not cross_seen and float(bar.close) > orb_mid:
                cross_seen = True
        if excursion > max_excursion:
            max_excursion = excursion

    dev_hit = max_excursion >= dev_target
    if not cross_seen:
        return "NO_CROSS", max_excursion
    if dev_hit:
        return "CROSS_HIT", max_excursion
    return "CROSS_MISS", max_excursion


# ---------------------------------------------------------------------------
# (5) compute_lane_buckets — orchestrates per-trade classification.
# ---------------------------------------------------------------------------


def compute_lane_buckets(con: duckdb.DuckDBPyConnection, lane: dict) -> tuple[pd.DataFrame, dict]:
    """For one lane: load filtered IS trades, classify each, return frame + diags."""
    raw = load_lane_outcomes(con, lane)
    n_unfiltered = len(raw)
    # exit_ts is in raw (orb_outcomes column) — preserve through filter
    filtered = apply_canonical_filter(raw, lane["filter_key"], lane["orb_label"])
    n_filtered = len(filtered)

    classified: list[dict] = []
    excluded_mid_eq = 0
    no_post_bars = 0
    no_exit_ts = 0

    # Per pre-reg bucket_definitions: post-entry window bounded by exit_ts
    # (the trade's resolution). This matches Yordanov's "Episode 1" framing
    # adapted to a stop/target trade — measures follow-through DURING the
    # trade, not into next session.
    for row in filtered.itertuples(index=False):
        entry_ts = pd.Timestamp(row.entry_ts)
        if entry_ts.tz is None:
            entry_ts = entry_ts.tz_localize("UTC")

        exit_ts_raw = getattr(row, "exit_ts", None)
        if exit_ts_raw is None or pd.isna(exit_ts_raw):
            no_exit_ts += 1
            continue
        exit_ts = pd.Timestamp(exit_ts_raw)
        if exit_ts.tz is None:
            exit_ts = exit_ts.tz_localize("UTC")
        # Must be strictly after entry_ts; if exit_ts <= entry_ts treat as
        # data anomaly and skip (extremely rare; would mean instant fill+exit).
        if exit_ts <= entry_ts:
            no_exit_ts += 1
            continue

        bars = load_post_entry_bars(con, lane["symbol"], entry_ts, exit_ts)
        if bars.empty:
            no_post_bars += 1

        try:
            bucket, excursion = classify_trade(row.orb_high, row.orb_low, float(row.entry_price), bars)
        except ValueError:
            excluded_mid_eq += 1
            continue

        classified.append(
            {
                "strategy_id": lane["strategy_id"],
                "trading_day": pd.Timestamp(row.trading_day).date(),
                "entry_ts": entry_ts,
                "entry_price": float(row.entry_price),
                "orb_high": float(row.orb_high),
                "orb_low": float(row.orb_low),
                "pnl_r": float(row.pnl_r),
                "bucket": bucket,
                "favourable_excursion_pts": excursion,
                "dev_target_pts": 0.5 * (float(row.orb_high) - float(row.orb_low)),
            }
        )

    df = pd.DataFrame(classified)
    diags = {
        "n_unfiltered_IS": n_unfiltered,
        "n_filtered_IS": n_filtered,
        "n_excluded_mid_eq": excluded_mid_eq,
        "n_no_post_bars": no_post_bars,
        "n_no_exit_ts": no_exit_ts,
        "n_classified": len(df),
    }
    return df, diags


# ---------------------------------------------------------------------------
# (6) summarize_lane — bucket counts, hit-rates, gap, lane verdict tier.
# ---------------------------------------------------------------------------


def summarize_lane(df: pd.DataFrame, lane: dict, diags: dict) -> dict:
    """Per-lane summary aligned with pooled-finding-rule per-lane table cols."""
    if df.empty:
        return {
            "strategy_id": lane["strategy_id"],
            "N_total": 0,
            "N_NO_CROSS": 0,
            "hit_rate_NO_CROSS": float("nan"),
            "expr_NO_CROSS": float("nan"),
            "N_CROSS_HIT": 0,
            "hit_rate_CROSS_HIT_05x": float("nan"),
            "expr_CROSS_HIT": float("nan"),
            "N_CROSS_MISS": 0,
            "hit_rate_CROSS_MISS_05x": float("nan"),
            "expr_CROSS_MISS": float("nan"),
            "gap_pp": float("nan"),
            "tier": "NO_DATA",
            "diags": diags,
        }

    # Hit at 0.5x dev: favourable_excursion_pts >= dev_target_pts
    df = df.copy()
    df["hit_05x"] = (df["favourable_excursion_pts"] >= df["dev_target_pts"]).astype(int)

    def bucket_stats(b: str) -> tuple[int, float, float]:
        sub = df[df["bucket"] == b]
        n = len(sub)
        if n == 0:
            return 0, float("nan"), float("nan")
        return n, float(sub["hit_05x"].mean()), float(sub["pnl_r"].mean())

    n_nc, hr_nc, expr_nc = bucket_stats("NO_CROSS")
    n_ch, hr_ch, expr_ch = bucket_stats("CROSS_HIT")
    n_cm, hr_cm, expr_cm = bucket_stats("CROSS_MISS")

    # Gap in percentage points
    if math.isnan(hr_nc) or math.isnan(hr_cm):
        gap_pp = float("nan")
    else:
        gap_pp = (hr_nc - hr_cm) * 100.0

    # Per-lane tier (informational; pooled verdict is what fires kill criteria)
    if any(n < 30 for n in (n_nc, n_ch, n_cm)):
        tier = "UNDERPOWERED"
    elif math.isnan(gap_pp):
        tier = "NO_DATA"
    elif gap_pp < 20:
        tier = "FALSIFIED"
    elif gap_pp >= 35:
        tier = "SURVIVES"
    else:
        tier = "AMBIGUOUS"

    return {
        "strategy_id": lane["strategy_id"],
        "N_total": len(df),
        "N_NO_CROSS": n_nc,
        "hit_rate_NO_CROSS": hr_nc,
        "expr_NO_CROSS": expr_nc,
        "N_CROSS_HIT": n_ch,
        "hit_rate_CROSS_HIT_05x": hr_ch,
        "expr_CROSS_HIT": expr_ch,
        "N_CROSS_MISS": n_cm,
        "hit_rate_CROSS_MISS_05x": hr_cm,
        "expr_CROSS_MISS": expr_cm,
        "gap_pp": gap_pp,
        "tier": tier,
        "diags": diags,
    }


# ---------------------------------------------------------------------------
# (7) pool_summary — pooled hit-rates, gap, flip rate, two-prop z-test.
# ---------------------------------------------------------------------------


def pool_summary(per_lane: list[dict]) -> dict:
    """Pool buckets across lanes; compute flip_rate_pct and z-test on NC vs CM.

    flip_rate_pct: share of lanes whose per-lane gap sign opposes the pooled
    gap sign. UNDERPOWERED lanes (N<30 in any bucket) excluded from the
    flip-rate denominator.
    """
    n_nc = sum(d["N_NO_CROSS"] for d in per_lane)
    n_ch = sum(d["N_CROSS_HIT"] for d in per_lane)
    n_cm = sum(d["N_CROSS_MISS"] for d in per_lane)
    n_total = n_nc + n_ch + n_cm

    # Pooled hit-rates (weighted by N within each bucket)
    def pooled_hit_rate(bucket_key: str, n_key: str) -> float:
        num = 0.0
        den = 0
        for d in per_lane:
            n = d[n_key]
            hr = d[bucket_key]
            if n > 0 and not math.isnan(hr):
                num += hr * n
                den += n
        return float("nan") if den == 0 else num / den

    pooled_hr_nc = pooled_hit_rate("hit_rate_NO_CROSS", "N_NO_CROSS")
    pooled_hr_ch = pooled_hit_rate("hit_rate_CROSS_HIT_05x", "N_CROSS_HIT")
    pooled_hr_cm = pooled_hit_rate("hit_rate_CROSS_MISS_05x", "N_CROSS_MISS")

    if math.isnan(pooled_hr_nc) or math.isnan(pooled_hr_cm):
        pooled_gap_pp = float("nan")
    else:
        pooled_gap_pp = (pooled_hr_nc - pooled_hr_cm) * 100.0

    # Two-proportion z-test: NO_CROSS hits vs CROSS_MISS hits.
    if not math.isnan(pooled_hr_nc) and not math.isnan(pooled_hr_cm) and n_nc > 0 and n_cm > 0:
        x_nc = pooled_hr_nc * n_nc
        x_cm = pooled_hr_cm * n_cm
        p_pool = (x_nc + x_cm) / (n_nc + n_cm)
        se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n_nc + 1.0 / n_cm))
        if se == 0:
            z, p_two = float("nan"), float("nan")
        else:
            z = (pooled_hr_nc - pooled_hr_cm) / se
            # Two-sided p via standard normal survival
            p_two = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    else:
        z, p_two = float("nan"), float("nan")

    # Flip rate: per-lane gap sign vs pooled gap sign
    if math.isnan(pooled_gap_pp):
        flip_rate_pct = float("nan")
    else:
        pooled_sign = 1 if pooled_gap_pp >= 0 else -1
        eligible = [d for d in per_lane if not math.isnan(d["gap_pp"]) and d["tier"] != "UNDERPOWERED"]
        if not eligible:
            flip_rate_pct = float("nan")
        else:
            flips = sum(1 for d in eligible if (1 if d["gap_pp"] >= 0 else -1) != pooled_sign)
            flip_rate_pct = 100.0 * flips / len(eligible)

    # Pooled verdict per locked kill criteria (precedence per pre-reg)
    if any(d["tier"] == "UNDERPOWERED" for d in per_lane) or math.isnan(pooled_gap_pp):
        verdict = "UNDERPOWERED"
    elif pooled_gap_pp < 20:
        verdict = "FALSIFIED"
    elif pooled_gap_pp >= 35:
        per_lane_passes = sum(
            1 for d in per_lane if not math.isnan(d["gap_pp"]) and d["gap_pp"] >= 25 and d["tier"] != "UNDERPOWERED"
        )
        verdict = "SURVIVES" if per_lane_passes >= 2 else "AMBIGUOUS"
    else:
        verdict = "AMBIGUOUS"

    return {
        "N_total": n_total,
        "N_NO_CROSS": n_nc,
        "N_CROSS_HIT": n_ch,
        "N_CROSS_MISS": n_cm,
        "pooled_hit_rate_NO_CROSS": pooled_hr_nc,
        "pooled_hit_rate_CROSS_HIT_05x": pooled_hr_ch,
        "pooled_hit_rate_CROSS_MISS_05x": pooled_hr_cm,
        "pooled_gap_pp": pooled_gap_pp,
        "z_two_prop": z,
        "p_two_sided": p_two,
        "flip_rate_pct": flip_rate_pct,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# (8) write_outputs — md result + companion CSV + decision-ledger append.
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x * 100:.1f}%"


def _fmt_pp(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.1f}pp"


def _fmt_num(x: float, fmt: str = "{:.4f}") -> str:
    return "n/a" if math.isnan(x) else fmt.format(x)


def write_outputs(
    per_lane: list[dict],
    pooled: dict,
    csv_path: Path,
    md_path: Path,
) -> None:
    """Emit per-lane CSV + result md with mandatory pooled-finding front-matter."""
    # CSV — one row per lane + one pooled row
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "scope",
                "lane",
                "N_total",
                "N_NO_CROSS",
                "hit_rate_NO_CROSS",
                "expr_NO_CROSS",
                "N_CROSS_HIT",
                "hit_rate_CROSS_HIT_05x",
                "expr_CROSS_HIT",
                "N_CROSS_MISS",
                "hit_rate_CROSS_MISS_05x",
                "expr_CROSS_MISS",
                "gap_pp",
                "tier",
            ]
        )
        for d in per_lane:
            w.writerow(
                [
                    "per_lane",
                    d["strategy_id"],
                    d["N_total"],
                    d["N_NO_CROSS"],
                    d["hit_rate_NO_CROSS"],
                    d["expr_NO_CROSS"],
                    d["N_CROSS_HIT"],
                    d["hit_rate_CROSS_HIT_05x"],
                    d["expr_CROSS_HIT"],
                    d["N_CROSS_MISS"],
                    d["hit_rate_CROSS_MISS_05x"],
                    d["expr_CROSS_MISS"],
                    d["gap_pp"],
                    d["tier"],
                ]
            )
        w.writerow(
            [
                "pooled",
                "ALL_3_LANES",
                pooled["N_total"],
                pooled["N_NO_CROSS"],
                pooled["pooled_hit_rate_NO_CROSS"],
                "",
                pooled["N_CROSS_HIT"],
                pooled["pooled_hit_rate_CROSS_HIT_05x"],
                "",
                pooled["N_CROSS_MISS"],
                pooled["pooled_hit_rate_CROSS_MISS_05x"],
                "",
                pooled["pooled_gap_pp"],
                pooled["verdict"],
            ]
        )

    # Per-lane table rows
    rows = []
    for d in per_lane:
        rows.append(
            f"| {d['strategy_id']} | {d['N_total']} | "
            f"{d['N_NO_CROSS']} | {_fmt_pct(d['hit_rate_NO_CROSS'])} | "
            f"{d['N_CROSS_HIT']} | {_fmt_pct(d['hit_rate_CROSS_HIT_05x'])} | "
            f"{d['N_CROSS_MISS']} | {_fmt_pct(d['hit_rate_CROSS_MISS_05x'])} | "
            f"{_fmt_pp(d['gap_pp'])} | {d['tier']} |"
        )
    per_lane_table = "\n".join(rows)

    pooled_row = (
        f"| **POOLED (3 lanes)** | {pooled['N_total']} | "
        f"{pooled['N_NO_CROSS']} | {_fmt_pct(pooled['pooled_hit_rate_NO_CROSS'])} | "
        f"{pooled['N_CROSS_HIT']} | {_fmt_pct(pooled['pooled_hit_rate_CROSS_HIT_05x'])} | "
        f"{pooled['N_CROSS_MISS']} | {_fmt_pct(pooled['pooled_hit_rate_CROSS_MISS_05x'])} | "
        f"{_fmt_pp(pooled['pooled_gap_pp'])} | {pooled['verdict']} |"
    )

    flip_rate = pooled["flip_rate_pct"]
    heterogeneity_line = "heterogeneity_ack: true\n" if not math.isnan(flip_rate) and flip_rate >= 25 else ""
    flip_rate_str = "n/a" if math.isnan(flip_rate) else f"{flip_rate:.1f}"

    # Echo locked kill criteria verbatim from pre-reg (no recomputation)
    locked_criteria = """\
- **K1 FALSIFIED**: pooled gap_pp < 20
- **K2 SURVIVES**: pooled gap_pp >= 35 **AND** >= 2 of 3 lanes show per-lane gap >= 25
- **K3 AMBIGUOUS**: pooled gap in [20, 35) OR (pooled >=35 BUT only 1/3 lanes confirms)
- **K4 UNDERPOWERED** (precedence: fires first): any of {N_NO_CROSS, N_CROSS_HIT, N_CROSS_MISS} < 30 on any lane
"""

    md = (
        f"""---
pooled_finding: true
per_cell_breakdown_path: {csv_path.as_posix()}
flip_rate_pct: {flip_rate_str}
{heterogeneity_line}---

# Yordanov 2026 § 3.8 Cross+Miss veto-signal triage on 3 deployed MNQ lanes

**Pre-reg:** `docs/audit/hypotheses/2026-05-07-mnq-yordanov-crossmiss-triage-v1.yaml`
**Pre-reg commit lock:** `{LOCKED_COMMIT_SHA}`
**Companion CSV:** `{csv_path.as_posix()}`
**Canonical DB:** `{GOLD_DB_PATH}`

## Scope

Pathway B K=1 individual-mechanism IS-only triage. Three deployed MNQ lanes (rebalance 2026-05-03) scored independently then pooled. IS window: `trading_day < {HOLDOUT_SACRED_FROM}` from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`. Filter delegation via `research.filter_utils.filter_signal`.

## Locked kill criteria (echoed verbatim from pre-reg, NOT recomputed)

{locked_criteria}

## Verdict

**{pooled["verdict"]}**

Pooled gap (NO_CROSS − CROSS_MISS hit-rate at 0.5×) = **{_fmt_pp(pooled["pooled_gap_pp"])}**.
Two-proportion z = {_fmt_num(pooled["z_two_prop"], "{:.3f}")}, two-sided p = {_fmt_num(pooled["p_two_sided"], "{:.4f}")}.
Per-lane gap-sign flip rate = **{flip_rate_str}%**.

Per-lane confirmations (gap_pp >= 25 AND not UNDERPOWERED): {sum(1 for d in per_lane if not math.isnan(d["gap_pp"]) and d["gap_pp"] >= 25 and d["tier"] != "UNDERPOWERED")} of 3.

## Per-lane summary

| lane | N_total | N_NO_CROSS | hit_NO_CROSS | N_CROSS_HIT | hit_CROSS_HIT_05x | N_CROSS_MISS | hit_CROSS_MISS_05x | gap_pp | tier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{per_lane_table}
{pooled_row}

## Diagnostics (per lane)

| lane | N_unfiltered_IS | N_filtered_IS | N_classified | N_excluded_mid_eq | N_no_post_bars | N_no_exit_ts |
|---|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(
            f"| {d['strategy_id']} | {d['diags']['n_unfiltered_IS']} | {d['diags']['n_filtered_IS']} | {d['diags']['n_classified']} | {d['diags']['n_excluded_mid_eq']} | {d['diags']['n_no_post_bars']} | {d['diags'].get('n_no_exit_ts', 0)} |"
            for d in per_lane
        )
        + f"""

## Method notes

- Canonical layers only: `orb_outcomes` JOIN `daily_features` on `(trading_day, symbol, orb_minutes)` (triple-join per `daily-features-joins.md`).
- Sacred holdout boundary: `trading_day < HOLDOUT_SACRED_FROM` ({HOLDOUT_SACRED_FROM}); OOS not consumed.
- Canonical filter delegation: `research.filter_utils.filter_signal(df, key, orb_label)` for OVNRNG_100, VWAP_MID_ALIGNED, COST_LT12.
- Look-ahead boundary: `bars_1m.ts_utc > entry_ts` (strict). `entry_ts` is bar-CLOSE per `trading_app/outcome_builder.py:277-282`.
- Bucket assignment: first-event-wins on (favourable_excursion >= 0.5×(orb_high−orb_low), bar.close re-cross of orb_mid). Side from entry_price vs orb_mid.
- Hit metric: P(post-entry favourable excursion >= 0.5× deviation target before exit_ts).
- scratch-policy: drop (`WHERE pnl_r IS NOT NULL`); matches deployed-lane convention.
- No writes to `validated_setups`, `experimental_strategies`, `lane_allocation.json`, `paper_trades`.

## Reproduction

```
python research/yordanov_crossmiss_triage_v1.py --self-review
```

## Caveats

- **Single-mechanism K=1 IS-only triage.** No OOS, no per-instrument generalization, no deployment claim. SURVIVES verdict triggers a separate confirmatory Pathway B K=1 pre-reg next session, not promotion.
- **POST-TRADE retrospective classifier.** The bucket label is computed from post-entry bars; this is NOT a trade-time-knowable feature. Confirmatory pre-reg must define a live-tracked mid-cross detector before any deployment evaluation.
- **Filter Range analogue.** Yordanov defines Filter Range from Volume Profile Value Area (Case A) or 10:00 candle (Case B/C); this triage uses ORB high/low as the project-canonical analogue. The mid-cross mechanism is what is being tested; faithful VA replication is queued as Stage D follow-up if SURVIVES.
- **OOS power-floor not invoked** (IS-only); confirmatory pre-reg next session would invoke per `backtesting-methodology.md` § 3.3.
- **Pooled-finding annotation:** per-cell breakdown above; `flip_rate_pct` reported in front-matter; `heterogeneity_ack` set if >=25% per pooled-finding-rule.md.

## Ledger entry (appended to `{DECISION_LEDGER_PATH.as_posix()}`)

`yordanov-crossmiss-triage-v1-2026-05-07` — {pooled["verdict"]}: pooled NO_CROSS−CROSS_MISS gap = {_fmt_pp(pooled["pooled_gap_pp"])}; per-lane gaps = {" / ".join(_fmt_pp(d["gap_pp"]) for d in per_lane)} on (COMEX_SETTLE / US_DATA_1000 / NYSE_OPEN). Locked kill criteria pre-committed in `2026-05-07-mnq-yordanov-crossmiss-triage-v1.yaml` (commit {LOCKED_COMMIT_SHA}).
"""
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")


def append_decision_ledger(per_lane: list[dict], pooled: dict) -> None:
    """One-line append to docs/runtime/decision-ledger.md."""
    next_action = {
        "FALSIFIED": "annotate_yordanov_extract_and_mechanism_priors",
        "SURVIVES": "confirmatory_prereg_next_session",
        "AMBIGUOUS": "park_no_promotion",
        "UNDERPOWERED": "underpowered_park",
    }[pooled["verdict"]]
    line = (
        f"- `yordanov-crossmiss-triage-v1-2026-05-07` — {pooled['verdict']}: "
        f"pooled NO_CROSS−CROSS_MISS gap = {_fmt_pp(pooled['pooled_gap_pp'])}; "
        f"per-lane gaps = "
        f"{' / '.join(_fmt_pp(d['gap_pp']) for d in per_lane)} on "
        f"(COMEX_SETTLE/US_DATA_1000/NYSE_OPEN). "
        f"Locked kill criteria pre-committed in "
        f"2026-05-07-mnq-yordanov-crossmiss-triage-v1.yaml "
        f"(commit {LOCKED_COMMIT_SHA}). Next: {next_action}.\n"
    )
    with DECISION_LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# (9) self_review — drift check + result-md sanity assertions.
# ---------------------------------------------------------------------------


def self_review(con: duckdb.DuckDBPyConnection) -> int:
    """Post-run integrity gate. Returns nonzero exit code on any failure."""
    fails: list[str] = []

    # 1. Result md exists, has front-matter
    if not RESULT_MD_PATH.exists():
        fails.append(f"missing result md: {RESULT_MD_PATH}")
    else:
        text = RESULT_MD_PATH.read_text(encoding="utf-8")
        for key in ("pooled_finding: true", "per_cell_breakdown_path:", "flip_rate_pct:"):
            if key not in text:
                fails.append(f"result md missing front-matter key: {key}")
        if "## Per-lane summary" not in text:
            fails.append("result md missing per-lane summary section")

    # 2. Companion CSV exists
    if not RESULT_CSV_PATH.exists():
        fails.append(f"missing companion CSV: {RESULT_CSV_PATH}")

    # 3. Decision-ledger has the new line
    if DECISION_LEDGER_PATH.exists():
        ledger = DECISION_LEDGER_PATH.read_text(encoding="utf-8")
        if "yordanov-crossmiss-triage-v1-2026-05-07" not in ledger:
            fails.append("decision-ledger missing triage entry")

    # 4. Look-ahead boundary spot check: pull one filtered trade per lane and
    #    confirm first post-entry bar's ts_utc is STRICTLY > entry_ts. Use
    #    exit_ts as upper bound per pre-reg.
    for lane in LANES:
        raw = load_lane_outcomes(con, lane)
        filtered = apply_canonical_filter(raw, lane["filter_key"], lane["orb_label"])
        if filtered.empty:
            continue
        sample = filtered.iloc[0]
        entry_ts = pd.Timestamp(sample["entry_ts"])
        if entry_ts.tz is None:
            entry_ts = entry_ts.tz_localize("UTC")
        exit_ts_raw = sample.get("exit_ts")
        if exit_ts_raw is None or pd.isna(exit_ts_raw):
            continue
        exit_ts = pd.Timestamp(exit_ts_raw)
        if exit_ts.tz is None:
            exit_ts = exit_ts.tz_localize("UTC")
        if exit_ts <= entry_ts:
            continue
        bars = load_post_entry_bars(con, lane["symbol"], entry_ts, exit_ts)
        if not bars.empty:
            first_bar_ts = pd.Timestamp(bars.iloc[0]["ts_utc"])
            if first_bar_ts.tz is None:
                first_bar_ts = first_bar_ts.tz_localize("UTC")
            if not (first_bar_ts > entry_ts):
                fails.append(
                    f"look-ahead violation on {lane['strategy_id']}: "
                    f"first post-entry bar ts {first_bar_ts} not > entry_ts {entry_ts}"
                )

    # 5. Drift check
    print("[self_review] running pipeline/check_drift.py …", flush=True)
    proc = subprocess.run(
        [sys.executable, "pipeline/check_drift.py"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        fails.append(f"check_drift.py failed: rc={proc.returncode}")
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], file=sys.stderr, flush=True)

    if fails:
        print("\n[self_review] FAILED:", flush=True)
        for f in fails:
            print(f"  - {f}", flush=True)
        return 1
    print("\n[self_review] ALL CHECKS PASSED", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-review",
        action="store_true",
        help="Run drift check + result-md assertions after the probe.",
    )
    parser.add_argument(
        "--skip-drift",
        action="store_true",
        help="(self-review only) skip the check_drift.py subprocess.",
    )
    args = parser.parse_args()

    print(f"[yordanov-crossmiss-triage-v1] DB: {GOLD_DB_PATH}", flush=True)
    print(f"[yordanov-crossmiss-triage-v1] IS boundary: {HOLDOUT_SACRED_FROM}", flush=True)
    print(f"[yordanov-crossmiss-triage-v1] Pre-reg commit: {LOCKED_COMMIT_SHA}", flush=True)
    print(f"[yordanov-crossmiss-triage-v1] {len(LANES)} lanes", flush=True)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        per_lane: list[dict] = []
        for lane in LANES:
            print(f"[lane] {lane['strategy_id']} …", flush=True)
            df, diags = compute_lane_buckets(con, lane)
            summary = summarize_lane(df, lane, diags)
            per_lane.append(summary)
            print(
                f"   N_total={summary['N_total']} "
                f"buckets NC/CH/CM={summary['N_NO_CROSS']}/{summary['N_CROSS_HIT']}/{summary['N_CROSS_MISS']} "
                f"gap_pp={summary['gap_pp']} tier={summary['tier']}",
                flush=True,
            )

        pooled = pool_summary(per_lane)
        print(
            f"[pooled] N_total={pooled['N_total']} "
            f"gap_pp={pooled['pooled_gap_pp']:.2f} z={pooled['z_two_prop']:.3f} "
            f"p={pooled['p_two_sided']:.4f} flip%={pooled['flip_rate_pct']} "
            f"verdict={pooled['verdict']}",
            flush=True,
        )

        write_outputs(per_lane, pooled, RESULT_CSV_PATH, RESULT_MD_PATH)
        append_decision_ledger(per_lane, pooled)
        print(f"[written] {RESULT_MD_PATH}", flush=True)
        print(f"[written] {RESULT_CSV_PATH}", flush=True)
        print(f"[appended] {DECISION_LEDGER_PATH}", flush=True)

        if args.self_review:
            if args.skip_drift:
                print("[self_review] --skip-drift set; skipping check_drift.py", flush=True)
                return 0
            return self_review(con)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
