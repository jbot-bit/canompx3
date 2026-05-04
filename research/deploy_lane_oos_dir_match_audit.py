"""Per-lane 2026 OOS dir_match audit — mandate #4 (2026-04-20).

For each of the 6 DEPLOY lanes in `docs/runtime/lane_allocation.json`:
  1. Compute IS statistics (trading_day < HOLDOUT_SACRED_FROM).
  2. Compute 2026 OOS statistics.
  3. Check dir_match (sign(ExpR_IS) == sign(ExpR_OOS)).
  4. Compute OOS power to detect IS effect at observed OOS_N (one-sample t).
  5. Classify per RULE 3.3 tier via canonical `oos_power.power_verdict`.

Canonical delegation (no re-encoded logic):
  - `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` — Mode A cutoff
  - `research.filter_utils.filter_signal` → `trading_app.config.ALL_FILTERS` — filter matching
  - `research.oos_power.power_verdict`, `POWER_TIERS` — tier thresholds

Note on one-sample vs two-sample framing:
  The canonical `oos_power.oos_ttest_power` is a two-sample Welch test designed
  for within-sample group comparisons (e.g., bear-day vs bull-day shorts). A
  per-lane dir_match is a one-sample question ("does OOS mean differ from
  zero when we expect it to equal ExpR_IS?") so we compute one-sample power
  inline via scipy. The tier thresholds (80% / 50%) and verdict names come
  from canonical `oos_power.POWER_TIERS` to stay RULE 3.3 compliant.

@research-source: mandate #4 in memory/next_session_mandates_2026_04_20.md
@data-source: orb_outcomes JOIN daily_features (triple-join canonical)
@revalidated-for: 2026-04-20 (Mode A IS; 2026 OOS)

Run:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/deploy_lane_oos_dir_match_audit.py
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.filter_utils import filter_signal  # noqa: E402
from research.oos_power import power_verdict  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

ALLOCATION_JSON = _ROOT / "docs" / "runtime" / "lane_allocation.json"
RESULT_DOC = _ROOT / "docs" / "audit" / "results" / "2026-04-20-deploy-lane-oos-dir-match-audit.md"
ALPHA = 0.05


@dataclass
class LaneSpec:
    strategy_id: str
    instrument: str
    orb_label: str
    rr_target: float
    filter_type: str
    orb_minutes: int
    entry_model: str = "E2"
    confirm_bars: int = 1

    @classmethod
    def from_allocation_row(cls, row: dict) -> "LaneSpec":
        sid = row["strategy_id"]
        if "_O15" in sid:
            orb_minutes = 15
        elif "_O30" in sid:
            orb_minutes = 30
        else:
            orb_minutes = 5
        return cls(
            strategy_id=sid,
            instrument=row["instrument"],
            orb_label=row["orb_label"],
            rr_target=float(row["rr_target"]),
            filter_type=row["filter_type"],
            orb_minutes=orb_minutes,
        )


@dataclass
class LaneAudit:
    strategy_id: str
    is_n: int
    is_mean: float
    is_std: float
    is_wr: float
    oos_n: int
    oos_mean: float
    oos_std: float
    oos_wr: float
    cohen_d: float
    oos_power: float
    oos_power_tier: str
    n_for_80pct_power: int
    dir_match: bool | None
    verdict: str
    oos_t: float
    oos_p_onesided: float
    oos_date_min: str
    oos_date_max: str


def load_lane_trades(con: duckdb.DuckDBPyConnection, lane: LaneSpec) -> pd.DataFrame:
    """Triple-join orb_outcomes + daily_features, apply canonical filter delegation."""
    q = f"""
    SELECT o.trading_day, o.symbol, o.orb_label, o.orb_minutes,
           o.entry_price, o.stop_price, o.pnl_r,
           d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{lane.instrument}'
      AND o.orb_label = '{lane.orb_label}'
      AND o.orb_minutes = {lane.orb_minutes}
      AND o.entry_model = '{lane.entry_model}'
      AND o.confirm_bars = {lane.confirm_bars}
      AND o.rr_target = {lane.rr_target}
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(q).fetchdf()
    df = df.loc[:, ~df.columns.duplicated()]
    # Canonical filter delegation
    fire = filter_signal(df, lane.filter_type, lane.orb_label)
    df = df.loc[fire == 1].copy()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    return df


def one_sample_power(d: float, n: int, alpha: float = ALPHA) -> float:
    """Two-sided one-sample t power at effect size d and sample N.

    H0: mean = 0; H1: mean = d * std. ncp = d * sqrt(n), df = n-1.
    Used because per-lane dir_match is a one-sample question
    ("does OOS mean differ from zero?"). Canonical
    `research.oos_power.oos_ttest_power` covers the two-sample case.
    """
    if d <= 0 or n < 2:
        return 0.0
    df = n - 1
    ncp = d * np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power_nct = 1.0 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    if np.isnan(power_nct):
        # Normal approximation at large ncp
        z_crit = stats.norm.ppf(1 - alpha / 2)
        return float(stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp))
    return float(power_nct)


def one_sample_n_for_power(d: float, target: float = 0.80, alpha: float = ALPHA) -> int:
    """Smallest N achieving `target` one-sample power at effect size d."""
    if d <= 0:
        return 10_000_000
    lo, hi = 2, 10_000_000
    if one_sample_power(d, hi, alpha) < target:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if one_sample_power(d, mid, alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def one_sample_tstat(mean: float, std: float, n: int) -> tuple[float, float]:
    """One-sample t-statistic + one-sided p (H0: mean=0, H1: mean > 0 if IS_mean > 0)."""
    if n < 2 or std <= 0:
        return float("nan"), float("nan")
    se = std / np.sqrt(n)
    t = mean / se
    df = n - 1
    # One-sided p in the direction of IS mean. Caller decides tail.
    p_one = float(stats.t.sf(abs(t), df))
    return float(t), p_one


def classify_lane(
    dir_match: bool | None, oos_power_tier: str
) -> str:
    """Apply pre-committed decision rule (stage file)."""
    if dir_match is None:
        return "UNVERIFIED_NO_OOS"
    # Matrix per stage file:
    if dir_match and oos_power_tier == "CAN_REFUTE":
        return "ALIVE_CONFIRMED"
    if dir_match and oos_power_tier == "DIRECTIONAL_ONLY":
        return "ALIVE_PROVISIONAL"
    if dir_match and oos_power_tier == "STATISTICALLY_USELESS":
        return "UNVERIFIED_ALIVE"
    if not dir_match and oos_power_tier == "CAN_REFUTE":
        return "DEAD"
    if not dir_match and oos_power_tier == "DIRECTIONAL_ONLY":
        return "WATCH"
    return "UNVERIFIED"  # not dir_match + STATISTICALLY_USELESS


def audit_lane(con: duckdb.DuckDBPyConnection, lane: LaneSpec) -> LaneAudit:
    df = load_lane_trades(con, lane)
    cutoff = HOLDOUT_SACRED_FROM
    is_df = df[df["trading_day"] < cutoff]
    oos_df = df[df["trading_day"] >= cutoff]

    is_pnl = is_df["pnl_r"].to_numpy(dtype=float)
    oos_pnl = oos_df["pnl_r"].to_numpy(dtype=float)

    is_n = int(is_pnl.size)
    oos_n = int(oos_pnl.size)
    is_mean = float(is_pnl.mean()) if is_n else float("nan")
    is_std = float(is_pnl.std(ddof=1)) if is_n >= 2 else float("nan")
    is_wr = float((is_pnl > 0).mean()) if is_n else float("nan")
    oos_mean = float(oos_pnl.mean()) if oos_n else float("nan")
    oos_std = float(oos_pnl.std(ddof=1)) if oos_n >= 2 else float("nan")
    oos_wr = float((oos_pnl > 0).mean()) if oos_n else float("nan")

    # Cohen's d for the IS effect — canonical one-sample effect size
    if is_n >= 2 and is_std > 0:
        cohen_d = abs(is_mean) / is_std
    else:
        cohen_d = 0.0

    pwr = one_sample_power(cohen_d, oos_n) if oos_n >= 2 else 0.0
    tier = power_verdict(pwr)
    n80 = one_sample_n_for_power(cohen_d)

    # dir_match only defined if both sides have data and nonzero means
    if oos_n == 0 or np.isnan(is_mean) or np.isnan(oos_mean):
        dir_match: bool | None = None
    else:
        dir_match = bool(np.sign(is_mean) == np.sign(oos_mean)) and (is_mean != 0)

    verdict = classify_lane(dir_match, tier)

    t_oos, p_oos = one_sample_tstat(oos_mean, oos_std, oos_n)

    oos_date_min = oos_df["trading_day"].min().isoformat() if oos_n else "-"
    oos_date_max = oos_df["trading_day"].max().isoformat() if oos_n else "-"

    return LaneAudit(
        strategy_id=lane.strategy_id,
        is_n=is_n,
        is_mean=is_mean,
        is_std=is_std,
        is_wr=is_wr,
        oos_n=oos_n,
        oos_mean=oos_mean,
        oos_std=oos_std,
        oos_wr=oos_wr,
        cohen_d=cohen_d,
        oos_power=pwr,
        oos_power_tier=tier,
        n_for_80pct_power=n80,
        dir_match=dir_match,
        verdict=verdict,
        oos_t=t_oos,
        oos_p_onesided=p_oos,
        oos_date_min=oos_date_min,
        oos_date_max=oos_date_max,
    )


def render_result_doc(audits: list[LaneAudit], lanes: list[LaneSpec]) -> str:
    lines: list[str] = []
    lines.append("# 2026 OOS per-lane dir_match audit — 6 DEPLOY lanes")
    lines.append("")
    lines.append(f"**Generated:** {date.today().isoformat()}")
    lines.append(f"**Script:** `research/deploy_lane_oos_dir_match_audit.py`")
    lines.append(f"**Mode A sacred cutoff:** `HOLDOUT_SACRED_FROM` = {HOLDOUT_SACRED_FROM.isoformat()} (from `trading_app.holdout_policy`)")
    lines.append("**Filter source:** canonical `trading_app.config.ALL_FILTERS` via `research.filter_utils.filter_signal` (no re-encoded logic)")
    lines.append("**Power thresholds:** canonical `research.oos_power.POWER_TIERS` (80% / 50%)")
    lines.append("")
    lines.append("## Audited claim")
    lines.append("")
    lines.append(
        "Mandate #4 from `memory/next_session_mandates_2026_04_20.md`: "
        "per-lane 2026 OOS dir_match audit on the 6 DEPLOY lanes in "
        "`docs/runtime/lane_allocation.json` (`topstep_50k_mnq_auto` rebalance "
        "2026-04-18). Must use canonical `oos_power` helper per RULE 3.3; "
        "must provide per-lane breakdown per RULE 14 — no pooled p-value claim."
    )
    lines.append("")
    lines.append("## Pre-committed decision rule")
    lines.append("")
    lines.append("Locked in `docs/runtime/stages/deploy_lane_oos_dir_match_audit.md` before implementation.")
    lines.append("")
    lines.append("| dir_match | OOS power tier | Lane verdict |")
    lines.append("|---|---|---|")
    lines.append("| TRUE  | CAN_REFUTE            | ALIVE_CONFIRMED |")
    lines.append("| TRUE  | DIRECTIONAL_ONLY      | ALIVE_PROVISIONAL |")
    lines.append("| TRUE  | STATISTICALLY_USELESS | UNVERIFIED_ALIVE |")
    lines.append("| FALSE | CAN_REFUTE            | **DEAD** |")
    lines.append("| FALSE | DIRECTIONAL_ONLY      | WATCH |")
    lines.append("| FALSE | STATISTICALLY_USELESS | UNVERIFIED |")
    lines.append("")
    lines.append("## Lanes under test")
    lines.append("")
    lines.append("| # | strategy_id | session | orb_min | RR | filter |")
    lines.append("|---:|---|---|---:|---:|---|")
    for i, l in enumerate(lanes, 1):
        lines.append(
            f"| L{i} | `{l.strategy_id}` | {l.orb_label} | {l.orb_minutes} | {l.rr_target} | {l.filter_type} |"
        )
    lines.append("")
    lines.append("## IS baseline + 2026 OOS per-lane statistics")
    lines.append("")
    lines.append(
        "| # | strategy_id | IS N | IS mean | IS std | IS WR | OOS N | OOS mean | OOS std | OOS WR | OOS t (1-sample) | p (1-sided) | OOS date span |"
    )
    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for i, a in enumerate(audits, 1):
        lines.append(
            f"| L{i} | `{a.strategy_id}` | {a.is_n} | {a.is_mean:+.4f} | {a.is_std:.3f} | {a.is_wr:.3f} | "
            f"{a.oos_n} | {a.oos_mean:+.4f} | {a.oos_std:.3f} | {a.oos_wr:.3f} | {a.oos_t:+.2f} | {a.oos_p_onesided:.3f} | "
            f"{a.oos_date_min} … {a.oos_date_max} |"
        )
    lines.append("")
    lines.append("## RULE 3.3 power-floor evaluation per lane")
    lines.append("")
    lines.append(
        "| # | strategy_id | Cohen's d (IS) | OOS N | OOS power | Tier (canonical) | N needed for 80% | dir_match | **Verdict** |"
    )
    lines.append("|---:|---|---:|---:|---:|---|---:|:---:|---|")
    for i, a in enumerate(audits, 1):
        dm = "—" if a.dir_match is None else ("TRUE" if a.dir_match else "FALSE")
        lines.append(
            f"| L{i} | `{a.strategy_id}` | {a.cohen_d:.3f} | {a.oos_n} | {a.oos_power*100:.1f}% | "
            f"{a.oos_power_tier} | {a.n_for_80pct_power:,} | {dm} | **{a.verdict}** |"
        )
    lines.append("")
    lines.append("## Portfolio summary")
    lines.append("")
    verdict_counts: dict[str, int] = {}
    for a in audits:
        verdict_counts[a.verdict] = verdict_counts.get(a.verdict, 0) + 1
    for v, c in sorted(verdict_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{v}:** {c}/{len(audits)}")
    lines.append("")
    dead = [a for a in audits if a.verdict == "DEAD"]
    watch = [a for a in audits if a.verdict == "WATCH"]
    unverified = [a for a in audits if a.verdict in ("UNVERIFIED", "UNVERIFIED_ALIVE", "UNVERIFIED_NO_OOS")]
    if dead:
        lines.append(f"**DEAD lanes (retire candidates):** {len(dead)}")
        for a in dead:
            lines.append(f"  - `{a.strategy_id}` — IS {a.is_mean:+.4f} / OOS {a.oos_mean:+.4f}, OOS power {a.oos_power*100:.1f}% (CAN_REFUTE)")
    if watch:
        lines.append(f"**WATCH lanes (shadow-monitor, no action):** {len(watch)}")
        for a in watch:
            lines.append(f"  - `{a.strategy_id}` — IS {a.is_mean:+.4f} / OOS {a.oos_mean:+.4f}, OOS power {a.oos_power*100:.1f}% (DIRECTIONAL_ONLY)")
    if unverified:
        lines.append(f"**UNVERIFIED lanes (power < 50%, do NOT kill):** {len(unverified)}")
        for a in unverified:
            lines.append(f"  - `{a.strategy_id}` — OOS power {a.oos_power*100:.1f}% (STATISTICALLY_USELESS); needs N={a.n_for_80pct_power:,} for 80% power")
    lines.append("")
    lines.append("## RULE 3.3 compliance note")
    lines.append("")
    lines.append(
        "Every `dir_match=FALSE` finding above is paired with its RULE 3.3 power tier. "
        "No lane is labeled DEAD unless its OOS power to detect the IS effect reaches "
        "CAN_REFUTE (≥80%). Where OOS power is STATISTICALLY_USELESS (<50%), the "
        "dir_match=FALSE outcome is noise-consistent and the lane is labeled "
        "UNVERIFIED — not DEAD. This replaces the pre-correction 2026-04-20 "
        "`bull_short_avoidance` error where a 7.9%-power OOS was used as a binary "
        "kill criterion. See `feedback_oos_power_floor.md` + PR #32."
    )
    lines.append("")
    lines.append("## RULE 14 compliance note")
    lines.append("")
    lines.append(
        "This audit is per-lane by construction — no pooled p-value is computed "
        "across the 6 lanes. RULE 14 explicitly requires a per-lane breakdown before "
        "any capital action rests on pooled evidence. See `feedback_per_lane_breakdown_required.md`."
    )
    lines.append("")
    lines.append("## Classification")
    lines.append("")
    lines.append("- **VALID** if canonical filter delegation verified + `oos_power.power_verdict` invoked + per-lane table + no pooled claim.")
    lines.append("- **CONDITIONAL** if any caveat applies (e.g., lanes with IS std near zero causing degenerate Cohen's d).")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append("DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/deploy_lane_oos_dir_match_audit.py")
    lines.append("```")
    lines.append("")
    lines.append("No randomness. Read-only DB. No writes to `validated_setups` / `experimental_strategies` / `live_config` / `lane_allocation.json`.")
    return "\n".join(lines)


def main() -> int:
    alloc = json.loads(ALLOCATION_JSON.read_text())
    lanes = [LaneSpec.from_allocation_row(row) for row in alloc["lanes"]]
    print(f"Loaded {len(lanes)} DEPLOY lanes from {ALLOCATION_JSON.name}")
    print(f"Mode A cutoff: trading_day < {HOLDOUT_SACRED_FROM.isoformat()}")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        audits: list[LaneAudit] = []
        for i, lane in enumerate(lanes, 1):
            print(f"\n[L{i}] {lane.strategy_id}")
            a = audit_lane(con, lane)
            print(
                f"    IS:  N={a.is_n} mean={a.is_mean:+.4f} std={a.is_std:.3f} WR={a.is_wr:.3f}"
            )
            print(
                f"    OOS: N={a.oos_n} mean={a.oos_mean:+.4f} std={a.oos_std:.3f} WR={a.oos_wr:.3f} span={a.oos_date_min}..{a.oos_date_max}"
            )
            print(
                f"    RULE 3.3: Cohen's d={a.cohen_d:.3f} power={a.oos_power*100:.1f}% tier={a.oos_power_tier}"
            )
            print(
                f"    dir_match={a.dir_match} -> **{a.verdict}**"
            )
            audits.append(a)

        doc = render_result_doc(audits, lanes)
        RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
        RESULT_DOC.write_text(doc, encoding="utf-8")
        print(f"\nResult doc: {RESULT_DOC}")
        # Summary
        counts: dict[str, int] = {}
        for a in audits:
            counts[a.verdict] = counts.get(a.verdict, 0) + 1
        print(f"Portfolio verdict counts: {counts}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
