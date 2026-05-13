"""Chordia-audit queue recompute - Mode A canonical, lit-grounded.

Read-only. No DB writes. No validated_setups mutation. No allocator mutation.
No chordia_audit_log writes. No pre-reg authorship.

Produces a ranked CSV (active validated strategies x canonical gate evaluation)
under docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv. The
queue tells the next session which strategies to author per-strategy Chordia
pre-regs for, anchored to canonical sources for every gate.

# scratch-policy: WHERE pnl_r IS NOT NULL on the IS leg; scratch_drop_count
#                 reported per candidate so reviewers can see the realized-eod
#                 approximation. True realized-eod requires bar-level recompute
#                 from bars_1m (deferred to per-strategy pre-reg author per
#                 v2 plan amendment E3).

# e2-lookahead-policy: this script does NOT introduce E2-look-ahead predictors;
#                     it only evaluates already-validated strategies whose
#                     filter_type passed promotion-time look-ahead audits.

Inputs (canonical only):
- gold.db validated_setups (active=True) -> candidate inventory + Sharpe
- gold.db orb_outcomes JOIN daily_features -> Mode A IS + OOS recompute
- docs/audit/results/2026-05-11-mnq-all-active-deployability.json -> MNQ
  family_status / hard_issues / verdict enrichment (786-row snapshot)
- docs/runtime/lane_allocation.json -> allocator state per strategy_id
- docs/runtime/chordia_audit_log.yaml -> existing audit verdicts

Outputs:
- docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv

Canonical delegations (no re-encoding):
- oos_ttest_power / one_sample_power / power_verdict : research.oos_power
- filter_signal                                      : research.filter_utils
- compute_chordia_t                                  : trading_app.chordia
- HOLDOUT_SACRED_FROM                                : trading_app.holdout_policy
- ALL_FILTERS                                        : trading_app.config
- GOLD_DB_PATH                                       : pipeline.paths
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml

# Canonical-source delegations. Every helper below is imported from its
# single source of truth; this script re-implements none of them.
from research.oos_power import one_sample_power, power_verdict
from research.filter_utils import filter_signal
from trading_app.chordia import compute_chordia_t, CHORDIA_T_WITHOUT_THEORY
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.config import ALL_FILTERS, StrategyFilter
from pipeline.paths import GOLD_DB_PATH


# ---------------------------------------------------------------------------
# Repo paths (resolved from script file, not CWD)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = _REPO_ROOT / "docs" / "audit" / "results" / "2026-05-11-mnq-all-active-deployability.json"
LANE_ALLOC_PATH = _REPO_ROOT / "docs" / "runtime" / "lane_allocation.json"
CHORDIA_LOG_PATH = _REPO_ROOT / "docs" / "runtime" / "chordia_audit_log.yaml"
OUTPUT_CSV = _REPO_ROOT / "docs" / "audit" / "results" / "2026-05-12-chordia-audit-queue-candidates.csv"


# ---------------------------------------------------------------------------
# Filter family map (Amendment E1: StrategyFilter has no .family attribute,
# so we map subclass-name -> family-key. Locked at script header; halt on
# any concrete StrategyFilter subclass that does not resolve.)
# ---------------------------------------------------------------------------
_FAMILY_MAP: dict[str, str] = {
    "NoFilter": "NO_FILTER",
    "OrbSizeFilter": "ORB_G",
    "CostRatioFilter": "COST_LT",
    "VolumeFilter": "ORB_VOL",
    "OrbVolumeFilter": "ORB_VOL",
    "CombinedATRVolumeFilter": "ORB_VOL",
    "CrossAssetATRFilter": "CROSS_ASSET_PERCENTILE",
    "ATRVelRatioFilter": "ATR_VEL",
    "OwnATRPercentileFilter": "INTRA_ASSET_PERCENTILE",
    "OvernightRangeFilter": "OVNRNG",
    "GARCHForecastVolPctFilter": "GARCH",
    "OvernightRangeAbsFilter": "OVNRNG",
    "PrevDayRangeNormFilter": "PD_RANGE",
    "PrevDayGeometryFilter": "PD_GEOMETRY",
    "GapNormFilter": "GAP",
    "DirectionFilter": "DIRECTION_CONDITIONAL",
    "CalendarSkipFilter": "CALENDAR_SKIP",
    "DayOfWeekSkipFilter": "DOW_SKIP",
    "ATRVelocityFilter": "ATR_VEL",
    "DoubleBreakFilter": "DBL_BREAK",
    "BreakSpeedFilter": "BREAK_SPEED",
    "BreakBarContinuesFilter": "BRK_CONT",
    "PitRangeFilter": "PIT_RANGE",
    "VWAPBreakDirectionFilter": "VWAP_MID_ALIGNED",
    "CrossSessionMomentumFilter": "X_SESSION_MOMENTUM",
    "CompositeFilter": "COMPOSITE",
}

# v2 plan Criterion 3 classification of filter families:
_PREFER_FAMILIES: set[str] = {
    "CROSS_ASSET_PERCENTILE",
    "INTRA_ASSET_PERCENTILE",
    "DIRECTION_CONDITIONAL",
}
_EXCLUDE_FAMILIES: set[str] = {
    "COST_LT",
    "OVNRNG",
    "ORB_VOL",
    "ORB_G",
    "NO_FILTER",
}


def _assert_family_map_complete() -> None:
    """Halt if any concrete StrategyFilter subclass in ALL_FILTERS misses a map entry."""
    unmapped: list[tuple[str, str]] = []
    for key, filt in ALL_FILTERS.items():
        cls_name = type(filt).__name__
        if cls_name not in _FAMILY_MAP:
            unmapped.append((key, cls_name))
    if unmapped:
        msg = "Unmapped StrategyFilter subclasses found in ALL_FILTERS:\n"
        for k, c in unmapped:
            msg += f"  filter_key={k!r} class={c!r}\n"
        msg += "Add the class -> family key in _FAMILY_MAP at the top of this script."
        raise SystemExit(msg)


def _family_for(filter_key: str) -> str:
    """Resolve filter_key -> canonical family-name via subclass dispatch."""
    if filter_key not in ALL_FILTERS:
        return "UNKNOWN_NOT_IN_ALL_FILTERS"
    filt: StrategyFilter = ALL_FILTERS[filter_key]
    cls_name = type(filt).__name__
    return _FAMILY_MAP.get(cls_name, f"UNMAPPED:{cls_name}")


# ---------------------------------------------------------------------------
# Bucket codes (Amendment E2: drop undefined G1-G14; use 7 canonical
# hard_issues + 3 new MES/MGC gap codes = 10 total)
# ---------------------------------------------------------------------------
_CANONICAL_HARD_ISSUES = {
    "c8_not_passed",
    "e2_deployment_unsafe_filter",
    "family_purged",
    "family_singleton",
    "replay_mismatch",
    "sample_size_below_deploy_threshold",
    "slippage_missing",
}
_NEW_GAP_CODES = {
    "NO_CHORDIA_AUDIT_LOG_ENTRY",
    "INSTRUMENT_REGIME_COLD_OR_WARM",
    "MODE_A_IS_EMPTY",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateRow:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    entry_model: str
    confirm_bars: int
    filter_type: str
    filter_family: str
    sample_size_stored: int
    sharpe_ratio_stored: float
    expectancy_r_stored: float
    last_trade_day: str
    years_tested: float
    c8_oos_status: str
    promotion_provenance: str
    family_status: str  # MNQ only from JSON; UNKNOWN for MES/MGC
    hard_issues_json: str  # comma-joined; empty string if none
    json_verdict: str  # MNQ only; empty for MES/MGC


@dataclass
class GateOutput:
    n_is_mode_a: int = 0
    mode_a_expr: float = float("nan")
    mode_a_std: float = float("nan")
    scratch_drop_count: int = 0
    scratch_drop_rate: float = float("nan")
    n_oos: int = 0
    oos_expr: float = float("nan")
    chordia_t: float = float("nan")
    chordia_passes_strict: bool = False
    oos_power: float = float("nan")
    oos_cohen_d: float = float("nan")
    oos_power_tier: str = "UNKNOWN"
    chordia_log_verdict: str = ""
    chordia_log_age_days: float = float("nan")
    allocator_status: str = "NOT_IN_LANE_ALLOC"
    blockers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------
def _load_candidate_inventory(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """All active validated strategies (3 instruments). Canonical source: validated_setups."""
    df = con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes,
               rr_target, entry_model, confirm_bars, filter_type,
               sample_size, sharpe_ratio, expectancy_r,
               last_trade_day, years_tested, c8_oos_status,
               promotion_provenance
        FROM validated_setups
        WHERE status = 'active'
        ORDER BY instrument, strategy_id
        """
    ).fetch_df()
    return df


def _load_json_enrichment() -> dict[str, dict[str, Any]]:
    """Return dict strategy_id -> {family_status, hard_issues, verdict} from MNQ JSON.

    JSON has 4 buckets (deployable_now/nearest/research_only/retire_or_purge)
    but only ~250 visible rows; the rest are truncated. Visible rows enrich.
    """
    data = json.loads(JSON_PATH.read_text())
    enriched: dict[str, dict[str, Any]] = {}
    for bucket_key in ("deployable_now", "nearest_to_deployable", "research_only", "retire_or_purge"):
        bucket = data.get("promotion_queue", {}).get(bucket_key, {})
        for row in bucket.get("rows", []):
            sid = row.get("strategy_id")
            if sid:
                enriched[sid] = {
                    "family_status": row.get("family_status", "UNKNOWN"),
                    "hard_issues": row.get("hard_issues", []) or [],
                    "verdict": row.get("verdict", ""),
                    "json_bucket": bucket_key,
                }
    return enriched


def _load_lane_allocation() -> dict[str, str]:
    """strategy_id -> allocator status string (DEPLOY / PAUSE / etc)."""
    data = json.loads(LANE_ALLOC_PATH.read_text())
    out: dict[str, str] = {}
    for lane in data.get("lanes", []):
        sid = lane.get("strategy_id")
        if sid:
            out[sid] = lane.get("verdict", lane.get("status", "UNKNOWN_ALLOC_STATE"))
    return out


def _load_chordia_log() -> dict[str, dict[str, Any]]:
    """strategy_id -> {verdict, audit_date} from the canonical YAML."""
    data = yaml.safe_load(CHORDIA_LOG_PATH.read_text())
    out: dict[str, dict[str, Any]] = {}
    for audit in data.get("audits", []) or []:
        sid = audit.get("strategy_id")
        if sid:
            out[sid] = {
                "verdict": audit.get("verdict", ""),
                "audit_date": str(audit.get("audit_date", "")),
            }
    return out


# ---------------------------------------------------------------------------
# Mode-A canonical recompute
# ---------------------------------------------------------------------------
def _recompute_mode_a(
    con: duckdb.DuckDBPyConnection,
    cand: CandidateRow,
) -> tuple[int, float, float, int, int, float]:
    """Return (n_is_mode_a, mode_a_expr, mode_a_std, scratch_drop_count, n_oos, oos_expr).

    Reads canonical orb_outcomes JOIN daily_features (triple-join on
    trading_day + symbol + orb_minutes per daily-features-joins.md), then
    applies the canonical filter via research.filter_utils.filter_signal.
    """
    df = con.execute(
        """
        SELECT o.trading_day, o.entry_ts, o.pnl_r,
               d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.rr_target = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
        ORDER BY o.trading_day
        """,
        [
            cand.instrument,
            cand.orb_label,
            int(cand.orb_minutes),
            float(cand.rr_target),
            cand.entry_model,
            int(cand.confirm_bars),
        ],
    ).fetch_df()

    if len(df) == 0:
        return 0, float("nan"), float("nan"), 0, 0, float("nan")

    # Apply canonical filter via filter_utils (delegates to ALL_FILTERS[filter_key].matches_df).
    try:
        fire = filter_signal(df, cand.filter_type, cand.orb_label)
    except KeyError:
        # Filter not registered (rare for active validated rows); mark gate-empty.
        return 0, float("nan"), float("nan"), 0, 0, float("nan")

    df = df.assign(_fire=fire)
    df_fired = df[df["_fire"] == 1].copy()

    # DuckDB returns trading_day as datetime64; HOLDOUT_SACRED_FROM is a date.
    # Use pd.Timestamp on the constant to satisfy dtype-aware comparison.
    holdout_ts = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_mask = df_fired["trading_day"] < holdout_ts
    oos_mask = df_fired["trading_day"] >= holdout_ts

    is_rows = df_fired[is_mask]
    oos_rows = df_fired[oos_mask]

    # Scratch handling (E3 amendment): drop NULL pnl_r for ExpR computation
    # but record the count for the realized-eod approximation flag.
    is_with_pnl = is_rows[is_rows["pnl_r"].notna()]
    oos_with_pnl = oos_rows[oos_rows["pnl_r"].notna()]
    scratch_drop_count = int(len(is_rows) - len(is_with_pnl))

    if len(is_with_pnl) == 0:
        mode_a_expr = float("nan")
        mode_a_std = float("nan")
    else:
        mode_a_expr = float(is_with_pnl["pnl_r"].mean())
        mode_a_std = float(is_with_pnl["pnl_r"].std(ddof=1)) if len(is_with_pnl) >= 2 else float("nan")

    oos_expr = float(oos_with_pnl["pnl_r"].mean()) if len(oos_with_pnl) > 0 else float("nan")

    return (
        int(len(is_with_pnl)),
        mode_a_expr,
        mode_a_std,
        scratch_drop_count,
        int(len(oos_with_pnl)),
        oos_expr,
    )


# ---------------------------------------------------------------------------
# Gate application
# ---------------------------------------------------------------------------
def _apply_gates(
    cand: CandidateRow,
    gate: GateOutput,
    chordia_log: dict[str, dict[str, Any]],
    lane_alloc: dict[str, str],
) -> GateOutput:
    """Apply Criterion 1 (a-g), Criterion 2 (OOS power), Criterion 3 (filter class).

    Mutates and returns the gate; appends blockers from the canonical 10-code list.
    """
    # --- Allocator state ---
    gate.allocator_status = lane_alloc.get(cand.strategy_id, "NOT_IN_LANE_ALLOC")

    # --- Chordia audit log status ---
    log_entry = chordia_log.get(cand.strategy_id)
    if log_entry is None:
        gate.blockers.append("NO_CHORDIA_AUDIT_LOG_ENTRY")
        gate.chordia_log_verdict = ""
    else:
        gate.chordia_log_verdict = log_entry.get("verdict", "")
        # Audit-age computation (canonical: 2026-05-12 - audit_date)
        try:
            audit_date = pd.to_datetime(log_entry["audit_date"]).date()
            today = pd.Timestamp("2026-05-12").date()
            gate.chordia_log_age_days = float((today - audit_date).days)
        except Exception:
            gate.chordia_log_age_days = float("nan")

    # --- Criterion 1 (a-g) Chordia readiness ---
    # (a) validated_setups row exists with status=active : already filtered upstream
    # (b) sample_size >= 100 : Criterion 7 deployable floor
    if cand.sample_size_stored < 100:
        gate.blockers.append("sample_size_below_deploy_threshold")
    # (c) sharpe_ratio non-null : enforced upstream (all 844 rows have it)
    # (d) family_status not in PURGED/SINGLETON (MNQ only)
    if cand.family_status == "PURGED":
        gate.blockers.append("family_purged")
    elif cand.family_status == "SINGLETON":
        gate.blockers.append("family_singleton")
    # (e) c8_oos_status not in {NEGATIVE_OOS_EXPR, FAILED_RATIO}
    if cand.c8_oos_status in {"NEGATIVE_OOS_EXPR", "FAILED_RATIO"}:
        gate.blockers.append("c8_not_passed")
    # (f) Mode-A n_is >= 50 deployable-sample floor
    if gate.n_is_mode_a < 50:
        gate.blockers.append("MODE_A_IS_EMPTY")

    # JSON-derived hard_issues (replay_mismatch / slippage_missing / e2_deployment_unsafe_filter)
    if cand.hard_issues_json:
        for issue in cand.hard_issues_json.split(","):
            issue = issue.strip()
            if issue in _CANONICAL_HARD_ISSUES and issue not in gate.blockers:
                gate.blockers.append(issue)

    # --- Chordia-t computation ---
    try:
        gate.chordia_t = compute_chordia_t(cand.sharpe_ratio_stored, cand.sample_size_stored)
        gate.chordia_passes_strict = gate.chordia_t >= CHORDIA_T_WITHOUT_THEORY
    except ValueError:
        gate.chordia_t = float("nan")
        gate.chordia_passes_strict = False

    # --- Criterion 2: OOS power (one-sample t-test on strategy's own returns) ---
    if (
        gate.n_is_mode_a >= 2
        and not math.isnan(gate.mode_a_expr)
        and not math.isnan(gate.mode_a_std)
        and gate.mode_a_std > 0
        and gate.n_oos >= 2
    ):
        d = abs(gate.mode_a_expr) / gate.mode_a_std
        gate.oos_cohen_d = float(d)
        gate.oos_power = float(one_sample_power(d=d, n=gate.n_oos, alpha=0.05))
        gate.oos_power_tier = power_verdict(gate.oos_power)
    else:
        gate.oos_cohen_d = float("nan")
        gate.oos_power = float("nan")
        gate.oos_power_tier = "STATISTICALLY_USELESS"

    return gate


# ---------------------------------------------------------------------------
# Tier + sort
# ---------------------------------------------------------------------------
def _queue_tier(cand: CandidateRow, gate: GateOutput) -> str:
    """Return tier label per v2 plan sort rule + AUDIT_GAP_ONLY fallback.

    TOP: passes Crit 1 + Crit 2=CAN_REFUTE + Crit 3 in PREFER
    READY: passes Crit 1 + Crit 2 in CAN_REFUTE/DIRECTIONAL_ONLY
    AUDIT_GAP_ONLY: only blocker is NO_CHORDIA_AUDIT_LOG_ENTRY AND Chordia-t
                     passes strict floor AND filter family not in EXCLUDE.
                     This is the HANDOFF item #2 target: strategies that
                     pass every canonical gate except they haven't been
                     per-strategy Chordia-audited yet. OOS power is below
                     CAN_REFUTE/DIRECTIONAL_ONLY because the OOS window is
                     only ~70 trading days, but the IS evidence is strong
                     enough that an audit is warranted.
    BLOCKED_ON_GAP: has any other blocker
    DEFERRED_FILTER_EXCLUDED: filter family in EXCLUDE
    """
    if cand.filter_family in _EXCLUDE_FAMILIES:
        return "DEFERRED_FILTER_EXCLUDED"
    if gate.blockers:
        # AUDIT_GAP_ONLY: only blocker is the missing audit-log entry
        # AND Chordia-t already clears strict floor.
        if (
            set(gate.blockers) == {"NO_CHORDIA_AUDIT_LOG_ENTRY"}
            and gate.chordia_passes_strict
        ):
            return "AUDIT_GAP_ONLY"
        return "BLOCKED_ON_GAP"
    if gate.oos_power_tier == "CAN_REFUTE" and cand.filter_family in _PREFER_FAMILIES:
        return "TOP"
    if gate.oos_power_tier in ("CAN_REFUTE", "DIRECTIONAL_ONLY"):
        return "READY"
    return "BLOCKED_ON_GAP"


def _sort_key(row: dict[str, Any]) -> tuple:
    """Sort: tier > power > filter pref > mode_a_expr desc > years desc > N desc."""
    tier_rank = {
        "TOP": 0,
        "READY": 1,
        "AUDIT_GAP_ONLY": 2,
        "BLOCKED_ON_GAP": 3,
        "DEFERRED_FILTER_EXCLUDED": 4,
    }
    power_rank = {
        "CAN_REFUTE": 0,
        "DIRECTIONAL_ONLY": 1,
        "STATISTICALLY_USELESS": 2,
        "UNKNOWN": 3,
    }
    pref_rank = 0 if row["filter_family"] in _PREFER_FAMILIES else 1
    expr_neg = -(row["mode_a_expr"] if not math.isnan(row["mode_a_expr"]) else -999.0)
    years_neg = -(row["years_tested"] or 0)
    n_neg = -(row["sample_size_stored"] or 0)
    return (
        tier_rank.get(row["queue_tier"], 99),
        power_rank.get(row["oos_power_tier"], 99),
        pref_rank,
        expr_neg,
        years_neg,
        n_neg,
    )


# ---------------------------------------------------------------------------
# Self-consistency check (E5 amendment: threshold 0.02 -> 0.05)
# ---------------------------------------------------------------------------
_CANONICAL_PASSED = {
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
}


def _self_consistency_halt(rows: list[dict[str, Any]]) -> None:
    """Halt the script if any canonical Chordia-PASSED strategy shows |stored - mode_a| > 0.05.

    Per E5 amendment: 0.05 is the realistic noise threshold; tighter false-positives
    on lanes that already passed strict-unlock CSVs at deltas up to ~0.037.
    """
    by_sid = {r["strategy_id"]: r for r in rows}
    failures: list[str] = []
    for sid in _CANONICAL_PASSED:
        r = by_sid.get(sid)
        if r is None:
            failures.append(f"  MISSING from CSV: {sid}")
            continue
        stored = r["expectancy_r_stored"]
        mode_a = r["mode_a_expr"]
        if math.isnan(stored) or math.isnan(mode_a):
            failures.append(f"  {sid}: NaN in stored ({stored}) or mode_a ({mode_a})")
            continue
        delta = abs(stored - mode_a)
        print(
            f"  self-consistency: {sid}: stored={stored:.4f} "
            f"mode_a={mode_a:.4f} delta={delta:.4f}"
        )
        if delta >= 0.05:
            failures.append(
                f"  {sid}: |stored - mode_a| = {delta:.4f} >= 0.05 "
                f"(stored={stored:.4f}, mode_a={mode_a:.4f})"
            )
    if failures:
        msg = (
            "Self-consistency halt: canonical Chordia-PASSED strategies show "
            "Mode-A drift >= 0.05R. This is a class-bug signal in validated_setups "
            "or the recompute pipeline, not a per-row issue. Propose a separate "
            "validated_setups revalidation before this queue lands.\n\n"
            + "\n".join(failures)
        )
        raise SystemExit(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Chordia-audit queue recompute (Mode A canonical, lit-grounded)")
    print("=" * 70)

    # Pre-flight: family map covers every ALL_FILTERS subclass
    print("\n[0/6] Asserting _FAMILY_MAP covers ALL_FILTERS subclasses...")
    _assert_family_map_complete()
    print(f"      OK ({len(ALL_FILTERS)} ALL_FILTERS entries; "
          f"{len(_FAMILY_MAP)} subclass keys)")

    # Inputs
    print("\n[1/6] Loading candidate inventory from validated_setups...")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    cand_df = _load_candidate_inventory(con)
    print(f"      {len(cand_df)} active validated strategies "
          f"(by instrument: "
          f"{cand_df.groupby('instrument').size().to_dict()})")

    print("\n[2/6] Loading JSON enrichment (MNQ family_status, hard_issues)...")
    json_enrich = _load_json_enrichment()
    print(f"      {len(json_enrich)} MNQ rows enriched from JSON")

    print("\n[3/6] Loading lane_allocation + chordia_audit_log...")
    lane_alloc = _load_lane_allocation()
    chordia_log = _load_chordia_log()
    print(f"      {len(lane_alloc)} lanes in allocator; "
          f"{len(chordia_log)} entries in chordia audit log")

    # Mode-A recompute + gate per row
    print("\n[4/6] Mode-A recompute + gate application per strategy...")
    rows_out: list[dict[str, Any]] = []
    for i, r in enumerate(cand_df.itertuples(index=False), 1):
        json_row = json_enrich.get(r.strategy_id, {})
        cand = CandidateRow(
            strategy_id=r.strategy_id,
            instrument=r.instrument,
            orb_label=r.orb_label,
            orb_minutes=int(r.orb_minutes),
            rr_target=float(r.rr_target),
            entry_model=r.entry_model,
            confirm_bars=int(r.confirm_bars),
            filter_type=r.filter_type,
            filter_family=_family_for(r.filter_type),
            sample_size_stored=int(r.sample_size),
            sharpe_ratio_stored=float(r.sharpe_ratio),
            expectancy_r_stored=float(r.expectancy_r),
            last_trade_day=str(r.last_trade_day),
            years_tested=float(r.years_tested or 0),
            c8_oos_status=str(r.c8_oos_status or ""),
            promotion_provenance=str(r.promotion_provenance or ""),
            family_status=json_row.get("family_status", "UNKNOWN"),
            hard_issues_json=",".join(json_row.get("hard_issues", [])),
            json_verdict=json_row.get("verdict", ""),
        )
        gate = GateOutput()
        n_is, ma_expr, ma_std, scratch, n_oos, oos_expr = _recompute_mode_a(con, cand)
        gate.n_is_mode_a = n_is
        gate.mode_a_expr = ma_expr
        gate.mode_a_std = ma_std
        gate.scratch_drop_count = scratch
        denom = n_is + scratch
        gate.scratch_drop_rate = (scratch / denom) if denom > 0 else float("nan")
        gate.n_oos = n_oos
        gate.oos_expr = oos_expr
        gate = _apply_gates(cand, gate, chordia_log, lane_alloc)

        rows_out.append({
            "strategy_id": cand.strategy_id,
            "instrument": cand.instrument,
            "orb_label": cand.orb_label,
            "orb_minutes": cand.orb_minutes,
            "rr_target": cand.rr_target,
            "entry_model": cand.entry_model,
            "confirm_bars": cand.confirm_bars,
            "filter_type": cand.filter_type,
            "filter_family": cand.filter_family,
            "sample_size_stored": cand.sample_size_stored,
            "sharpe_ratio_stored": cand.sharpe_ratio_stored,
            "expectancy_r_stored": cand.expectancy_r_stored,
            "last_trade_day": cand.last_trade_day,
            "years_tested": cand.years_tested,
            "c8_oos_status": cand.c8_oos_status,
            "promotion_provenance": cand.promotion_provenance,
            "family_status": cand.family_status,
            "json_verdict": cand.json_verdict,
            "hard_issues_json": cand.hard_issues_json,
            "n_is_mode_a": gate.n_is_mode_a,
            "mode_a_expr": gate.mode_a_expr,
            "mode_a_std": gate.mode_a_std,
            "stored_minus_mode_a": cand.expectancy_r_stored - gate.mode_a_expr if not math.isnan(gate.mode_a_expr) else float("nan"),
            "scratch_drop_count": gate.scratch_drop_count,
            "scratch_drop_rate": gate.scratch_drop_rate,
            "n_oos": gate.n_oos,
            "oos_expr": gate.oos_expr,
            "chordia_t": gate.chordia_t,
            "chordia_passes_strict": gate.chordia_passes_strict,
            "oos_cohen_d": gate.oos_cohen_d,
            "oos_power": gate.oos_power,
            "oos_power_tier": gate.oos_power_tier,
            "chordia_log_verdict": gate.chordia_log_verdict,
            "chordia_log_age_days": gate.chordia_log_age_days,
            "allocator_status": gate.allocator_status,
            "blockers": "|".join(gate.blockers),
            "queue_tier": _queue_tier(cand, gate),
        })

        if i % 100 == 0:
            print(f"      progress: {i}/{len(cand_df)}")

    con.close()

    # Self-consistency check (E5)
    print("\n[5/6] Self-consistency check on 3 canonical Chordia-PASSED strategies...")
    _self_consistency_halt(rows_out)

    # Sort + write CSV
    print("\n[6/6] Sorting + writing CSV...")
    rows_out.sort(key=_sort_key)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"      wrote {OUTPUT_CSV.relative_to(_REPO_ROOT)} "
          f"({len(rows_out)} rows)")

    # Final summary by tier
    print("\n=== Queue summary by tier ===")
    tier_counts: dict[str, int] = {}
    for r in rows_out:
        tier_counts[r["queue_tier"]] = tier_counts.get(r["queue_tier"], 0) + 1
    for tier in ("TOP", "READY", "AUDIT_GAP_ONLY", "BLOCKED_ON_GAP", "DEFERRED_FILTER_EXCLUDED"):
        print(f"  {tier}: {tier_counts.get(tier, 0)}")

    # Halt-on-zero-actionable: TOP/READY/AUDIT_GAP_ONLY all empty
    actionable = (
        tier_counts.get("TOP", 0)
        + tier_counts.get("READY", 0)
        + tier_counts.get("AUDIT_GAP_ONLY", 0)
    )
    if actionable == 0:
        print("\nWARN: 0 actionable candidates (TOP/READY/AUDIT_GAP_ONLY). "
              "Top-3 recommendation will report NO_ACTIONABLE_CANDIDATE.")

    # Diagnostic: top-15 preview
    print("\n=== Top-15 preview ===")
    for r in rows_out[:15]:
        print(f"  {r['queue_tier']:25} {r['strategy_id']:55} "
              f"tier={r['oos_power_tier']:20} "
              f"t={r['chordia_t']:.2f} "
              f"mode_a_expr={r['mode_a_expr']:.4f} "
              f"N_IS={r['n_is_mode_a']}")


if __name__ == "__main__":
    main()
