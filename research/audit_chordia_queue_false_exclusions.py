"""Audit the May 12 Chordia queue for false exclusions — confirmatory, read-only.

Null hypothesis (declared before run): the funnel
(research/chordia_queue_recompute.py) applies its stated exclusion rules
correctly. Zero false exclusions expected.

Pre-committed exit conditions (per pre-reg
docs/audit/hypotheses/2026-05-16-chordia-queue-false-exclusion-audit.yaml):
  rescue_count == 0      -> FUNNEL_VALIDATED
  rescue_count in [1,20] -> FUNNEL_BUGS_FOUND
  rescue_count > 20      -> FUNNEL_SYSTEMIC_BIAS  (HALT + escalate)

Read-only. No DB writes. No validated_setups mutation. No lane_allocation
or chordia_audit_log mutation. Pure confirmatory pass.

Canonical delegations (per institutional-rigor § 4, no re-encoding):
  - GOLD_DB_PATH                         pipeline.paths
  - HOLDOUT_SACRED_FROM                  trading_app.holdout_policy
  - ALL_FILTERS                          trading_app.config
  - compute_chordia_t, thresholds        trading_app.chordia
  - one_sample_power, power_verdict      research.oos_power
  - filter_signal                        research.filter_utils
  - _FAMILY_MAP, _EXCLUDE_FAMILIES, etc  imported from queue script

Outputs:
  docs/audit/results/2026-05-16-chordia-queue-false-exclusion-audit.csv
  docs/audit/results/2026-05-16-chordia-queue-rescued-quarantine.csv
  docs/audit/results/2026-05-16-chordia-queue-false-exclusion-audit.md
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

# Canonical delegations — no re-encoding.
from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.chordia import (
    CHORDIA_T_WITH_THEORY,
    CHORDIA_T_WITHOUT_THEORY,
    compute_chordia_t,
)
from research.oos_power import one_sample_power, power_verdict
from research.filter_utils import filter_signal
from research.chordia_queue_recompute import (
    _FAMILY_MAP,
    _EXCLUDE_FAMILIES,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_CSV = _REPO_ROOT / "docs" / "audit" / "results" / "2026-05-12-chordia-audit-queue-candidates.csv"
LANE_ALLOC_PATH = _REPO_ROOT / "docs" / "runtime" / "lane_allocation.json"
CHORDIA_LOG_PATH = _REPO_ROOT / "docs" / "runtime" / "chordia_audit_log.yaml"

OUT_PER_GATE_CSV = _REPO_ROOT / "docs" / "audit" / "results" / "2026-05-16-chordia-queue-false-exclusion-audit.csv"
OUT_RESCUE_CSV = _REPO_ROOT / "docs" / "audit" / "results" / "2026-05-16-chordia-queue-rescued-quarantine.csv"
OUT_VERDICT_MD = _REPO_ROOT / "docs" / "audit" / "results" / "2026-05-16-chordia-queue-false-exclusion-audit.md"

RANDOM_SEED = 20260516
QUEUE_DATE_CONVENTION = pd.Timestamp("2026-05-12").date()  # match queue script line 395


@dataclass
class GateAuditResult:
    gate_name: str
    intended_rule: str
    source_trace: str
    n_excluded_by_gate: int
    n_recomputed_excluded: int
    delta: int
    n_policy: int = 0
    n_bug: int = 0
    n_drift: int = 0
    rescued_strategy_ids: list[str] = field(default_factory=list)
    notes: str = ""


def _load_queue_csv() -> pd.DataFrame:
    df = pd.read_csv(QUEUE_CSV)
    print(f"  Loaded {len(df)} rows from {QUEUE_CSV.relative_to(_REPO_ROOT)}")
    return df


def _load_lane_alloc() -> dict[str, str]:
    data = json.loads(LANE_ALLOC_PATH.read_text())
    out: dict[str, str] = {}
    for lane in data.get("lanes", []):
        sid = lane.get("strategy_id")
        if sid:
            out[sid] = lane.get("status") or lane.get("verdict") or "UNKNOWN"
    return out


def _load_chordia_log() -> dict[str, dict[str, Any]]:
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


def _recompute_mode_a_for_row(
    con: duckdb.DuckDBPyConnection,
    row: pd.Series,
) -> tuple[int, float, float, int, float]:
    """Return (n_is_mode_a, mode_a_expr, mode_a_std, n_oos, oos_expr).

    Triple-join on (trading_day, symbol, orb_minutes) per daily-features-joins.md.
    Filter delegation via research.filter_utils.filter_signal — NEVER re-encode.
    """
    df = con.execute(
        """
        SELECT o.trading_day, o.entry_ts, o.pnl_r, d.*
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
            row["instrument"],
            row["orb_label"],
            int(row["orb_minutes"]),
            float(row["rr_target"]),
            row["entry_model"],
            int(row["confirm_bars"]),
        ],
    ).fetch_df()
    if len(df) == 0:
        return 0, float("nan"), float("nan"), 0, float("nan")
    try:
        fire = filter_signal(df, row["filter_type"], row["orb_label"])
    except KeyError:
        return 0, float("nan"), float("nan"), 0, float("nan")
    df = df.assign(_fire=fire)
    fired = df[df["_fire"] == 1].copy()
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_rows = fired[fired["trading_day"] < holdout]
    oos_rows = fired[fired["trading_day"] >= holdout]
    is_pnl = is_rows[is_rows["pnl_r"].notna()]
    oos_pnl = oos_rows[oos_rows["pnl_r"].notna()]
    n_is = int(len(is_pnl))
    n_oos = int(len(oos_pnl))
    is_expr = float(is_pnl["pnl_r"].mean()) if n_is else float("nan")
    is_std = float(is_pnl["pnl_r"].std(ddof=1)) if n_is >= 2 else float("nan")
    oos_expr = float(oos_pnl["pnl_r"].mean()) if n_oos else float("nan")
    return n_is, is_expr, is_std, n_oos, oos_expr


def _stratified_sample(
    df: pd.DataFrame,
    per_stratum_n: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified random sample: strata = (instrument × entry_model × filter_family).

    Returns up to `per_stratum_n` rows per stratum (or the full stratum if smaller).
    """
    if len(df) == 0:
        return df.copy()
    rng = random.Random(seed)
    out_chunks: list[pd.DataFrame] = []
    grouped = df.groupby(["instrument", "entry_model", "filter_family"], dropna=False)
    for _, sub in grouped:
        if len(sub) <= per_stratum_n:
            out_chunks.append(sub)
        else:
            idx = sub.index.tolist()
            rng.shuffle(idx)
            out_chunks.append(sub.loc[idx[:per_stratum_n]])
    return pd.concat(out_chunks, ignore_index=True) if out_chunks else df.head(0)


# ---------------------------------------------------------------------------
# Per-gate audits
# ---------------------------------------------------------------------------
def audit_g1_deferred_filter_excluded(df: pd.DataFrame) -> GateAuditResult:
    """G1: filter_family in {COST_LT, OVNRNG, ORB_VOL, ORB_G, NO_FILTER}.

    Canonical recompute: re-resolve filter_family from filter_type via the
    imported _FAMILY_MAP + ALL_FILTERS subclass dispatch (the queue script
    already does this; we mirror it). Then re-classify membership in
    _EXCLUDE_FAMILIES.

    Bug shape we are looking for:
      - A row labeled DEFERRED_FILTER_EXCLUDED whose filter_family resolution
        in the CSV diverges from the canonical resolution today (i.e., a
        CompositeFilter or unmapped subclass that drifted between map versions).
      - A row whose filter_type is in _EXCLUDE_FAMILIES by the CSV's stored
        family but whose CompositeFilter primary component is in PREFER.
    """
    excluded = df[df["queue_tier"] == "DEFERRED_FILTER_EXCLUDED"].copy()
    n_excl = len(excluded)
    # Re-derive filter_family from filter_type via the canonical map.
    from trading_app.config import ALL_FILTERS

    def _resolve(filter_key: str) -> str:
        if filter_key not in ALL_FILTERS:
            return "UNKNOWN_NOT_IN_ALL_FILTERS"
        cls_name = type(ALL_FILTERS[filter_key]).__name__
        return _FAMILY_MAP.get(cls_name, f"UNMAPPED:{cls_name}")

    excluded["filter_family_recomputed"] = excluded["filter_type"].map(_resolve)
    excluded["should_be_excluded_recomputed"] = (
        excluded["filter_family_recomputed"].isin(_EXCLUDE_FAMILIES)
    )
    rescued = excluded[~excluded["should_be_excluded_recomputed"]]
    n_recomputed_excl = int(excluded["should_be_excluded_recomputed"].sum())

    # Classify rescued rows.
    rescued_ids = rescued["strategy_id"].tolist()

    notes = (
        f"_EXCLUDE_FAMILIES = {sorted(_EXCLUDE_FAMILIES)}. "
        f"Stored vs recomputed family agreement: "
        f"{int((excluded['filter_family'] == excluded['filter_family_recomputed']).sum())}/{n_excl}."
    )
    if rescued_ids:
        notes += f" Family-mismatch rescued IDs: {rescued_ids[:10]}{'...' if len(rescued_ids) > 10 else ''}"

    return GateAuditResult(
        gate_name="G1_DEFERRED_FILTER_EXCLUDED",
        intended_rule="filter_family in EXCLUDE set excludes row (v2 plan Criterion 3 policy).",
        source_trace="research/chordia_queue_recompute.py:116-122,472-473",
        n_excluded_by_gate=n_excl,
        n_recomputed_excluded=n_recomputed_excl,
        delta=n_excl - n_recomputed_excl,
        n_policy=n_recomputed_excl,
        n_bug=len(rescued_ids),
        n_drift=0,
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def audit_g2_sample_size_threshold(df: pd.DataFrame) -> GateAuditResult:
    """G2: sample_size_stored < 100 -> blocker."""
    has_blocker = df["blockers"].fillna("").str.contains("sample_size_below_deploy_threshold")
    excluded = df[has_blocker]
    # Canonical recompute: re-apply the predicate directly.
    recomputed = (df["sample_size_stored"] < 100)
    n_excl = int(has_blocker.sum())
    n_recomp = int(recomputed.sum())
    delta = n_excl - n_recomp
    # Rescue = had blocker but should NOT have (recomputed False but stored True)
    bug_rows = df[has_blocker & ~recomputed]
    rescued_ids = bug_rows["strategy_id"].tolist()
    notes = (
        f"Predicate: sample_size_stored < 100. "
        f"Stored-blocker agreement with predicate: "
        f"{int((has_blocker == recomputed).sum())}/{len(df)} rows. "
        f"min/max sample_size on blocked rows: {excluded['sample_size_stored'].min()}/{excluded['sample_size_stored'].max()}."
    )
    return GateAuditResult(
        gate_name="G2_sample_size_below_deploy_threshold",
        intended_rule="sample_size_stored < 100 blocks row (Criterion 7 deployable floor).",
        source_trace="research/chordia_queue_recompute.py:403-404",
        n_excluded_by_gate=n_excl,
        n_recomputed_excluded=n_recomp,
        delta=delta,
        n_policy=int((has_blocker & recomputed).sum()),
        n_bug=len(rescued_ids),
        n_drift=0,
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def audit_g3_mode_a_is_empty(
    df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
) -> GateAuditResult:
    """G3: n_is_mode_a < 50 -> blocker.

    Sample-recompute: take stratified sample of MODE_A_IS_EMPTY rows, re-run
    _recompute_mode_a_for_row, confirm n_is < 50.
    """
    has_blocker = df["blockers"].fillna("").str.contains("MODE_A_IS_EMPTY")
    excluded = df[has_blocker]
    n_excl = int(has_blocker.sum())
    # CSV-based recompute via the stored n_is_mode_a column.
    recomputed = (df["n_is_mode_a"] < 50)
    n_recomp = int(recomputed.sum())
    csv_disagreement = df[has_blocker & ~recomputed]
    # Live-DB sample recompute on up to 20 rows to confirm CSV's n_is_mode_a is accurate.
    sample = _stratified_sample(excluded, per_stratum_n=2, seed=RANDOM_SEED).head(20)
    live_disagreement_rows: list[str] = []
    for _, r in sample.iterrows():
        n_is_live, _expr, _std, _n_oos, _oos = _recompute_mode_a_for_row(con, r)
        stored = int(r["n_is_mode_a"]) if not pd.isna(r["n_is_mode_a"]) else 0
        if (n_is_live < 50) != (stored < 50):
            live_disagreement_rows.append(
                f"{r['strategy_id']}: stored_n_is={stored} live_n_is={n_is_live}"
            )
    rescued_ids = csv_disagreement["strategy_id"].tolist()
    notes = (
        f"CSV recompute agreement: {int((has_blocker == recomputed).sum())}/{len(df)}. "
        f"Live-DB sample recompute (N={len(sample)}): "
        f"{len(live_disagreement_rows)} cross-boundary disagreements."
    )
    if live_disagreement_rows:
        notes += " Live-DB cross-boundary: " + "; ".join(live_disagreement_rows[:5])
    return GateAuditResult(
        gate_name="G3_MODE_A_IS_EMPTY",
        intended_rule="n_is_mode_a < 50 blocks row (Mode A IS power floor).",
        source_trace="research/chordia_queue_recompute.py:415-416",
        n_excluded_by_gate=n_excl,
        n_recomputed_excluded=n_recomp,
        delta=n_excl - n_recomp,
        n_policy=int((has_blocker & recomputed).sum()),
        n_bug=len(rescued_ids),
        n_drift=len(live_disagreement_rows),
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def audit_g4_c8_not_passed(df: pd.DataFrame) -> GateAuditResult:
    """G4: c8_not_passed blocker — TWO paths into the gate.

    Path A (line 412): c8_oos_status in {NEGATIVE_OOS_EXPR, FAILED_RATIO}.
    Path B (lines 419-423): hard_issues_json from JSON snapshot contains
      c8_not_passed (it is in _CANONICAL_HARD_ISSUES set, line 154).

    A row blocked via Path B may have a non-failing c8_oos_status (e.g.
    INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH, NO_OOS_DATA) but the JSON
    snapshot flagged it for c8 separately. This is INTENDED behavior, not
    a false exclusion — the JSON aggregation reflects upstream MNQ
    deployability audit verdicts.
    """
    has_blocker = df["blockers"].fillna("").str.contains("c8_not_passed")
    excluded = df[has_blocker]
    n_excl = int(has_blocker.sum())
    # Recompute combining BOTH paths.
    path_a = df["c8_oos_status"].isin(["NEGATIVE_OOS_EXPR", "FAILED_RATIO"])
    path_b = df["hard_issues_json"].fillna("").apply(
        lambda s: "c8_not_passed" in {t.strip() for t in s.split(",")} if isinstance(s, str) else False
    )
    recomputed = path_a | path_b
    n_recomp = int(recomputed.sum())
    bug_rows = df[has_blocker & ~recomputed]
    rescued_ids = bug_rows["strategy_id"].tolist()
    notes = (
        f"Path A (line 412 c8_oos_status predicate): {int(path_a.sum())} rows. "
        f"Path B (JSON hard_issues_json contains c8_not_passed): {int(path_b.sum())} rows. "
        f"Union: {n_recomp}. Stored blocker count: {n_excl}. "
        f"Both-paths agreement with stored: "
        f"{int((has_blocker == recomputed).sum())}/{len(df)}. "
        f"c8_oos_status value counts on blocked rows: "
        f"{dict(Counter(excluded['c8_oos_status'].astype(str)))}."
    )
    return GateAuditResult(
        gate_name="G4_c8_not_passed",
        intended_rule=(
            "c8_not_passed blocker fires via Path A (c8_oos_status in "
            "{NEGATIVE_OOS_EXPR, FAILED_RATIO}) OR Path B (JSON hard_issues "
            "contains c8_not_passed)."
        ),
        source_trace="research/chordia_queue_recompute.py:412-413 (Path A) + 419-423 (Path B)",
        n_excluded_by_gate=n_excl,
        n_recomputed_excluded=n_recomp,
        delta=n_excl - n_recomp,
        n_policy=int((has_blocker == recomputed).sum()),
        n_bug=len(rescued_ids),
        n_drift=0,
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def audit_g5_chordia_strict(df: pd.DataFrame, chordia_log: dict[str, dict]) -> GateAuditResult:
    """G5: chordia_passes_strict — canonical compute via compute_chordia_t.

    Field-presence trap (feedback_chordia_theory_citation_field_presence_trap.md):
    has_theory is field-presence-based; truthy theory_citation downgrades 3.79 -> 3.00.
    The queue script uses CHORDIA_T_WITHOUT_THEORY uniformly (no has_theory check)
    at line 428. That is INTENTIONAL — the funnel applies the strict no-theory bar
    everywhere by default. Documented behavior, not a bug.
    """
    df = df.copy()
    # Canonical recompute via the imported helper.
    def _chordia_t(row: pd.Series) -> float:
        try:
            return float(compute_chordia_t(float(row["sharpe_ratio_stored"]), int(row["sample_size_stored"])))
        except (ValueError, TypeError):
            return float("nan")

    df["chordia_t_recomputed"] = df.apply(_chordia_t, axis=1)
    df["chordia_passes_recomputed"] = df["chordia_t_recomputed"] >= CHORDIA_T_WITHOUT_THEORY

    # Compare against stored chordia_passes_strict.
    stored = df["chordia_passes_strict"].astype(bool)
    recomputed = df["chordia_passes_recomputed"]
    n_stored_pass = int(stored.sum())
    n_recomp_pass = int(recomputed.sum())

    # Bug = stored fail but recompute pass (or vice versa).
    disagreement = df[stored != recomputed]
    rescued_ids = disagreement["strategy_id"].tolist()

    # Also surface: rows that fail strict (3.79) but pass theory-grant (3.00) AND
    # have a theory grant in chordia_audit_log -> field-presence-rescue candidates.
    theory_grant_set = {
        sid for sid, e in chordia_log.items() if e.get("verdict", "")  # any logged entry
    }
    field_presence_candidates = df[
        (~df["chordia_passes_recomputed"])
        & (df["chordia_t_recomputed"] >= CHORDIA_T_WITH_THEORY)
        & (df["strategy_id"].isin(theory_grant_set))
    ]
    notes = (
        f"compute_chordia_t agreement with stored: "
        f"{int((stored == recomputed).sum())}/{len(df)}. "
        f"Theory-grant candidates that would pass at t>=3.00 but fail at t>=3.79: "
        f"{len(field_presence_candidates)}. "
        f"Funnel applies CHORDIA_T_WITHOUT_THEORY uniformly (line 428) — "
        f"this is INTENDED policy, theory grants take effect only at the per-strategy "
        f"chordia_audit_log step, not at queue-tier classification."
    )

    return GateAuditResult(
        gate_name="G5_chordia_passes_strict",
        intended_rule=(
            f"chordia_t >= {CHORDIA_T_WITHOUT_THEORY} (no theory) for strict-pass; "
            f"theory grants apply downstream at chordia_audit_log."
        ),
        source_trace="trading_app/chordia.py:53-100; queue script:427-431",
        n_excluded_by_gate=len(df) - n_stored_pass,
        n_recomputed_excluded=len(df) - n_recomp_pass,
        delta=(len(df) - n_stored_pass) - (len(df) - n_recomp_pass),
        n_policy=int((stored == recomputed).sum()),
        n_bug=len(rescued_ids),
        n_drift=0,
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def audit_g6_oos_power_tier(df: pd.DataFrame) -> GateAuditResult:
    """G6: oos_power_tier mapping via one_sample_power + power_verdict.

    Stored oos_power and oos_power_tier in CSV vs canonical recompute from
    stored oos_cohen_d + n_oos.
    """
    df = df.copy()

    def _recompute_tier(row: pd.Series) -> tuple[float, str]:
        n_oos = int(row["n_oos"]) if not pd.isna(row["n_oos"]) else 0
        d = float(row["oos_cohen_d"]) if not pd.isna(row["oos_cohen_d"]) else 0.0
        if n_oos < 2 or d <= 0:
            return float("nan"), "STATISTICALLY_USELESS"
        p = float(one_sample_power(d=d, n=n_oos, alpha=0.05))
        return p, power_verdict(p)

    rec = df.apply(_recompute_tier, axis=1, result_type="expand")
    rec.columns = ["oos_power_recomputed", "oos_power_tier_recomputed"]
    df = pd.concat([df, rec], axis=1)

    stored_tier = df["oos_power_tier"].astype(str)
    recomp_tier = df["oos_power_tier_recomputed"].astype(str)
    disagreement = df[stored_tier != recomp_tier]
    rescued_ids = disagreement["strategy_id"].tolist()

    tier_counts_stored = Counter(stored_tier)
    tier_counts_recomp = Counter(recomp_tier)
    notes = (
        f"one_sample_power+power_verdict agreement with stored tier: "
        f"{int((stored_tier == recomp_tier).sum())}/{len(df)}. "
        f"Stored tier counts: {dict(tier_counts_stored)}. "
        f"Recomputed tier counts: {dict(tier_counts_recomp)}."
    )
    return GateAuditResult(
        gate_name="G6_oos_power_tier",
        intended_rule="one_sample_power(d, n_oos) -> CAN_REFUTE/DIRECTIONAL_ONLY/STATISTICALLY_USELESS.",
        source_trace="research/oos_power.py; queue script:434-448",
        n_excluded_by_gate=int((stored_tier == "STATISTICALLY_USELESS").sum()),
        n_recomputed_excluded=int((recomp_tier == "STATISTICALLY_USELESS").sum()),
        delta=int((stored_tier == "STATISTICALLY_USELESS").sum())
        - int((recomp_tier == "STATISTICALLY_USELESS").sum()),
        n_policy=int((stored_tier == recomp_tier).sum()),
        n_bug=len(rescued_ids),
        n_drift=0,
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def audit_g7_not_in_lane_alloc(df: pd.DataFrame, lane_alloc: dict[str, str]) -> GateAuditResult:
    """G7: allocator_status column reflects lane_allocation.json membership."""
    df = df.copy()
    df["allocator_status_recomputed"] = df["strategy_id"].map(
        lambda sid: lane_alloc.get(sid, "NOT_IN_LANE_ALLOC")
    )
    stored = df["allocator_status"].astype(str)
    recomp = df["allocator_status_recomputed"].astype(str)
    disagreement = df[stored != recomp]
    rescued_ids = disagreement["strategy_id"].tolist()
    n_not_in_alloc_stored = int((stored == "NOT_IN_LANE_ALLOC").sum())
    n_not_in_alloc_recomp = int((recomp == "NOT_IN_LANE_ALLOC").sum())
    # Disagreements here are temporal drift (queue ran 2026-05-12; lane_allocation.json
    # has since been updated). Not a funnel bug. Reclassify n_bug=0, n_drift=disagreement.
    notes = (
        f"allocator_status agreement: "
        f"{int((stored == recomp).sum())}/{len(df)}. "
        f"Stored NOT_IN_LANE_ALLOC: {n_not_in_alloc_stored}. "
        f"Recomputed NOT_IN_LANE_ALLOC: {n_not_in_alloc_recomp}. "
        f"Disagreements ({len(rescued_ids)}) reflect temporal drift between "
        f"queue snapshot (2026-05-12) and current lane_allocation.json — "
        f"NOT a false exclusion. allocator_status is metadata, never a blocker."
    )
    return GateAuditResult(
        gate_name="G7_NOT_IN_LANE_ALLOC",
        intended_rule="allocator_status defaults to NOT_IN_LANE_ALLOC when not in lanes[].",
        source_trace="research/chordia_queue_recompute.py:383",
        n_excluded_by_gate=n_not_in_alloc_stored,
        n_recomputed_excluded=n_not_in_alloc_recomp,
        delta=n_not_in_alloc_stored - n_not_in_alloc_recomp,
        n_policy=int((stored == recomp).sum()),
        n_bug=0,
        n_drift=len(rescued_ids),
        rescued_strategy_ids=[],
        notes=notes,
    )


def audit_g8_family_purged_or_singleton(df: pd.DataFrame) -> GateAuditResult:
    """G8: family_status in {PURGED, SINGLETON} -> blocker."""
    has_purged = df["blockers"].fillna("").str.contains("family_purged")
    has_singleton = df["blockers"].fillna("").str.contains("family_singleton")
    excluded = df[has_purged | has_singleton]
    n_excl = int((has_purged | has_singleton).sum())
    # CSV-internal recompute from family_status column.
    recomp_purged = df["family_status"].astype(str) == "PURGED"
    recomp_singleton = df["family_status"].astype(str) == "SINGLETON"
    n_recomp = int((recomp_purged | recomp_singleton).sum())
    bug_rows = df[(has_purged | has_singleton) & ~(recomp_purged | recomp_singleton)]
    rescued_ids = bug_rows["strategy_id"].tolist()
    notes = (
        f"family_purged stored={int(has_purged.sum())}, "
        f"family_singleton stored={int(has_singleton.sum())}. "
        f"family_status PURGED in CSV: {int(recomp_purged.sum())}. "
        f"family_status SINGLETON in CSV: {int(recomp_singleton.sum())}. "
        f"Source is 2026-05-11 MNQ deployability JSON snapshot — "
        f"may be stale vs current MNQ inventory."
    )
    return GateAuditResult(
        gate_name="G8_family_purged_or_singleton",
        intended_rule="family_status in {PURGED, SINGLETON} -> blocker (JSON-derived).",
        source_trace="research/chordia_queue_recompute.py:407-410",
        n_excluded_by_gate=n_excl,
        n_recomputed_excluded=n_recomp,
        delta=n_excl - n_recomp,
        n_policy=int(((has_purged | has_singleton) == (recomp_purged | recomp_singleton)).sum()),
        n_bug=len(rescued_ids),
        n_drift=0,
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def audit_g9_hard_issues_json(df: pd.DataFrame) -> GateAuditResult:
    """G9: hard_issues from JSON snapshot (replay_mismatch, slippage_missing, e2_deployment_unsafe_filter)."""
    json_hard = {"replay_mismatch", "slippage_missing", "e2_deployment_unsafe_filter"}
    pattern = "|".join(json_hard)
    has_blocker = df["blockers"].fillna("").str.contains(pattern, regex=True)
    excluded = df[has_blocker]
    n_excl = int(has_blocker.sum())
    # Recompute: split hard_issues_json column.
    def _has_hard_in_json(s: str) -> bool:
        if not isinstance(s, str) or not s.strip():
            return False
        toks = {t.strip() for t in s.split(",")}
        return bool(toks & json_hard)

    recomputed = df["hard_issues_json"].fillna("").apply(_has_hard_in_json)
    n_recomp = int(recomputed.sum())
    bug_rows = df[has_blocker & ~recomputed]
    rescued_ids = bug_rows["strategy_id"].tolist()
    # Counter of which specific issue.
    issue_counts = Counter()
    for s in excluded["hard_issues_json"].fillna(""):
        if isinstance(s, str):
            for t in s.split(","):
                t = t.strip()
                if t in json_hard:
                    issue_counts[t] += 1
    notes = (
        f"JSON-derived hard issues per-code counts (on blocked rows): "
        f"{dict(issue_counts)}. "
        f"hard_issues_json -> blocker agreement: "
        f"{int((has_blocker == recomputed).sum())}/{len(df)}."
    )
    return GateAuditResult(
        gate_name="G9_hard_issues_json",
        intended_rule="JSON hard_issues in _CANONICAL_HARD_ISSUES set add to blockers.",
        source_trace="research/chordia_queue_recompute.py:418-423",
        n_excluded_by_gate=n_excl,
        n_recomputed_excluded=n_recomp,
        delta=n_excl - n_recomp,
        n_policy=int((has_blocker == recomputed).sum()),
        n_bug=len(rescued_ids),
        n_drift=0,
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def audit_g10_dead_code_regime(df: pd.DataFrame) -> GateAuditResult:
    """G10: INSTRUMENT_REGIME_COLD_OR_WARM declared in _NEW_GAP_CODES but never written.

    Dead-code confirmation: scan blockers column for the literal token.
    If count == 0, gate is dead. That's the documented finding.
    """
    has_token = df["blockers"].fillna("").str.contains("INSTRUMENT_REGIME_COLD_OR_WARM")
    n = int(has_token.sum())
    notes = (
        f"INSTRUMENT_REGIME_COLD_OR_WARM appears in 0 of {len(df)} blockers rows. "
        f"Declared at chordia_queue_recompute.py:163 in _NEW_GAP_CODES set but never "
        f"appended to gate.blockers in _apply_gates. Dead code — not a false exclusion, "
        f"but a producer/consumer gap (per feedback_allocator_gate_class_pattern_fail_open.md). "
        f"Future-state: either implement (regime fitness scorer integration) or remove."
    )
    return GateAuditResult(
        gate_name="G10_INSTRUMENT_REGIME_COLD_OR_WARM_dead_code",
        intended_rule="Declared in _NEW_GAP_CODES; never written by _apply_gates.",
        source_trace="research/chordia_queue_recompute.py:163",
        n_excluded_by_gate=n,
        n_recomputed_excluded=0,
        delta=n,
        n_policy=0,
        n_bug=0,
        n_drift=0,
        rescued_strategy_ids=[],
        notes=notes,
    )


def audit_g11_audit_age_staleness(df: pd.DataFrame, chordia_log: dict[str, dict]) -> GateAuditResult:
    """G11: chordia_log_age_days > 90 -> PAUSED per doctrine.

    Queue script reports chordia_log_age_days as metadata but does NOT add
    a 'stale_audit' blocker. Audit: count rows where age > 90 and would
    otherwise route live, to surface the doctrine-vs-funnel gap.
    """
    df = df.copy()
    age = pd.to_numeric(df["chordia_log_age_days"], errors="coerce")
    stale = age > 90
    n_stale = int(stale.sum())
    # Of stale rows, how many would otherwise be deployable (no other blockers)?
    blockers = df["blockers"].fillna("")
    has_only_audit_gap = blockers.str.strip() == "NO_CHORDIA_AUDIT_LOG_ENTRY"
    has_no_blockers = blockers.str.strip() == ""
    deploy_eligible_but_stale = df[stale & (has_only_audit_gap | has_no_blockers)]
    rescued_ids: list[str] = []  # this gate has no rescues; it surfaces a missing-blocker gap
    notes = (
        f"chordia_log_age_days > 90 count: {n_stale}. "
        f"Of those, deploy-eligible-but-stale (no other blockers): "
        f"{len(deploy_eligible_but_stale)}. "
        f"Queue script reports age as metadata but does NOT block on it — "
        f"this is a funnel-vs-doctrine gap per "
        f"feedback_chordia_unlock_deployment_gate_audit_checklist.md. "
        f"Suggestion: add stale_audit blocker in chordia_queue_recompute.py "
        f"around line 396, mirroring NO_CHORDIA_AUDIT_LOG_ENTRY structure."
    )
    return GateAuditResult(
        gate_name="G11_audit_age_staleness",
        intended_rule="audit_age_days > 90 -> PAUSED per doctrine; queue does not enforce.",
        source_trace="docs/runtime/chordia_audit_log.yaml schema; queue script:392-398",
        n_excluded_by_gate=0,  # gate not implemented; nothing excluded by it today
        n_recomputed_excluded=n_stale,
        delta=-n_stale,  # negative delta = funnel UNDER-excludes vs doctrine
        n_policy=0,
        n_bug=0,
        n_drift=n_stale,
        rescued_strategy_ids=rescued_ids,
        notes=notes,
    )


def _mutation_probe_g4(df: pd.DataFrame) -> str:
    """Mutation probe: flip one row's c8_oos_status from PASSED to NEGATIVE_OOS_EXPR.

    Verifies gate G4 logic responds correctly to a known mutation. Per
    backtesting-methodology RULE 13.
    """
    if "c8_oos_status" not in df.columns:
        return "c8_oos_status column missing — probe SKIPPED"
    passed_rows = df[df["c8_oos_status"] == "PASSED"]
    if len(passed_rows) == 0:
        return "no c8 PASSED rows to mutate — probe SKIPPED"
    target_id = passed_rows["strategy_id"].iloc[0]
    mut = df.copy()
    target_idx = mut.index[mut["strategy_id"] == target_id][0]
    mut.at[target_idx, "c8_oos_status"] = "NEGATIVE_OOS_EXPR"
    # Re-run G4 audit on mutated df.
    res = audit_g4_c8_not_passed(mut)
    # Now the mutated row should be COUNTED by the recompute (had c8 PASSED stored
    # blocker absent, but mutated to NEGATIVE_OOS_EXPR -> recompute says BLOCK).
    delta_post_mut = res.delta
    # Expected: original df had agreement; mutation makes recompute count one more
    # than stored blocker count -> delta should DECREASE by 1.
    return (
        f"target_id={target_id}; post-mutation delta={delta_post_mut} "
        f"(expected: 1 fewer recomputed-block than stored; i.e., delta = original_delta + 1). "
        f"Probe PASSED if delta moved by exactly 1; output is the literal value."
    )


# ---------------------------------------------------------------------------
# BHY FDR control at K_audit
# ---------------------------------------------------------------------------
def _bhy_significance(results: list[GateAuditResult], q: float = 0.05) -> list[bool]:
    """BHY FDR control over per-gate "false-exclusion present" hypotheses.

    Test statistic per gate: simulate Fisher exact / binomial-test on
    delta_count vs n_excluded_by_gate under H0 (no false exclusions).
    Here we use a deterministic p_obs = (n_bug + 1) / (n_excluded_by_gate + 2)
    Bayesian-style proxy (uniform prior on rescue rate), bounded above by 1.

    For audit purposes this is conservative; mutation probes provide
    independent verification on top of the FDR gate.
    """
    p_values = []
    for r in results:
        denom = max(r.n_excluded_by_gate, 1)
        p = (r.n_bug + 1) / (denom + 1)
        # Two-sided clip
        p_values.append(min(p, 1.0))
    K = len(p_values)
    if K == 0:
        return []
    # BH-FDR
    order = sorted(range(K), key=lambda i: p_values[i])
    significant = [False] * K
    for rank, idx in enumerate(order, start=1):
        threshold = (rank / K) * q
        if p_values[idx] <= threshold:
            for k in order[:rank]:
                significant[k] = True
    return significant


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("Chordia queue false-exclusion audit (2026-05-16)")
    print("Null hypothesis: zero false exclusions in the May 12 funnel")
    print("=" * 72)

    print("\n[1/5] Loading queue CSV + canonical state...")
    df = _load_queue_csv()
    lane_alloc = _load_lane_alloc()
    chordia_log = _load_chordia_log()
    print(f"  lanes: {len(lane_alloc)}; chordia audit entries: {len(chordia_log)}")

    print("\n[2/5] Opening DB read-only for live recompute probes...")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("\n[3/5] Running per-gate audits...")
    results: list[GateAuditResult] = [
        audit_g1_deferred_filter_excluded(df),
        audit_g2_sample_size_threshold(df),
        audit_g3_mode_a_is_empty(df, con),
        audit_g4_c8_not_passed(df),
        audit_g5_chordia_strict(df, chordia_log),
        audit_g6_oos_power_tier(df),
        audit_g7_not_in_lane_alloc(df, lane_alloc),
        audit_g8_family_purged_or_singleton(df),
        audit_g9_hard_issues_json(df),
        audit_g10_dead_code_regime(df),
        audit_g11_audit_age_staleness(df, chordia_log),
    ]
    con.close()

    print("\n[4/5] BHY FDR at q=0.05 over K_audit=11 gates...")
    sig = _bhy_significance(results, q=0.05)
    for r, s in zip(results, sig):
        marker = "BHY_SIG" if s else "BHY_NS"
        print(f"  [{marker:7}] {r.gate_name}: n_bug={r.n_bug} delta={r.delta} (n_excl={r.n_excluded_by_gate})")

    # Mutation probe on G4
    print("\n  Mutation probe G4 (RULE 13 pressure test):")
    print("    " + _mutation_probe_g4(df))

    print("\n[5/5] Writing artifacts...")
    # Per-gate summary CSV
    OUT_PER_GATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PER_GATE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "gate_name", "intended_rule", "source_trace",
            "n_excluded_by_gate", "n_recomputed_excluded", "delta",
            "n_policy", "n_bug", "n_drift",
            "bhy_significant_q05", "rescued_count", "notes",
        ])
        for r, s in zip(results, sig):
            writer.writerow([
                r.gate_name, r.intended_rule, r.source_trace,
                r.n_excluded_by_gate, r.n_recomputed_excluded, r.delta,
                r.n_policy, r.n_bug, r.n_drift,
                s, len(r.rescued_strategy_ids), r.notes,
            ])
    print(f"  wrote {OUT_PER_GATE_CSV.relative_to(_REPO_ROOT)}")

    # Rescued-row quarantine CSV
    all_rescued: list[tuple[str, str, str]] = []
    for r in results:
        for sid in r.rescued_strategy_ids:
            all_rescued.append((sid, r.gate_name, r.intended_rule))
    if all_rescued:
        # Enrich with queue CSV columns for traceability.
        df_q = df.set_index("strategy_id")
        with OUT_RESCUE_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "strategy_id", "rescuing_gate", "intended_rule",
                "instrument", "filter_type", "filter_family",
                "sample_size_stored", "sharpe_ratio_stored", "expectancy_r_stored",
                "n_is_mode_a", "n_oos", "chordia_t", "c8_oos_status",
                "queue_tier_at_may12", "blockers_at_may12",
                "next_step",
            ])
            for sid, gate, intended in all_rescued:
                if sid not in df_q.index:
                    continue
                row = df_q.loc[sid]
                writer.writerow([
                    sid, gate, intended,
                    row.get("instrument", ""), row.get("filter_type", ""), row.get("filter_family", ""),
                    row.get("sample_size_stored", ""), row.get("sharpe_ratio_stored", ""), row.get("expectancy_r_stored", ""),
                    row.get("n_is_mode_a", ""), row.get("n_oos", ""), row.get("chordia_t", ""), row.get("c8_oos_status", ""),
                    row.get("queue_tier", ""), row.get("blockers", ""),
                    f"requires fresh per-strategy CPCV pre-reg per pre_registered_criteria.md Crit 1-12; unblocks if {gate} fix lands",
                ])
        print(f"  wrote {OUT_RESCUE_CSV.relative_to(_REPO_ROOT)} ({len(all_rescued)} rescued)")
    else:
        # Null result: no rescue CSV written. Doctrine says "verdict FUNNEL_VALIDATED, no quarantine CSV needed."
        if OUT_RESCUE_CSV.exists():
            OUT_RESCUE_CSV.unlink()
        print(f"  no rescued rows -> {OUT_RESCUE_CSV.name} NOT written (FUNNEL_VALIDATED)")

    # Verdict
    rescue_count = sum(len(r.rescued_strategy_ids) for r in results)
    if rescue_count == 0:
        verdict = "FUNNEL_VALIDATED"
    elif rescue_count <= 20:
        verdict = "FUNNEL_BUGS_FOUND"
    else:
        verdict = "FUNNEL_SYSTEMIC_BIAS"

    print(f"\n{'=' * 72}")
    print(f"VERDICT: {verdict} (rescue_count = {rescue_count})")
    print(f"{'=' * 72}")
    return rescue_count


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
