"""
FDR / Multiple Testing Integrity Audit — Real Data Validation
==============================================================
Purpose: End-to-end audit of BH FDR implementation correctness.
Validates claims from the Mar 2026 pipeline-wide FDR audit against live DB.

Run:  python scripts/tools/audit_fdr_integrity.py [--db-path gold.db] [--verbose]

Schema notes (verified against live DB):
  - validated_setups.family_hash = prefixed hash (e.g. "MGC_5m_<md5>")
  - experimental_strategies.trade_day_hash = raw MD5
  - validated_setups.p_value is NULL; raw p lives in experimental_strategies.p_value
  - fdr_adjusted_p and fdr_significant are populated in validated_setups
  - No 'direction' column — direction is implicit in the ORB break
  - orb_outcomes uses 'symbol' not 'instrument'

Checks:
  1. BH algorithm correctness (recompute from raw p-values, compare stored adj_p)
  2. K computation audit (stored discovery_k vs current canonical count)
  3. K uniformity (same K within each session)
  4. Ghost strategy detection (NULL fdr_adjusted_p in active validated)
  5. FDR gate consistency (no strategy with adj_p >= 0.05 still active)
  6. Trade-day-hash deduplication (family_hash uniqueness properties)
  7. Parameter redundancy (how many variants per unique stream)
  8. Independent bets via Jaccard similarity matrix
  9. Session-stratified K vs global K impact analysis
  10. Cross-instrument K analysis
  11. Stop multiplier independence (S0.75 vs S1.0 same entries?)
  12. RR target independence (same entries, different take-profit?)
  13. Honest counts summary

Exit codes: 0=PASS, 1=FAIL (correctness issue), 2=WARN (honest reporting issue)
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402

# Fail-closed: refuse to run if no active instruments (would produce garbled SQL)
if not ACTIVE_ORB_INSTRUMENTS:
    print("FATAL: ACTIVE_ORB_INSTRUMENTS is empty — cannot run FDR audit", file=sys.stderr)
    sys.exit(1)

# SQL fragment for filtering to active instruments only (matches validator behavior)
_ACTIVE_IN = ", ".join(f"'{i}'" for i in ACTIVE_ORB_INSTRUMENTS)


def parse_args():
    p = argparse.ArgumentParser(description="FDR Integrity Audit")
    p.add_argument("--db-path", type=str, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


# ---------------------------------------------------------------------------
# BH FDR reference implementation (independent of strategy_validator.py)
# ---------------------------------------------------------------------------


def bh_fdr_reference(
    p_values: list[tuple[str, float]], alpha: float = 0.05, total_tests: int | None = None
) -> dict[str, dict]:
    """Independent BH FDR implementation for cross-validation."""
    valid = [(sid, p) for sid, p in p_values if p is not None and not np.isnan(p)]
    if not valid:
        return {}
    valid.sort(key=lambda x: x[1])
    m = total_tests if total_tests is not None else len(valid)
    if m < len(valid):
        raise ValueError(f"total_tests ({m}) < n valid ({len(valid)})")

    n = len(valid)
    adjusted = [0.0] * n
    for i in range(n - 1, -1, -1):
        rank = i + 1
        raw = valid[i][1] * m / rank
        raw = min(raw, 1.0)
        if i < n - 1:
            raw = min(raw, adjusted[i + 1])
        adjusted[i] = raw

    result = {}
    for i, (sid, raw_p) in enumerate(valid):
        adj_rounded = round(adjusted[i], 6)  # match production rounding (strategy_validator.py:129)
        result[sid] = {
            "raw_p": raw_p,
            "adjusted_p": adj_rounded,
            "fdr_significant": adj_rounded < alpha,
            "fdr_rank": i + 1,
        }
    return result


# ---------------------------------------------------------------------------
# Audit checks
# ---------------------------------------------------------------------------


class AuditResult:
    def __init__(self):
        self.checks: list[dict] = []
        self.worst = "PASS"

    def add(self, name: str, status: str, detail: str, data: dict | None = None):
        self.checks.append({"name": name, "status": status, "detail": detail, "data": data or {}})
        if status == "FAIL":
            self.worst = "FAIL"
        elif status == "WARN" and self.worst != "FAIL":
            self.worst = "WARN"

    def summary(self):
        lines = ["\n" + "=" * 80, "FDR INTEGRITY AUDIT — RESULTS", "=" * 80]
        for c in self.checks:
            icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(c["status"], "??")
            lines.append(f"  [{icon}] {c['status']:4s}  {c['name']}")
            if c["detail"]:
                for dl in c["detail"].split("\n"):
                    lines.append(f"           {dl}")
        lines.append("-" * 80)
        lines.append(f"  OVERALL: {self.worst}")
        lines.append("=" * 80)
        return "\n".join(lines)


def check_1_bh_correctness(con, alpha, verbose, audit: AuditResult):
    """Recompute BH FDR using the FULL canonical pool per session and compare
    to stored fdr_adjusted_p in validated_setups.

    Key insight: BH was applied to ALL canonical strategies per session (not just
    the survivors). The stored discovery_k was the pool size at validation time.
    Since the pool has grown, we can't exactly reconstruct the old pool. Instead:

    A) Recompute BH on the CURRENT full pool with CURRENT K → shows current-state truth
    B) Algebraic spot-check: verify stored adj_p is plausible given raw_p and discovery_k
    """
    # --- Part A: Current-state recomputation ---
    # Get all sessions with validated strategies
    sessions = con.execute("""
        SELECT DISTINCT orb_label FROM validated_setups WHERE status = 'active'
    """).fetchall()

    lines = []
    total_validated = 0
    current_would_fail = []

    for (session,) in sessions:
        # Get FULL canonical pool for this session — ACTIVE instruments only
        # (matches validator behavior at strategy_validator.py:1255)
        full_pool = con.execute(
            f"""
            SELECT strategy_id, p_value
            FROM experimental_strategies
            WHERE is_canonical = TRUE AND orb_label = ? AND p_value IS NOT NULL
            AND instrument IN ({_ACTIVE_IN})
        """,
            [session],
        ).fetchall()

        current_k = len(full_pool)
        if current_k == 0:
            continue

        p_pairs = [(r[0], r[1]) for r in full_pool]
        ref = bh_fdr_reference(p_pairs, alpha=alpha, total_tests=current_k)

        # Check each validated strategy against current-state BH
        validated = con.execute(
            """
            SELECT v.strategy_id, v.fdr_adjusted_p, v.fdr_significant, v.discovery_k
            FROM validated_setups v
            WHERE v.status = 'active' AND v.orb_label = ?
        """,
            [session],
        ).fetchall()

        for sid, stored_adj, stored_sig, stored_k in validated:
            total_validated += 1
            if sid in ref:
                current_adj = ref[sid]["adjusted_p"]
                current_sig = ref[sid]["fdr_significant"]
                if not current_sig and stored_sig:
                    current_would_fail.append(
                        f"  {sid}: stored adj_p={stored_adj:.6f} (K={stored_k}) → "
                        f"current adj_p={current_adj:.6f} (K={current_k}) — WOULD FAIL NOW"
                    )

    lines.append(f"Part A — Current-state recomputation across {len(sessions)} sessions:")
    lines.append(f"  {total_validated} validated strategies checked against current full pools")
    if current_would_fail:
        lines.append(f"  {len(current_would_fail)} would FAIL under current K (pool has grown):")
        lines.extend(current_would_fail[:15])
    else:
        lines.append("  All validated strategies still pass BH under current (larger) K")

    # --- Part B: Algebraic plausibility check ---
    # For each validated strategy, verify: adj_p ≈ raw_p * K / rank
    # We can't know the exact rank, but we can verify adj_p is in valid range
    algebraic_issues = []
    rows = con.execute("""
        SELECT v.strategy_id, v.orb_label, e.p_value, v.fdr_adjusted_p,
               v.fdr_significant, v.discovery_k
        FROM validated_setups v
        JOIN experimental_strategies e ON v.strategy_id = e.strategy_id
        WHERE v.status = 'active' AND e.p_value IS NOT NULL AND v.fdr_adjusted_p IS NOT NULL
    """).fetchall()

    for sid, _orb, raw_p, adj_p, sig, dk in rows:
        # adj_p should be >= raw_p (BH can only inflate)
        if adj_p < raw_p - 1e-10:
            algebraic_issues.append(f"{sid}: adj_p={adj_p:.8f} < raw_p={raw_p:.8f} — IMPOSSIBLE")
        # adj_p should be <= 1.0
        if adj_p > 1.0 + 1e-10:
            algebraic_issues.append(f"{sid}: adj_p={adj_p:.8f} > 1.0 — IMPOSSIBLE")
        # If significant, adj_p must be < alpha
        if sig and adj_p >= alpha:
            algebraic_issues.append(f"{sid}: sig=TRUE but adj_p={adj_p:.6f} >= {alpha}")
        # If not significant, adj_p must be >= alpha
        if not sig and adj_p < alpha:
            algebraic_issues.append(f"{sid}: sig=FALSE but adj_p={adj_p:.6f} < {alpha}")
        # Implied rank: rank = raw_p * K / adj_p (if adj_p > 0)
        if adj_p > 0 and dk:
            implied_rank = raw_p * dk / adj_p
            if implied_rank < 0.5 or implied_rank > dk + 0.5:
                algebraic_issues.append(f"{sid}: implied rank={implied_rank:.1f} outside [1, {dk}]")

    lines.append(f"\nPart B — Algebraic plausibility ({len(rows)} strategies):")
    if algebraic_issues:
        lines.append(f"  {len(algebraic_issues)} issues found:")
        lines.extend(f"  {i}" for i in algebraic_issues[:15])
    else:
        lines.append("  All adj_p values are algebraically consistent (adj_p >= raw_p, rank in [1,K])")

    # Determine overall status
    has_impossible = any("IMPOSSIBLE" in i for i in algebraic_issues)
    if has_impossible:
        audit.add("1. BH Correctness", "FAIL", "\n".join(lines))
    elif current_would_fail:
        audit.add(
            "1. BH Correctness",
            "WARN",
            "\n".join(lines) + "\n\nNote: K drift means current BH is stricter. "
            "Re-validation with current K recommended.",
        )
    else:
        audit.add("1. BH Correctness", "PASS", "\n".join(lines))


def check_2_k_drift(con, verbose, audit: AuditResult):
    """Compare stored discovery_k to current canonical strategy count per session."""
    # Active instruments only — matches validator at strategy_validator.py:1255
    current_k = con.execute(f"""
        SELECT orb_label, COUNT(*) as cnt
        FROM experimental_strategies
        WHERE is_canonical = TRUE
        AND instrument IN ({_ACTIVE_IN})
        GROUP BY orb_label
    """).fetchall()
    current_map = {r[0]: r[1] for r in current_k}

    stored_k = con.execute("""
        SELECT DISTINCT orb_label, discovery_k
        FROM validated_setups
        WHERE status = 'active' AND discovery_k IS NOT NULL
    """).fetchall()

    drifts = []
    for orb, dk in stored_k:
        cur = current_map.get(orb)
        if cur is None:
            drifts.append(f"{orb}: stored K={dk}, no current canonical strategies found")
        elif dk != cur:
            direction = "CONSERVATIVE (old K < current)" if dk < cur else "LIBERAL (old K > current — DANGER)"
            pct = abs(cur - dk) / dk * 100
            drifts.append(f"{orb}: stored K={dk}, current={cur}, delta={cur - dk} ({pct:.1f}%) — {direction}")

    if not drifts:
        audit.add("2. K Drift", "PASS", f"All {len(stored_k)} session K values match current canonical counts")
    else:
        has_liberal = any("LIBERAL" in d for d in drifts)
        status = "FAIL" if has_liberal else "WARN"
        note = (
            "\n\nNote: discovery_k is frozen on first write (2026-03-30). CONSERVATIVE drift "
            "is expected as the canonical pool grows. Only LIBERAL drift (old K > current) is dangerous."
        )
        audit.add("2. K Drift", status, f"{len(drifts)} sessions drifted:\n" + "\n".join(drifts) + note)


def check_3_k_uniformity(con, verbose, audit: AuditResult):
    """Within each session, all strategies should have the same discovery_k."""
    rows = con.execute("""
        SELECT orb_label, COUNT(DISTINCT discovery_k) as n_k,
               MIN(discovery_k) as min_k, MAX(discovery_k) as max_k,
               COUNT(*) as n_strats
        FROM validated_setups
        WHERE status = 'active' AND discovery_k IS NOT NULL
        GROUP BY orb_label
        HAVING COUNT(DISTINCT discovery_k) > 1
    """).fetchall()

    if not rows:
        total = con.execute("""
            SELECT COUNT(DISTINCT orb_label) FROM validated_setups
            WHERE status = 'active' AND discovery_k IS NOT NULL
        """).fetchone()[0]
        audit.add("3. K Uniformity", "PASS", f"All {total} sessions have uniform discovery_k")
    else:
        detail = "\n".join(f"{r[0]}: {r[1]} distinct K values (range {r[2]}-{r[3]}, {r[4]} strats)" for r in rows)
        note = (
            "\n\nNote: Post-freeze (2026-03-30), strategies validated in different runs will "
            "have different discovery_k. This is expected and correct — each strategy preserves "
            "the K under which it was originally promoted."
        )
        audit.add("3. K Uniformity", "WARN", f"K not uniform in {len(rows)} sessions:\n{detail}" + note)


def check_4_ghost_strategies(con, verbose, audit: AuditResult):
    """Check for NULL values in FDR columns of active validated strategies."""
    null_adj = con.execute("""
        SELECT COUNT(*) FROM validated_setups
        WHERE status = 'active' AND fdr_adjusted_p IS NULL
    """).fetchone()[0]

    null_sig = con.execute("""
        SELECT COUNT(*) FROM validated_setups
        WHERE status = 'active' AND fdr_significant IS NULL
    """).fetchone()[0]

    null_dk = con.execute("""
        SELECT COUNT(*) FROM validated_setups
        WHERE status = 'active' AND discovery_k IS NULL
    """).fetchone()[0]

    # Also check: validated strategies missing from experimental_strategies
    orphans = con.execute("""
        SELECT COUNT(*) FROM validated_setups v
        LEFT JOIN experimental_strategies e ON v.strategy_id = e.strategy_id
        WHERE v.status = 'active' AND e.strategy_id IS NULL
    """).fetchone()[0]

    # Check p_value availability in experimental for validated strategies
    no_p = con.execute("""
        SELECT COUNT(*) FROM validated_setups v
        JOIN experimental_strategies e ON v.strategy_id = e.strategy_id
        WHERE v.status = 'active' AND e.p_value IS NULL
    """).fetchone()[0]

    issues = []
    if null_adj > 0:
        issues.append(f"{null_adj} active strategies with NULL fdr_adjusted_p")
    if null_sig > 0:
        issues.append(f"{null_sig} active strategies with NULL fdr_significant")
    if null_dk > 0:
        issues.append(f"{null_dk} active strategies with NULL discovery_k")
    if orphans > 0:
        issues.append(f"{orphans} validated strategies missing from experimental_strategies")
    if no_p > 0:
        issues.append(f"{no_p} validated strategies with NULL p_value in experimental_strategies")

    total = con.execute("SELECT COUNT(*) FROM validated_setups WHERE status = 'active'").fetchone()[0]

    if not issues:
        audit.add(
            "4. Ghost Strategies",
            "PASS",
            f"All {total} active strategies have complete FDR columns and experimental_strategies linkage",
        )
    else:
        audit.add("4. Ghost Strategies", "FAIL", "\n".join(issues))


def check_5_fdr_gate_consistency(con, alpha, verbose, audit: AuditResult):
    """No active strategy should have fdr_adjusted_p >= alpha or fdr_significant=FALSE."""
    leakers = con.execute(f"""
        SELECT strategy_id, orb_label, fdr_adjusted_p
        FROM validated_setups
        WHERE status = 'active' AND fdr_adjusted_p >= {alpha}
    """).fetchall()

    false_sig = con.execute("""
        SELECT strategy_id, orb_label, fdr_significant, fdr_adjusted_p
        FROM validated_setups
        WHERE status = 'active' AND fdr_significant = FALSE
    """).fetchall()

    issues = []
    if leakers:
        issues.append(f"{len(leakers)} strategies with adj_p >= {alpha} still active:")
        for s in leakers[:10]:
            issues.append(f"  {s[0]} ({s[1]}): adj_p={s[2]:.6f}")
    if false_sig:
        issues.append(f"{len(false_sig)} strategies with fdr_significant=FALSE still active:")
        for s in false_sig[:10]:
            issues.append(f"  {s[0]} ({s[1]}): sig={s[2]}, adj_p={s[3]:.6f}")

    if not issues:
        audit.add("5. FDR Gate Consistency", "PASS", "No active strategies leak past FDR gate")
    else:
        audit.add("5. FDR Gate Consistency", "FAIL", "\n".join(issues))


def check_6_trade_day_hash_accuracy(con, verbose, audit: AuditResult):
    """Verify trade_day_hash properties in experimental_strategies."""
    # Check: same hash should NOT span different (entry_model, orb_label) combos
    # Active instruments only — matches validator scope
    cross_signal = con.execute(f"""
        SELECT trade_day_hash,
               COUNT(DISTINCT strategy_id) as n,
               COUNT(DISTINCT entry_model) as n_em,
               COUNT(DISTINCT orb_label) as n_sess
        FROM experimental_strategies
        WHERE is_canonical = TRUE
        AND instrument IN ({_ACTIVE_IN})
        GROUP BY trade_day_hash
        HAVING COUNT(DISTINCT entry_model) > 1 OR COUNT(DISTINCT orb_label) > 1
    """).fetchall()

    # Check: family_hash in validated should map to trade_day_hash in experimental
    hash_link = con.execute("""
        SELECT COUNT(*),
               COUNT(CASE WHEN e.trade_day_hash IS NOT NULL THEN 1 END)
        FROM validated_setups v
        JOIN experimental_strategies e ON v.strategy_id = e.strategy_id
        WHERE v.status = 'active'
    """).fetchone()

    lines = []
    if cross_signal:
        lines.append(
            f"{len(cross_signal)} hashes shared across different entry_model/session combos — potential collision"
        )
        for h in cross_signal[:5]:
            lines.append(f"  hash {h[0][:16]}...: {h[1]} strats, {h[2]} entry_models, {h[3]} sessions")
        status = "WARN"
    else:
        lines.append("Trade-day hashes are unique within (entry_model, session) — no cross-signal collisions")
        status = "PASS"

    lines.append(
        f"Hash linkage: {hash_link[0]} validated strategies, {hash_link[1]} have trade_day_hash in experimental"
    )

    audit.add("6. Trade-Day-Hash Accuracy", status, "\n".join(lines))


def check_7_parameter_redundancy(con, verbose, audit: AuditResult):
    """Quantify how many validated strategies are parameter variants of the same signal."""
    # Use family_hash from validated_setups (the correct column)
    rows = con.execute("""
        SELECT v.instrument,
               COUNT(*) as total,
               COUNT(DISTINCT v.family_hash) as unique_families,
               COUNT(DISTINCT e.trade_day_hash) as unique_hashes
        FROM validated_setups v
        JOIN experimental_strategies e ON v.strategy_id = e.strategy_id
        WHERE v.status = 'active'
        GROUP BY v.instrument
    """).fetchall()

    if not rows:
        audit.add("7. Parameter Redundancy", "WARN", "No active validated strategies")
        return

    lines = []
    for inst, total, families, hashes in rows:
        ratio = total / hashes if hashes > 0 else float("inf")
        lines.append(
            f"{inst}: {total} validated → {hashes} unique trade-day-hashes → "
            f"{families} family_hashes (avg {ratio:.1f} variants per stream)"
        )

    # Breakdown: what varies within same trade_day_hash?
    variant_analysis = con.execute("""
        SELECT e.instrument, e.trade_day_hash,
               COUNT(DISTINCT v.rr_target) as n_rr,
               COUNT(DISTINCT v.stop_multiplier) as n_stop,
               COUNT(DISTINCT v.filter_type) as n_filter,
               COUNT(*) as n_total
        FROM validated_setups v
        JOIN experimental_strategies e ON v.strategy_id = e.strategy_id
        WHERE v.status = 'active'
        GROUP BY e.instrument, e.trade_day_hash
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """).fetchall()

    if variant_analysis:
        lines.append("\nTop 10 most-duplicated trade streams:")
        for inst, tdh, n_rr, n_stop, n_filt, n_tot in variant_analysis:
            lines.append(f"  {inst} {tdh[:16]}...: {n_tot} strats ({n_rr} RR × {n_stop} stop × {n_filt} filter)")

    audit.add(
        "7. Parameter Redundancy",
        "WARN",
        "\n".join(lines),
        {"by_instrument": {r[0]: {"total": r[1], "unique_families": r[2], "unique_hashes": r[3]} for r in rows}},
    )


def check_8_independent_bets_jaccard(con, verbose, audit: AuditResult):
    """Compute Jaccard similarity between unique trade streams to find independent bets."""
    # Get unique trade-day-hashes per instrument from validated strategies
    hashes = con.execute("""
        SELECT DISTINCT v.instrument, e.trade_day_hash
        FROM validated_setups v
        JOIN experimental_strategies e ON v.strategy_id = e.strategy_id
        WHERE v.status = 'active' AND e.trade_day_hash IS NOT NULL
    """).fetchall()

    if not hashes:
        audit.add("8. Independent Bets (Jaccard)", "WARN", "No active strategies with trade_day_hash")
        return

    by_instrument: dict[str, list[str]] = defaultdict(list)
    for inst, tdh in hashes:
        by_instrument[inst].append(tdh)

    lines = []
    for inst, tdh_list in sorted(by_instrument.items()):
        if len(tdh_list) <= 1:
            lines.append(f"{inst}: {len(tdh_list)} stream — trivially independent")
            continue

        # Get trade days for each hash by finding a representative strategy
        hash_days: dict[str, set] = {}
        for tdh in tdh_list:
            rep = con.execute(
                """
                SELECT e.strategy_id, e.instrument, e.orb_label, e.entry_model,
                       e.rr_target, e.confirm_bars, e.filter_type, e.stop_multiplier,
                       e.orb_minutes
                FROM experimental_strategies e
                WHERE e.trade_day_hash = ? AND e.instrument = ? AND e.is_canonical = TRUE
                LIMIT 1
            """,
                [tdh, inst],
            ).fetchone()
            if rep is None:
                continue

            sid, inst2, orb, em, rr, cb, ft, sm, om = rep
            # orb_outcomes uses 'symbol' not 'instrument', and has no stop_multiplier
            days = con.execute(
                """
                SELECT DISTINCT o.trading_day
                FROM orb_outcomes o
                WHERE o.symbol = ?
                    AND o.orb_label = ?
                    AND o.orb_minutes = ?
                    AND o.rr_target = ?
                    AND o.confirm_bars = ?
                    AND o.entry_model = ?
            """,
                [inst2, orb, om, rr, cb, em],
            ).fetchall()
            if days:
                hash_days[tdh] = set(str(d[0]) for d in days)

        if len(hash_days) < 2:
            lines.append(f"{inst}: could only resolve {len(hash_days)}/{len(tdh_list)} streams from orb_outcomes")
            continue

        # Compute pairwise Jaccard
        tdh_keys = list(hash_days.keys())
        n = len(tdh_keys)
        jaccard_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a, b = hash_days[tdh_keys[i]], hash_days[tdh_keys[j]]
                union = len(a | b)
                if union == 0:
                    j_val = 0.0
                else:
                    j_val = len(a & b) / union
                jaccard_matrix[i][j] = j_val
                jaccard_matrix[j][i] = j_val

        # Count independent clusters (greedy: J < 0.3 = independent)
        independent = []
        used = set()
        for i in range(n):
            if i in used:
                continue
            cluster = {i}
            for j in range(i + 1, n):
                if j in used:
                    continue
                if jaccard_matrix[i][j] >= 0.3:
                    cluster.add(j)
                    used.add(j)
            independent.append(cluster)
            used.add(i)

        upper_tri = jaccard_matrix[np.triu_indices(n, 1)]
        n_high_overlap = int(np.sum(upper_tri > 0.7))

        lines.append(
            f"{inst}: {n} unique streams → {len(independent)} independent clusters (J<0.3)\n"
            f"  {n_high_overlap}/{len(upper_tri)} pairs with J>0.7 (effectively same bet)\n"
            f"  Jaccard: min={upper_tri.min():.3f}, "
            f"median={np.median(upper_tri):.3f}, "
            f"mean={upper_tri.mean():.3f}, "
            f"max={upper_tri.max():.3f}"
        )

    audit.add("8. Independent Bets (Jaccard)", "WARN", "\n".join(lines))


def check_9_session_k_vs_global(con, verbose, audit: AuditResult):
    """Compare session-stratified K to what a global K would produce."""
    session_k = con.execute(f"""
        SELECT orb_label, COUNT(*) as k
        FROM experimental_strategies
        WHERE is_canonical = TRUE
        AND instrument IN ({_ACTIVE_IN})
        GROUP BY orb_label
    """).fetchall()
    total_global = sum(r[1] for r in session_k)
    session_k_map = {r[0]: r[1] for r in session_k}

    lines = [f"Global K (all sessions, all instruments): {total_global}"]
    for orb, k in sorted(session_k, key=lambda x: -x[1]):
        lines.append(f"  {orb}: K={k} ({k / total_global * 100:.1f}%)")

    # How many strategies would FAIL under global K but pass under session K?
    # Must use FULL canonical pool (not just survivors) for correct BH ranking.
    # Then check which validated strategies survive under each K.
    casualties = []
    for orb, k_session in session_k_map.items():
        # Full canonical pool for this session (matches validator at strategy_validator.py:1250-1257)
        full_pool = con.execute(
            f"""
            SELECT strategy_id, p_value
            FROM experimental_strategies
            WHERE is_canonical = TRUE AND orb_label = ? AND p_value IS NOT NULL
            AND instrument IN ({_ACTIVE_IN})
        """,
            [orb],
        ).fetchall()
        if not full_pool:
            continue

        # Get validated strategy IDs in this session
        validated_ids = set(
            r[0]
            for r in con.execute(
                """
            SELECT strategy_id FROM validated_setups
            WHERE status = 'active' AND orb_label = ?
        """,
                [orb],
            ).fetchall()
        )
        if not validated_ids:
            continue

        p_pairs = [(s[0], s[1]) for s in full_pool]
        try:
            ref_session = bh_fdr_reference(p_pairs, alpha=0.05, total_tests=k_session)
            ref_global = bh_fdr_reference(p_pairs, alpha=0.05, total_tests=total_global)

            # Count only validated strategies that pass/fail under each K
            session_sig = sum(1 for sid in validated_ids if sid in ref_session and ref_session[sid]["fdr_significant"])
            global_sig = sum(1 for sid in validated_ids if sid in ref_global and ref_global[sid]["fdr_significant"])
            if session_sig != global_sig:
                casualties.append(
                    f"{orb}: {session_sig} pass session K={k_session}, "
                    f"only {global_sig} pass global K={total_global} "
                    f"(would lose {session_sig - global_sig})"
                )
        except ValueError:
            # k_session < len(strats) edge case
            continue

    if casualties:
        lines.append("\nCasualties under global K:")
        lines.extend(f"  {c}" for c in casualties)
    else:
        lines.append("\nNo additional casualties under global K (all session survivors also survive global)")

    audit.add("9. Session K vs Global K", "WARN", "\n".join(lines))


def check_10_cross_instrument_k(con, verbose, audit: AuditResult):
    """Check cross-instrument K composition per session.

    INTENTIONALLY includes dead instruments to show full K composition.
    The validator filters on ACTIVE_ORB_INSTRUMENTS (check_1/2/9 match this).
    This check shows what the pool WOULD be if dead instruments weren't filtered.
    """
    rows = con.execute("""
        SELECT instrument, orb_label, COUNT(*) as k
        FROM experimental_strategies
        WHERE is_canonical = TRUE
        GROUP BY instrument, orb_label
        ORDER BY orb_label, instrument
    """).fetchall()

    by_session: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for inst, orb, k in rows:
        by_session[orb].append((inst, k))

    lines = []
    multi = {s: insts for s, insts in by_session.items() if len(insts) > 1}
    if not multi:
        lines.append("No sessions tested across multiple instruments")
    else:
        lines.append(f"{len(multi)} sessions span multiple instruments:")
        for session, insts in sorted(multi.items()):
            total = sum(k for _, k in insts)
            breakdown = ", ".join(f"{i}={k}" for i, k in sorted(insts))
            lines.append(f"  {session}: combined K={total} ({breakdown})")

    audit.add("10. Cross-Instrument K", "WARN", "\n".join(lines))


def check_11_stop_multiplier_independence(con, verbose, audit: AuditResult):
    """Test whether S0.75 and S1.0 produce different trade days."""
    pairs = con.execute("""
        SELECT a.strategy_id, b.strategy_id,
               a.trade_day_hash as hash_a, b.trade_day_hash as hash_b,
               a.stop_multiplier as sm_a, b.stop_multiplier as sm_b
        FROM experimental_strategies a
        JOIN experimental_strategies b
            ON a.instrument = b.instrument
            AND a.orb_label = b.orb_label
            AND a.entry_model = b.entry_model
            AND a.rr_target = b.rr_target
            AND a.confirm_bars = b.confirm_bars
            AND a.filter_type = b.filter_type
            AND a.orb_minutes = b.orb_minutes
            AND a.stop_multiplier < b.stop_multiplier
            AND a.is_canonical = TRUE
            AND b.is_canonical = TRUE
        LIMIT 500
    """).fetchall()

    if not pairs:
        audit.add("11. Stop Multiplier Independence", "PASS", "No stop multiplier pairs found")
        return

    same_hash = sum(1 for p in pairs if p[2] == p[3])
    diff_hash = sum(1 for p in pairs if p[2] != p[3])

    if diff_hash > 0:
        detail = (
            f"Out of {len(pairs)} pairs: {same_hash} share hash, {diff_hash} differ.\n"
            f"Different hashes mean stop_multiplier affects WHICH days trade (e.g. stop-outs\n"
            f"prevent entry on some days). Per Aronson: still parameter optimization on the SAME signal."
        )
        audit.add("11. Stop Multiplier Independence", "WARN", detail)
    else:
        audit.add(
            "11. Stop Multiplier Independence",
            "PASS",
            f"All {same_hash} S0.75/S1.0 pairs share the same trade_day_hash — "
            f"confirmed parameter variants, not independent signals",
        )


def check_12_rr_target_independence(con, verbose, audit: AuditResult):
    """Test whether different RR targets produce different trade days."""
    pairs = con.execute("""
        SELECT a.trade_day_hash, b.trade_day_hash,
               a.rr_target as rr_a, b.rr_target as rr_b
        FROM experimental_strategies a
        JOIN experimental_strategies b
            ON a.instrument = b.instrument
            AND a.orb_label = b.orb_label
            AND a.entry_model = b.entry_model
            AND a.confirm_bars = b.confirm_bars
            AND a.filter_type = b.filter_type
            AND a.orb_minutes = b.orb_minutes
            AND a.stop_multiplier = b.stop_multiplier
            AND a.rr_target < b.rr_target
            AND a.is_canonical = TRUE
            AND b.is_canonical = TRUE
        LIMIT 500
    """).fetchall()

    if not pairs:
        audit.add("12. RR Target Independence", "PASS", "No RR pairs to compare")
        return

    same_hash = sum(1 for p in pairs if p[0] == p[1])
    diff_hash = sum(1 for p in pairs if p[0] != p[1])

    detail = (
        f"Out of {len(pairs)} pairs: {same_hash} share hash, {diff_hash} differ.\n"
        f"RR target affects take-profit exit but NOT entry → same entry days expected.\n"
        f"These are parameter optimization (Aronson p.282), not independent discoveries."
    )
    status = "PASS" if diff_hash == 0 else "WARN"
    audit.add("12. RR Target Independence", status, detail)


def check_13_honest_counts(con, verbose, audit: AuditResult):
    """Produce the honest accounting table for each instrument."""
    instruments = con.execute("""
        SELECT DISTINCT instrument FROM validated_setups WHERE status = 'active' ORDER BY instrument
    """).fetchall()

    lines = []
    grand_total = 0
    grand_families = 0
    grand_hashes = 0
    for (inst,) in instruments:
        total = con.execute(
            """
            SELECT COUNT(*) FROM validated_setups WHERE status = 'active' AND instrument = ?
        """,
            [inst],
        ).fetchone()[0]

        unique_hashes = con.execute(
            """
            SELECT COUNT(DISTINCT e.trade_day_hash)
            FROM validated_setups v
            JOIN experimental_strategies e ON v.strategy_id = e.strategy_id
            WHERE v.status = 'active' AND v.instrument = ?
        """,
            [inst],
        ).fetchone()[0]

        unique_families = con.execute(
            """
            SELECT COUNT(DISTINCT family_hash)
            FROM validated_setups WHERE status = 'active' AND instrument = ?
        """,
            [inst],
        ).fetchone()[0]

        try:
            ef_count = con.execute(
                """
                SELECT COUNT(*) FROM edge_families
                WHERE instrument = ? AND robustness_status != 'PURGED'
            """,
                [inst],
            ).fetchone()[0]
        except Exception:
            ef_count = "N/A"

        ratio = total / unique_hashes if unique_hashes > 0 else 0
        lines.append(
            f"{inst}: {total} validated → {unique_hashes} trade-day-hashes → "
            f"{unique_families} family_hashes → {ef_count} edge families "
            f"(avg {ratio:.1f}x redundancy)"
        )
        grand_total += total
        grand_families += unique_families
        grand_hashes += unique_hashes

    lines.append(f"\nTOTAL: {grand_total} validated → {grand_hashes} unique streams → {grand_families} families")
    lines.append("\n*** Use unique streams or edge families for portfolio sizing, NOT raw strategy count ***")

    audit.add("13. Honest Counts", "WARN", "HONEST ACCOUNTING (use these in all reporting):\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    db_path = args.db_path or str(GOLD_DB_PATH)

    print(f"FDR Integrity Audit — DB: {db_path}")
    print(f"Alpha: {args.alpha}")

    con = duckdb.connect(db_path, read_only=True)
    audit = AuditResult()

    checks = [
        ("1. BH Correctness", lambda: check_1_bh_correctness(con, args.alpha, args.verbose, audit)),
        ("2. K Drift", lambda: check_2_k_drift(con, args.verbose, audit)),
        ("3. K Uniformity", lambda: check_3_k_uniformity(con, args.verbose, audit)),
        ("4. Ghost Strategies", lambda: check_4_ghost_strategies(con, args.verbose, audit)),
        ("5. FDR Gate Consistency", lambda: check_5_fdr_gate_consistency(con, args.alpha, args.verbose, audit)),
        ("6. Trade-Day-Hash Accuracy", lambda: check_6_trade_day_hash_accuracy(con, args.verbose, audit)),
        ("7. Parameter Redundancy", lambda: check_7_parameter_redundancy(con, args.verbose, audit)),
        ("8. Independent Bets (Jaccard)", lambda: check_8_independent_bets_jaccard(con, args.verbose, audit)),
        ("9. Session K vs Global K", lambda: check_9_session_k_vs_global(con, args.verbose, audit)),
        ("10. Cross-Instrument K", lambda: check_10_cross_instrument_k(con, args.verbose, audit)),
        ("11. Stop Mult Independence", lambda: check_11_stop_multiplier_independence(con, args.verbose, audit)),
        ("12. RR Target Independence", lambda: check_12_rr_target_independence(con, args.verbose, audit)),
        ("13. Honest Counts", lambda: check_13_honest_counts(con, args.verbose, audit)),
    ]

    for name, fn in checks:
        try:
            print(f"  Running {name}...", end="", flush=True)
            fn()
            print(" done")
        except Exception as e:
            print(" EXCEPTION")
            audit.add(name, "FAIL", f"Exception: {e}")

    con.close()
    print(audit.summary())

    if audit.worst == "FAIL":
        sys.exit(1)
    elif audit.worst == "WARN":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
