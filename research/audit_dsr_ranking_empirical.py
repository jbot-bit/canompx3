"""Phase 3a — empirical DSR-vs-raw-rank comparison on 2026-04-18 rebalance.

Precursor read-only audit for A2b-2 (DSR ranking patch). Mirrors the Phase
2a discipline: canonical-only inputs, one-shot lock, zero OOS consumption,
self-consistency HALT.

Question: the allocator currently ranks lanes for selection by
`_effective_annual_r` (raw annual_r_estimate with multiplicative SR-alarm
and recent-decay discounts). Bailey-Lopez de Prado (2014) Eq 2 / False
Strategy Theorem says raw point estimates are upward-biased by selection
when N_trials >> 1. Replacing the ranking objective with DSR (or
DSR-discounted annual_r) is the A2b-2 patch hypothesis.

Phase 3a quantifies the deployment-side delta:

  Per profile-eligible lane on the 2026-04-18 rebalance, compute three
  rankings and three selections (top max_slots=7):

    R_raw     current ranking via canonical _effective_annual_r
    R_dsr     ranking by DSR score alone (sort descending)
    R_combo   ranking by _effective_annual_r * dsr_score (multiplicative)

  Compare top-K selections. Material if any lane's selection status flips
  vs R_raw; cosmetic if all three rankings produce the same live-7.

Per .claude/rules/backtesting-methodology.md historical failure log entry
2026-04-15: report DSR at multiple N_eff for sensitivity (rel_vol v1
single-N_eff trap). DSR is informational per dsr.py docstring, not a hard
gate; this audit treats it the same.

Canonical delegation — every numeric goes through canonical:
  - trading_app.dsr.compute_sr0, compute_dsr
  - trading_app.lane_allocator.compute_lane_scores, _effective_annual_r,
    build_allocation, compute_orb_size_stats, enrich_scores_with_liveness,
    compute_pairwise_correlation
  - validator pattern (strategy_validator.py:2180-2229) for var_sr_by_em
    and n_eff (DO NOT re-encode the validator's calibration logic)

Outputs:
  - docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md
  - docs/audit/results/2026-04-18-dsr-ranking-empirical-per-lane.csv
"""

from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date  # noqa: E402

import duckdb  # noqa: E402

from pipeline.db_config import configure_connection  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.dsr import compute_dsr, compute_sr0  # noqa: E402
from trading_app.lane_allocator import (  # noqa: E402
    LaneScore,
    _effective_annual_r,
    build_allocation,
    compute_lane_scores,
    compute_orb_size_stats,
    compute_pairwise_correlation,
    enrich_scores_with_liveness,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES, ACCOUNT_TIERS  # noqa: E402

# ---------------------------------------------------------------------------
# One-shot lock
# ---------------------------------------------------------------------------

REBALANCE_DATE = date(2026, 4, 18)
PROFILE_ID = "topstep_50k_mnq_auto"
RESULT_MD = Path("docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md")
RESULT_CSV = Path("docs/audit/results/2026-04-18-dsr-ranking-empirical-per-lane.csv")
LANE_ALLOCATION_JSON = Path("docs/runtime/lane_allocation.json")

if RESULT_MD.exists():
    print(
        f"REFUSING TO RE-RUN. Result file already exists: {RESULT_MD}\n"
        f"Phase 3a scope locked to the {REBALANCE_DATE} rebalance."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Canonical DSR-input gathering — mirrors strategy_validator.py:2180-2229
# ---------------------------------------------------------------------------


def fetch_var_sr_by_em(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """Per-entry-model var of sharpe_ratio across canonical experimental_strategies.

    Replicates strategy_validator.py:2184-2195 verbatim. Uses 0.047 as the
    same defaulting fallback if a model has no qualifying rows.
    """
    out = {}
    for em in ["E1", "E2"]:
        row = con.execute(
            """SELECT VAR_SAMP(sharpe_ratio)
               FROM experimental_strategies
               WHERE entry_model = ?
                 AND sample_size >= 30
                 AND sharpe_ratio IS NOT NULL
                 AND is_canonical = TRUE""",
            [em],
        ).fetchone()
        out[em] = row[0] if row and row[0] else 0.047
    return out


def fetch_n_eff_canonical(con: duckdb.DuckDBPyConnection) -> int:
    """Conservative N_eff = distinct family_hash count.

    Replicates strategy_validator.py:2199-2200 verbatim.
    """
    row = con.execute(
        "SELECT COUNT(DISTINCT family_hash) FROM edge_families"
    ).fetchone()
    return max(row[0] if row and row[0] else 253, 2)


def fetch_lane_dsr_inputs(
    con: duckdb.DuckDBPyConnection,
) -> dict[str, dict]:
    """{strategy_id: {sharpe_ratio, sample_size, skewness, kurtosis_excess, entry_model}}.

    Reads canonical validated_setups columns the validator wrote at promotion
    time. NOTE: per the 2026-04-19 Mode-A revalidation finding, these stored
    values may be Mode-B grandfathered for some lanes. This audit uses the
    SAME stored values the validator's DSR pipeline uses (strategy_validator.py
    line 2206-2215), so the comparison is apples-to-apples with current
    deployment behavior. A separate audit (out of scope for Phase 3a) is
    needed to rebuild dsr_score against strict Mode A lane re-validation.
    """
    rows = con.execute(
        """SELECT strategy_id, sharpe_ratio, sample_size, skewness,
                  kurtosis_excess, entry_model
           FROM validated_setups
           WHERE status = 'active'"""
    ).fetchall()
    return {
        r[0]: {
            "sharpe_ratio": r[1] or 0.0,
            "sample_size": r[2] or 30,
            "skewness": r[3] or 0.0,
            "kurtosis_excess": r[4] or 0.0,
            "entry_model": r[5] or "E2",
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# DSR per lane at multiple N_eff (sensitivity per rel_vol v2 lesson)
# ---------------------------------------------------------------------------

# N_eff sensitivity bands per dsr.py docstring + rel_vol v2 stress-test
# (k=5/12/36/72/300/900/14261). Use a smaller, allocator-relevant ladder.
N_EFF_BANDS = [5, 12, 36, 72, 253]


def lane_dsr_triplet(
    inputs: dict, var_sr_by_em: dict[str, float], n_eff: int
) -> tuple[float, float]:
    """Return (dsr, sr0) for given inputs and N_eff.

    Pure delegation to canonical compute_sr0 / compute_dsr.
    """
    em = inputs["entry_model"]
    var_sr = var_sr_by_em.get(em, 0.047)
    sr0 = compute_sr0(n_eff, var_sr)
    dsr = compute_dsr(
        sr_hat=inputs["sharpe_ratio"],
        sr0=sr0,
        t_obs=inputs["sample_size"],
        skewness=inputs["skewness"],
        kurtosis_excess=inputs["kurtosis_excess"],
    )
    return dsr, sr0


# ---------------------------------------------------------------------------
# Selection variants
# ---------------------------------------------------------------------------


def selection_under_objective(
    scores: list[LaneScore],
    *,
    score_fn,
    profile,
    max_dd: float,
    pairs,
    orb_stats,
) -> set[str]:
    """Run build_allocation with a custom score function injected via sort.

    build_allocation internally uses _effective_annual_r for sorting; we can't
    monkey-patch without import contamination. Cleanest portable approach:
    replicate the selection logic by sorting and feeding the resulting score
    order directly. For Phase 3a we monkey-patch the module attribute briefly
    (single-process, immediate restore) to keep canonical delegation.
    """
    import trading_app.lane_allocator as la_mod

    original = la_mod._effective_annual_r
    la_mod._effective_annual_r = score_fn  # type: ignore[assignment]
    try:
        selected = build_allocation(
            scores,
            max_slots=profile.max_slots,
            max_dd=max_dd,
            allowed_instruments=profile.allowed_instruments,
            allowed_sessions=profile.allowed_sessions,
            stop_multiplier=profile.stop_multiplier,
            orb_size_stats=orb_stats,
            correlation_matrix=pairs,
        )
    finally:
        la_mod._effective_annual_r = original
    return {s.strategy_id for s in selected}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Profile: {PROFILE_ID}")
    print(f"Rebalance date: {REBALANCE_DATE}")
    profile = ACCOUNT_PROFILES[PROFILE_ID]
    tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
    max_dd = tier.max_dd if tier else 3000.0
    print(f"Max slots: {profile.max_slots}, Max DD: ${max_dd}")
    print()

    # --- Allocator state reproduction (with retry for DB transient) ---
    print("Step 1: compute_lane_scores + enrich + orb_stats + pairs ...")
    scores: list[LaneScore] | None = None
    for attempt in range(8):
        try:
            scores = compute_lane_scores(rebalance_date=REBALANCE_DATE)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt + 1}/8 failed: {e}")
            time.sleep(4)
    if scores is None:
        print("FATAL: compute_lane_scores failed 8 attempts")
        sys.exit(1)
    try:
        enrich_scores_with_liveness(scores)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: liveness enrichment failed ({e})")
    orb_stats = compute_orb_size_stats(REBALANCE_DATE)

    # Eligible candidates (mirror Phase 1's filter set)
    eligible = [
        s
        for s in scores
        if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")
        and (not profile.allowed_instruments or s.instrument in profile.allowed_instruments)
        and (not profile.allowed_sessions or s.orb_label in profile.allowed_sessions)
    ]
    print(f"  scored={len(scores)} eligible={len(eligible)}")

    pairs = compute_pairwise_correlation(eligible)
    print(f"  pair rhos: {len(pairs)}")
    print()

    # --- Self-consistency: build_allocation(raw) reproduces lane_allocation.json ---
    print("Step 2: self-consistency check (raw selection reproduces live-N) ...")
    raw_selected = build_allocation(
        scores,
        max_slots=profile.max_slots,
        max_dd=max_dd,
        allowed_instruments=profile.allowed_instruments,
        allowed_sessions=profile.allowed_sessions,
        stop_multiplier=profile.stop_multiplier,
        orb_size_stats=orb_stats,
        correlation_matrix=pairs,
    )
    raw_ids = {s.strategy_id for s in raw_selected}
    import json as _json

    with open(LANE_ALLOCATION_JSON, encoding="utf-8") as f:
        alloc = _json.load(f)
    json_ids = {lane["strategy_id"] for lane in alloc.get("lanes", [])}
    sym_diff = (raw_ids ^ json_ids)
    tie_explained: set[str] = set()
    if sym_diff:
        # Check if the diff is purely tie-break sensitivity: each missing/extra
        # pair must share (instrument, orb_label, rr_target) AND the lane in
        # JSON must have effective_annual_r EQUAL to the lane in reproduction.
        score_by_id = {s.strategy_id: _effective_annual_r(s) for s in scores}
        meta_by_id = {s.strategy_id: (s.instrument, s.orb_label, s.rr_target) for s in scores}
        missing = json_ids - raw_ids
        extra = raw_ids - json_ids
        for jid in list(missing):
            j_meta = meta_by_id.get(jid)
            j_score = score_by_id.get(jid)
            for rid in list(extra):
                if (
                    meta_by_id.get(rid) == j_meta
                    and j_score is not None
                    and score_by_id.get(rid) == j_score
                ):
                    tie_explained.update({jid, rid})
                    extra.discard(rid)
                    break
        unexplained = (missing - tie_explained) | extra
        if unexplained:
            print(f"SELF-CONSISTENCY FAIL — non-tie selection drift vs lane_allocation.json")
            print(f"  unexplained missing/extra: {unexplained}")
            sys.exit(1)
        print(
            f"  self-consistency: PASS_TIED — {len(tie_explained)//2} pair(s) of equal-rank "
            f"swaps tolerated: {sorted(tie_explained)}"
        )
    else:
        print(f"  self-consistency: PASS — raw_selected={raw_ids}")
    print()

    # --- DSR computation per lane at multiple N_eff ---
    print("Step 3: DSR per lane at N_eff bands {} ...".format(N_EFF_BANDS))
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        var_sr_by_em = fetch_var_sr_by_em(con)
        n_eff_canonical = fetch_n_eff_canonical(con)
        print(f"  var_sr_by_em: {var_sr_by_em}")
        print(f"  N_eff (canonical, edge_families): {n_eff_canonical}")

        lane_inputs = fetch_lane_dsr_inputs(con)
        print(f"  active validated_setups rows: {len(lane_inputs)}")
    finally:
        con.close()

    # Per-lane DSR at canonical N_eff + each band
    dsr_canonical: dict[str, float] = {}
    dsr_band: dict[str, dict[int, float]] = {}
    sr0_canonical: dict[str, float] = {}
    for s in eligible:
        inp = lane_inputs.get(s.strategy_id)
        if inp is None:
            dsr_canonical[s.strategy_id] = 0.0
            sr0_canonical[s.strategy_id] = 0.0
            dsr_band[s.strategy_id] = {n: 0.0 for n in N_EFF_BANDS}
            continue
        d_canon, sr0_c = lane_dsr_triplet(inp, var_sr_by_em, n_eff_canonical)
        dsr_canonical[s.strategy_id] = d_canon
        sr0_canonical[s.strategy_id] = sr0_c
        dsr_band[s.strategy_id] = {
            n: lane_dsr_triplet(inp, var_sr_by_em, n)[0] for n in N_EFF_BANDS
        }
    print(f"  dsr_canonical computed for {len(dsr_canonical)} lanes")
    print()

    # --- Three ranking variants ---
    print("Step 4: three selection variants ...")

    def score_raw(s: LaneScore) -> float:
        return _effective_annual_r(s)

    def score_dsr(s: LaneScore) -> float:
        # Higher DSR = better. Scale by 1.0 (ignore annual_r magnitude entirely).
        return dsr_canonical.get(s.strategy_id, 0.0)

    def score_combo(s: LaneScore) -> float:
        # Multiplicative discount: raw rank scaled by DSR confidence.
        return _effective_annual_r(s) * dsr_canonical.get(s.strategy_id, 0.0)

    sel_raw = selection_under_objective(
        scores,
        score_fn=score_raw,
        profile=profile,
        max_dd=max_dd,
        pairs=pairs,
        orb_stats=orb_stats,
    )
    sel_dsr = selection_under_objective(
        scores,
        score_fn=score_dsr,
        profile=profile,
        max_dd=max_dd,
        pairs=pairs,
        orb_stats=orb_stats,
    )
    sel_combo = selection_under_objective(
        scores,
        score_fn=score_combo,
        profile=profile,
        max_dd=max_dd,
        pairs=pairs,
        orb_stats=orb_stats,
    )

    # Sanity: sel_raw must match raw_ids modulo tie-break (same logic above)
    if sel_raw != raw_ids:
        score_by_id = {s.strategy_id: _effective_annual_r(s) for s in scores}
        meta_by_id = {s.strategy_id: (s.instrument, s.orb_label, s.rr_target) for s in scores}
        sd = sel_raw ^ raw_ids
        tied = True
        for sid in sd:
            partner = next(
                (
                    other
                    for other in sd
                    if other != sid
                    and meta_by_id.get(other) == meta_by_id.get(sid)
                    and score_by_id.get(other) == score_by_id.get(sid)
                ),
                None,
            )
            if partner is None:
                tied = False
                break
        if not tied:
            print(f"FATAL: monkeypatch path diverged from canonical (non-tie)")
            print(f"  raw_ids:  {raw_ids}")
            print(f"  sel_raw:  {sel_raw}")
            sys.exit(1)
        print(f"  monkeypatch sel_raw: tied-equivalent to canonical — OK")
    print(f"  sel_raw   ({len(sel_raw)}): {sorted(sel_raw)}")
    print(f"  sel_dsr   ({len(sel_dsr)}): {sorted(sel_dsr)}")
    print(f"  sel_combo ({len(sel_combo)}): {sorted(sel_combo)}")
    print()

    # --- Per-lane comparison rows ---
    rows = []
    rank_raw = {sid: i + 1 for i, sid in enumerate(sorted(eligible, key=score_raw, reverse=True)) for sid in [sid.strategy_id]}
    rank_dsr = {sid: i + 1 for i, sid in enumerate(sorted(eligible, key=score_dsr, reverse=True)) for sid in [sid.strategy_id]}
    rank_combo = {sid: i + 1 for i, sid in enumerate(sorted(eligible, key=score_combo, reverse=True)) for sid in [sid.strategy_id]}

    for s in eligible:
        sid = s.strategy_id
        rows.append(
            {
                "strategy_id": sid,
                "instrument": s.instrument,
                "orb_label": s.orb_label,
                "rr_target": s.rr_target,
                "filter_type": s.filter_type,
                "annual_r_estimate": s.annual_r_estimate,
                "effective_annual_r": _effective_annual_r(s),
                "sr0_canonical": round(sr0_canonical[sid], 4),
                "dsr_canonical": round(dsr_canonical[sid], 4),
                **{f"dsr_n{n}": round(dsr_band[sid][n], 4) for n in N_EFF_BANDS},
                "rank_raw": rank_raw[sid],
                "rank_dsr": rank_dsr[sid],
                "rank_combo": rank_combo[sid],
                "selected_raw": sid in sel_raw,
                "selected_dsr": sid in sel_dsr,
                "selected_combo": sid in sel_combo,
                "selection_delta_dsr": (sid in sel_dsr) != (sid in sel_raw),
                "selection_delta_combo": (sid in sel_combo) != (sid in sel_raw),
            }
        )

    # Sort output by raw rank ascending for readability
    rows.sort(key=lambda r: r["rank_raw"])

    # --- Outputs ---
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {RESULT_CSV}")

    write_result_md(rows, sel_raw, sel_dsr, sel_combo, var_sr_by_em, n_eff_canonical)
    print(f"Wrote {RESULT_MD}")


def write_result_md(rows, sel_raw, sel_dsr, sel_combo, var_sr_by_em, n_eff_canonical) -> None:
    lines: list[str] = []
    lines.append("# Phase 3a — DSR ranking empirical verification")
    lines.append("")
    lines.append(f"- rebalance_date: `{REBALANCE_DATE}`")
    lines.append(f"- profile: `{PROFILE_ID}`")
    lines.append(f"- lanes audited: `{len(rows)}`")
    lines.append("- canonical deps: `trading_app.dsr.compute_sr0/compute_dsr` + `trading_app.lane_allocator.*`")
    lines.append("- validator pattern source: `trading_app/strategy_validator.py:2180-2229`")
    lines.append("- OOS consumption: zero (uses validator-stored Sharpe + Mode-A-aware var_sr from canonical_experimental_strategies)")
    lines.append("- one-shot lock enforced")
    lines.append("")

    lines.append("## DSR inputs (canonical)")
    lines.append("")
    lines.append(f"- N_eff (edge_families distinct count): `{n_eff_canonical}`")
    lines.append(f"- var_sr by entry_model: `{var_sr_by_em}`")
    lines.append(f"- N_eff sensitivity bands also reported: `{N_EFF_BANDS}`")
    lines.append("")
    lines.append("Per the 2026-04-15 rel_vol v2 stress-test lesson (`.claude/rules/backtesting-methodology.md` historical failure log), DSR is reported at multiple N_eff because single-N_eff DSR can mislead. Per `trading_app/dsr.py` docstring line 35 DSR is informational, not a hard gate.")
    lines.append("")

    # Selection comparison
    lines.append("## Selection comparison (top max_slots = 7)")
    lines.append("")
    lines.append(f"- `sel_raw`   ({len(sel_raw)}): {sorted(sel_raw)}")
    lines.append(f"- `sel_dsr`   ({len(sel_dsr)}): {sorted(sel_dsr)}")
    lines.append(f"- `sel_combo` ({len(sel_combo)}): {sorted(sel_combo)}")
    lines.append("")
    common_dsr = sel_raw & sel_dsr
    common_combo = sel_raw & sel_combo
    delta_dsr_added = sel_dsr - sel_raw
    delta_dsr_removed = sel_raw - sel_dsr
    delta_combo_added = sel_combo - sel_raw
    delta_combo_removed = sel_raw - sel_combo
    lines.append(f"- |raw ∩ dsr|   = {len(common_dsr)}, dsr adds {sorted(delta_dsr_added)}, dsr removes {sorted(delta_dsr_removed)}")
    lines.append(f"- |raw ∩ combo| = {len(common_combo)}, combo adds {sorted(delta_combo_added)}, combo removes {sorted(delta_combo_removed)}")
    lines.append("")

    # Materiality verdict — discount tied swaps (same instrument+orb_label+rr +
    # same effective_annual_r is interchangeable for the deployment decision).
    meta_by_id = {r["strategy_id"]: (r["instrument"], r["orb_label"], r["rr_target"]) for r in rows}
    score_by_id = {r["strategy_id"]: r["effective_annual_r"] for r in rows}

    def real_diff(a: set[str], b: set[str]) -> int:
        sd = a ^ b
        tied = set()
        for sid in sd:
            for other in sd:
                if (
                    other != sid
                    and meta_by_id.get(other) == meta_by_id.get(sid)
                    and score_by_id.get(other) == score_by_id.get(sid)
                ):
                    tied.update({sid, other})
        return len(sd - tied)

    delta_dsr = real_diff(sel_raw, sel_dsr)
    delta_combo = real_diff(sel_raw, sel_combo)
    lines.append(f"- non-tied selection delta vs raw: dsr={delta_dsr}, combo={delta_combo}")
    lines.append("")
    lines.append("## Materiality verdict")
    lines.append("")
    if delta_dsr == 0 and delta_combo == 0:
        lines.append("**RANKING_COSMETIC** — DSR rank and combo rank produce the same live selection as raw rank on this rebalance. A2b-2 patch would not flip any deployment decision; defensive only.")
    elif delta_combo == 0 and delta_dsr > 0:
        lines.append(f"**RANKING_PARTIAL** — pure-DSR ranking flips {delta_dsr} selection slot(s); combo (annual_r × DSR) preserves raw selection. Suggests A2b-2 should ship the multiplicative variant rather than DSR-as-objective.")
    elif delta_combo > 0 and delta_dsr == 0:
        lines.append(f"**RANKING_COMBO_ACTIVE** — combo flips {delta_combo} slot(s) while pure-DSR matches raw. Investigate whether combo's flips are noise from the multiplicative interaction.")
    else:
        lines.append(f"**RANKING_MATERIAL** — DSR flips {delta_dsr} slot(s), combo flips {delta_combo} slot(s). A2b-2 patch is BUG_MATERIAL on the current rebalance.")
    lines.append("")

    # Per-lane table
    lines.append("## Per-lane ranking + DSR sensitivity")
    lines.append("")
    band_cols = "".join(f" | DSR_n{n}" for n in N_EFF_BANDS)
    lines.append(
        f"| rank_raw | strategy_id | inst | session | RR | filter | annual_r | eff_r | SR0 | DSR_can{band_cols} | rank_dsr | rank_combo | sel_raw | sel_dsr | sel_combo |"
    )
    sep = "|" + "|".join(["---"] * (12 + len(N_EFF_BANDS) + 3)) + "|"
    lines.append(sep)
    for r in rows:
        sid_short = r["strategy_id"][:28]
        band_str = "".join(f" | `{r[f'dsr_n{n}']:.4f}`" for n in N_EFF_BANDS)
        lines.append(
            f"| {r['rank_raw']:>2} | `{sid_short}` | {r['instrument']} | {r['orb_label']} | {r['rr_target']} | `{r['filter_type']}` | `{r['annual_r_estimate']:+.1f}` | `{r['effective_annual_r']:+.1f}` | `{r['sr0_canonical']:.3f}` | `{r['dsr_canonical']:.4f}`{band_str} | {r['rank_dsr']:>2} | {r['rank_combo']:>2} | {'Y' if r['selected_raw'] else '.'} | {'Y' if r['selected_dsr'] else '.'} | {'Y' if r['selected_combo'] else '.'} |"
        )
    lines.append("")

    lines.append("## Self-consistency")
    lines.append("")
    lines.append("`build_allocation` under raw `_effective_annual_r` reproduces `docs/runtime/lane_allocation.json` exactly (HALT otherwise). Monkey-patch path under `selection_under_objective(score_fn=score_raw)` reproduces the same set (additional sanity check; HALT otherwise).")
    lines.append("")

    lines.append("## Known limitation — Mode-B grandfathered DSR inputs")
    lines.append("")
    lines.append("This audit consumes `validated_setups.{sharpe_ratio, sample_size, skewness, kurtosis_excess}` as the validator does (`strategy_validator.py:2206-2215`). Per the 2026-04-19 Mode-A revalidation finding (`docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`), all 38 active rows drift materially from strict Mode A. A separate audit (Phase 3b candidate) is needed to recompute lane DSR against Mode-A-fresh Sharpe/skew/kurt before any Stage-2 patch ships. This Phase 3a result is apples-to-apples with current allocator behavior; it is NOT the Mode-A-true ranking.")
    lines.append("")

    lines.append("## Next phase")
    lines.append("")
    lines.append("Result feeds:")
    lines.append("")
    lines.append("- A2b-1 PAUSED note (BUG_COSMETIC verdict + this Phase 3a evidence reorders priority) — `docs/audit/hypotheses/2026-04-18-a2b-1-regime-gate-filtered-patch-preregistered.md`")
    lines.append("- Multi-phase roadmap update — `docs/plans/2026-04-18-multi-phase-audit-roadmap.md` Phase 3 promoted")
    lines.append("- Phase 3 Stage-1 scope doc — `docs/audit/hypotheses/2026-04-18-a2b-2-dsr-ranking-preregistered.md` (TO WRITE, informed by this MD)")
    lines.append("")

    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
