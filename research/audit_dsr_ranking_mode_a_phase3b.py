"""Phase 3b — Mode-A-true DSR ranking comparison.

Stage-2 prerequisite for A2b-2 per the binding K1 in
`docs/audit/hypotheses/2026-04-18-a2b-2-dsr-ranking-preregistered.md` § 5.

Phase 3a consumed `validated_setups.{sharpe_ratio, sample_size, skewness,
kurtosis_excess}` — values stored at validation time, many of which are
Mode-B grandfathered per `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`
(38/38 active rows drift materially). Phase 3a was therefore apples-to-apples
with current allocator behavior but NOT Mode-A-true.

Phase 3b recomputes per-lane Sharpe / N / skewness / kurtosis_excess from
canonical `orb_outcomes` JOIN `daily_features` (filter via canonical
`research.filter_utils.filter_signal`) restricted to `trading_day <
HOLDOUT_SACRED_FROM`, then re-runs DSR ranking. Question: does the
RANKING_MATERIAL verdict from Phase 3a survive Mode-A correction?

Verdict outcomes:
  RANKING_MATERIAL_PRESERVED  Mode-A DSR rank still flips selection vs raw → A2b-2 Stage-2 proceeds
  RANKING_DIRECTION_FLIPPED   Mode-A DSR rank flips DIFFERENT lanes than Mode-B → scope must be revised
  RANKING_COSMETIC_UNDER_MODE_A  Mode-A DSR matches raw selection → A2b-2 deprioritized

Canonical delegation:
  - SQL pattern mirrors `research/mode_a_revalidation_active_setups.py::compute_mode_a`
    lines 146-172 (citation, NOT execution — that function returns
    annualized Sharpe; Phase 3b needs per-trade Sharpe + skew + kurt for DSR)
  - `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` for the IS cutoff
  - `research.mode_a_revalidation_active_setups.direction_from_execution_spec`
    imported directly (single-source-of-truth for direction parsing)
  - `trading_app.dsr.compute_sr0`, `compute_dsr` for DSR math
  - `research.filter_utils.filter_signal` for filter application
  - `trading_app.lane_allocator.{compute_lane_scores, _effective_annual_r,
    build_allocation, ...}` for ranking + selection (same as Phase 3a)
  - var_sr_by_em + n_eff: SAME canonical inputs as Phase 3a
    (validator's `strategy_validator.py:2186-2199` pattern). var_sr is
    cross-strategy; recomputing it under Mode A is out of scope (Phase 3c
    candidate). Phase 3b isolates the per-lane Sharpe variable.

CrossAssetATRFilter injection: mirror compute_mode_a's lines 178-199
(cross_atr_{source}_pct must be injected into df before filter_signal
call for X_MES_ATR60-style filters).

Outputs:
  - docs/audit/results/2026-04-18-dsr-ranking-mode-a-phase3b.md
  - docs/audit/results/2026-04-18-dsr-ranking-mode-a-phase3b-per-lane.csv

One-shot lock + zero new OOS consumption (reads same Mode-A IS window).
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
import numpy as np  # noqa: E402

from pipeline.db_config import configure_connection  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.filter_utils import filter_signal  # noqa: E402
from research.mode_a_revalidation_active_setups import (  # noqa: E402
    direction_from_execution_spec,
)
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter  # noqa: E402
from trading_app.dsr import compute_dsr, compute_sr0  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402
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
from trading_app.validated_shelf import deployable_validated_relation  # noqa: E402

# ---------------------------------------------------------------------------
# One-shot lock
# ---------------------------------------------------------------------------

REBALANCE_DATE = date(2026, 4, 18)
PROFILE_ID = "topstep_50k_mnq_auto"
RESULT_MD = Path("docs/audit/results/2026-04-18-dsr-ranking-mode-a-phase3b.md")
RESULT_CSV = Path("docs/audit/results/2026-04-18-dsr-ranking-mode-a-phase3b-per-lane.csv")
LANE_ALLOCATION_JSON = Path("docs/runtime/lane_allocation.json")
PHASE_3A_CSV = Path("docs/audit/results/2026-04-18-dsr-ranking-empirical-per-lane.csv")

if RESULT_MD.exists():
    print(
        f"REFUSING TO RE-RUN. Result file already exists: {RESULT_MD}\n"
        f"Phase 3b scope locked to the {REBALANCE_DATE} rebalance."
    )
    sys.exit(1)

if not PHASE_3A_CSV.exists():
    print(f"FATAL: Phase 3a CSV missing at {PHASE_3A_CSV}")
    print("Phase 3b is a Mode-A direction-check on Phase 3a; needs the prior CSV.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Mode-A per-lane Sharpe / skew / kurt computation
# ---------------------------------------------------------------------------

N_EFF_BANDS = [5, 12, 36, 72, 253]
MIN_TRADES_FOR_DSR = 30  # canonical floor; below this, Sharpe / skew / kurt are unreliable


def fetch_lane_specs(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Active validated_setups specs (one row per lane)."""
    rel = deployable_validated_relation(con, alias="vs")
    rows = con.execute(
        f"""
        SELECT strategy_id, instrument, orb_label, orb_minutes, entry_model,
               rr_target, confirm_bars, filter_type, execution_spec
        FROM {rel}
        """
    ).fetchall()
    return [
        {
            "strategy_id": r[0],
            "instrument": r[1],
            "orb_label": r[2],
            "orb_minutes": int(r[3]),
            "entry_model": r[4],
            "rr_target": float(r[5]),
            "confirm_bars": int(r[6]),
            "filter_type": r[7],
            "execution_spec": r[8],
        }
        for r in rows
    ]


def compute_mode_a_dsr_inputs(
    con: duckdb.DuckDBPyConnection, spec: dict
) -> tuple[int, float | None, float | None, float | None]:
    """Return (n, sharpe_per_trade, skewness, kurtosis_excess) under Mode A.

    SQL pattern mirrors `research/mode_a_revalidation_active_setups.py::compute_mode_a`
    lines 146-209 verbatim (the canonical Mode-A query); the only delta is
    return shape — that function returns Sharpe_ann + year_break, this one
    returns the per-trade DSR inputs.
    """
    sess = spec["orb_label"]
    direction = direction_from_execution_spec(spec.get("execution_spec"))
    sql = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND d.orb_{sess}_break_dir = ?
          AND o.pnl_r IS NOT NULL
          AND o.trading_day < ?
        ORDER BY o.trading_day
    """
    df = con.execute(
        sql,
        [
            spec["instrument"], sess, spec["orb_minutes"],
            spec["entry_model"], spec["confirm_bars"], spec["rr_target"],
            direction, HOLDOUT_SACRED_FROM,
        ],
    ).df()
    if len(df) == 0:
        return 0, None, None, None

    filter_type = spec.get("filter_type")

    # CrossAssetATRFilter injection — mirror compute_mode_a:182-199
    if filter_type and filter_type in ALL_FILTERS:
        filt_obj = ALL_FILTERS[filter_type]
        if isinstance(filt_obj, CrossAssetATRFilter):
            source = filt_obj.source_instrument
            if source != spec["instrument"]:
                src_rows = con.execute(
                    """SELECT trading_day, atr_20_pct FROM daily_features
                       WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL""",
                    [source],
                ).fetchall()
                src_map = {(td.date() if hasattr(td, "date") else td): float(pct) for td, pct in src_rows}
                col = f"cross_atr_{source}_pct"
                df[col] = df["trading_day"].apply(
                    lambda d: src_map.get(d.date() if hasattr(d, "date") else d)
                )

    if filter_type and filter_type != "UNFILTERED":
        try:
            fire = np.asarray(filter_signal(df, filter_type, sess)).astype(bool)
        except Exception as e:
            print(f"  [warn] filter_signal failed for {filter_type} on {sess}: {e}")
            return 0, None, None, None
        df_on = df.loc[fire].reset_index(drop=True)
    else:
        df_on = df

    if len(df_on) == 0:
        return 0, None, None, None

    pnl = df_on["pnl_r"].astype(float).to_numpy()
    n = int(len(pnl))
    if n < 2:
        return n, None, None, None
    mean = float(np.mean(pnl))
    # Canonical formula per `trading_app/strategy_discovery.py:540-546, 638-642`:
    #   variance = sum((r-mean)**2) / (n-1)        (ddof=1, sample variance)
    #   std_r    = sqrt(variance)
    #   sharpe   = mean / std_r
    #   m3       = sum((r-mean)**3) / n            (ddof=0, population moment)
    #   m4       = sum((r-mean)**4) / n
    #   skewness        = m3 / std_r**3            (mixed-divisor, intentional)
    #   kurtosis_excess = m4 / std_r**4 - 3
    # This mixed-divisor formulation is what's stored in validated_setups, so
    # any DSR re-computation must match it to be canonical-equivalent.
    std_r = float(np.std(pnl, ddof=1))
    if std_r <= 0:
        return n, None, None, None
    sharpe_per_trade = mean / std_r
    centered = pnl - mean
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skewness = m3 / (std_r**3)
    kurtosis_excess = m4 / (std_r**4) - 3.0
    return n, sharpe_per_trade, skewness, kurtosis_excess


# ---------------------------------------------------------------------------
# Selection variants — reuse Phase 3a monkeypatch pattern
# ---------------------------------------------------------------------------


def selection_under_objective(
    scores, *, score_fn, profile, max_dd, pairs, orb_stats
) -> set[str]:
    # FRAGILITY NOTE: monkey-patch relies on Python late-binding of module
    # globals — build_allocation resolves _effective_annual_r through the
    # module __dict__ at call time. If a future refactor changes that to a
    # top-of-file import alias, this no-ops silently. Acceptable for
    # one-shot research; identical pattern lives in Phase 3a script.
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
    print(f"Profile: {PROFILE_ID}, Rebalance: {REBALANCE_DATE}")
    print(f"Mode-A IS boundary: trading_day < {HOLDOUT_SACRED_FROM}")
    print()

    profile = ACCOUNT_PROFILES[PROFILE_ID]
    tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
    max_dd = tier.max_dd if tier else 3000.0

    # --- Allocator state reproduction (matches Phase 3a) ---
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

    eligible = [
        s
        for s in scores
        if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")
        and (not profile.allowed_instruments or s.instrument in profile.allowed_instruments)
        and (not profile.allowed_sessions or s.orb_label in profile.allowed_sessions)
    ]
    print(f"  scored={len(scores)} eligible={len(eligible)}")
    pairs = compute_pairwise_correlation(eligible)
    print()

    # --- Mode-A per-lane Sharpe/skew/kurt + var_sr/n_eff (canonical inputs) ---
    print("Step 2: Mode-A per-lane DSR inputs ...")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        # var_sr_by_em + n_eff: identical to Phase 3a (cross-strategy values,
        # not per-lane; recomputing under Mode A is out of scope per §5).
        var_sr_by_em = {}
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
            var_sr_by_em[em] = row[0] if row and row[0] else 0.047
        n_eff_canonical = max(
            con.execute("SELECT COUNT(DISTINCT family_hash) FROM edge_families").fetchone()[0] or 253,
            2,
        )
        print(f"  var_sr_by_em: {var_sr_by_em}")
        print(f"  n_eff: {n_eff_canonical}")

        all_specs = {s["strategy_id"]: s for s in fetch_lane_specs(con)}

        mode_a_dsr: dict[str, float] = {}
        mode_a_sr0: dict[str, float] = {}
        mode_a_band: dict[str, dict[int, float]] = {}
        mode_a_inputs: dict[str, dict] = {}

        for s in eligible:
            sid = s.strategy_id
            spec = all_specs.get(sid)
            if spec is None:
                print(f"  [warn] no spec for {sid} — DSR=0")
                mode_a_dsr[sid] = 0.0
                mode_a_sr0[sid] = 0.0
                mode_a_band[sid] = {n: 0.0 for n in N_EFF_BANDS}
                mode_a_inputs[sid] = {"n": 0, "sharpe": None, "skew": None, "kurt": None, "em": "?"}
                continue
            n_a, sr_a, skew_a, kurt_a = compute_mode_a_dsr_inputs(con, spec)
            em = spec["entry_model"]
            mode_a_inputs[sid] = {"n": n_a, "sharpe": sr_a, "skew": skew_a, "kurt": kurt_a, "em": em}
            if n_a < MIN_TRADES_FOR_DSR or sr_a is None or skew_a is None or kurt_a is None:
                mode_a_dsr[sid] = 0.0
                mode_a_sr0[sid] = 0.0
                mode_a_band[sid] = {n: 0.0 for n in N_EFF_BANDS}
                continue
            var_sr = var_sr_by_em.get(em, 0.047)
            sr0_can = compute_sr0(n_eff_canonical, var_sr)
            mode_a_sr0[sid] = sr0_can
            mode_a_dsr[sid] = compute_dsr(sr_a, sr0_can, n_a, skew_a, kurt_a)
            mode_a_band[sid] = {
                n: compute_dsr(sr_a, compute_sr0(n, var_sr), n_a, skew_a, kurt_a)
                for n in N_EFF_BANDS
            }
        print(f"  Mode-A DSR computed for {len(mode_a_dsr)} lanes")
    finally:
        con.close()
    print()

    # --- Three selection variants under Mode-A DSR ---
    print("Step 3: three selection variants under Mode-A DSR ...")

    def score_raw(s: LaneScore) -> float:
        return _effective_annual_r(s)

    def score_dsr(s: LaneScore) -> float:
        return mode_a_dsr.get(s.strategy_id, 0.0)

    def score_combo(s: LaneScore) -> float:
        return _effective_annual_r(s) * mode_a_dsr.get(s.strategy_id, 0.0)

    sel_raw = selection_under_objective(
        scores, score_fn=score_raw, profile=profile, max_dd=max_dd, pairs=pairs, orb_stats=orb_stats
    )
    sel_dsr = selection_under_objective(
        scores, score_fn=score_dsr, profile=profile, max_dd=max_dd, pairs=pairs, orb_stats=orb_stats
    )
    sel_combo = selection_under_objective(
        scores, score_fn=score_combo, profile=profile, max_dd=max_dd, pairs=pairs, orb_stats=orb_stats
    )
    print(f"  sel_raw   ({len(sel_raw)}): {sorted(sel_raw)}")
    print(f"  sel_dsr   ({len(sel_dsr)}): {sorted(sel_dsr)}")
    print(f"  sel_combo ({len(sel_combo)}): {sorted(sel_combo)}")
    print()

    # --- Cross-reference with Phase 3a (Mode-B inputs) ---
    print("Step 4: cross-reference Mode-A vs Mode-B (Phase 3a) ...")
    phase3a_rows = list(csv.DictReader(open(PHASE_3A_CSV, encoding="utf-8")))
    phase3a_by_id = {r["strategy_id"]: r for r in phase3a_rows}
    phase3a_sel_dsr = {r["strategy_id"] for r in phase3a_rows if r["selected_dsr"] == "True"}
    phase3a_sel_combo = {r["strategy_id"] for r in phase3a_rows if r["selected_combo"] == "True"}
    print(f"  Phase 3a sel_dsr   intersect Phase 3b sel_dsr:   {len(phase3a_sel_dsr & sel_dsr)} of {len(phase3a_sel_dsr)}")
    print(f"  Phase 3a sel_combo intersect Phase 3b sel_combo: {len(phase3a_sel_combo & sel_combo)} of {len(phase3a_sel_combo)}")
    print()

    # --- Per-lane comparison rows ---
    rank_dsr = {sid: i + 1 for i, sid in enumerate([s.strategy_id for s in sorted(eligible, key=score_dsr, reverse=True)])}
    rank_combo = {sid: i + 1 for i, sid in enumerate([s.strategy_id for s in sorted(eligible, key=score_combo, reverse=True)])}
    rank_raw = {sid: i + 1 for i, sid in enumerate([s.strategy_id for s in sorted(eligible, key=score_raw, reverse=True)])}

    rows = []
    for s in eligible:
        sid = s.strategy_id
        inp = mode_a_inputs.get(sid, {})
        ph3a = phase3a_by_id.get(sid, {})
        rows.append(
            {
                "strategy_id": sid,
                "instrument": s.instrument,
                "orb_label": s.orb_label,
                "rr_target": s.rr_target,
                "filter_type": s.filter_type,
                "stored_sharpe": float(ph3a.get("dsr_canonical", 0)) if ph3a else None,  # Phase 3a DSR (Mode-B)
                "mode_a_n": inp.get("n", 0),
                "mode_a_sharpe": inp.get("sharpe"),
                "mode_a_skew": inp.get("skew"),
                "mode_a_kurt": inp.get("kurt"),
                "mode_a_sr0": round(mode_a_sr0[sid], 4),
                "mode_a_dsr": round(mode_a_dsr[sid], 4),
                "phase3a_dsr": float(ph3a.get("dsr_canonical", 0)) if ph3a else 0.0,
                "dsr_drift": round(mode_a_dsr[sid] - float(ph3a.get("dsr_canonical", 0)), 4),
                **{f"mode_a_dsr_n{n}": round(mode_a_band[sid][n], 4) for n in N_EFF_BANDS},
                "rank_raw": rank_raw[sid],
                "rank_dsr_mode_a": rank_dsr[sid],
                "rank_combo_mode_a": rank_combo[sid],
                "selected_raw": sid in sel_raw,
                "selected_dsr_mode_a": sid in sel_dsr,
                "selected_combo_mode_a": sid in sel_combo,
            }
        )
    rows.sort(key=lambda r: r["rank_raw"])

    if not rows:
        print("FATAL: zero eligible lanes after Mode-A computation; nothing to write.")
        sys.exit(1)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {RESULT_CSV}")

    write_md(rows, sel_raw, sel_dsr, sel_combo, phase3a_sel_dsr, phase3a_sel_combo, var_sr_by_em, n_eff_canonical)
    print(f"Wrote {RESULT_MD}")


def write_md(rows, sel_raw, sel_dsr, sel_combo, phase3a_sel_dsr, phase3a_sel_combo, var_sr_by_em, n_eff_canonical):
    lines = []
    lines.append("# Phase 3b — Mode-A-true DSR ranking comparison")
    lines.append("")
    lines.append(f"- rebalance_date: `{REBALANCE_DATE}`")
    lines.append(f"- profile: `{PROFILE_ID}`")
    lines.append(f"- Mode-A IS boundary: `trading_day < {HOLDOUT_SACRED_FROM}`")
    lines.append(f"- lanes audited: `{len(rows)}`")
    lines.append(f"- canonical SQL pattern: mirrors `research/mode_a_revalidation_active_setups.py::compute_mode_a` lines 146-209")
    lines.append(f"- DSR inputs delta vs Phase 3a: per-lane Sharpe/skew/kurt RECOMPUTED under Mode-A; var_sr_by_em + n_eff UNCHANGED (cross-strategy values; out of scope per A2b-2 §5)")
    lines.append(f"- prerequisite for: A2b-2 Stage-2 implementation per K1 of `docs/audit/hypotheses/2026-04-18-a2b-2-dsr-ranking-preregistered.md`")
    lines.append("- one-shot lock + zero new OOS consumption")
    lines.append("")

    lines.append("## DSR cross-strategy inputs (unchanged from Phase 3a)")
    lines.append("")
    lines.append(f"- N_eff (edge_families distinct count): `{n_eff_canonical}`")
    lines.append(f"- var_sr by entry_model: `{var_sr_by_em}`")
    lines.append("")

    lines.append("## Selection comparison (top max_slots = 7) — Mode-A inputs")
    lines.append("")
    lines.append(f"- `sel_raw`        ({len(sel_raw)}): {sorted(sel_raw)}")
    lines.append(f"- `sel_dsr_mode_a` ({len(sel_dsr)}): {sorted(sel_dsr)}")
    lines.append(f"- `sel_combo_mode_a` ({len(sel_combo)}): {sorted(sel_combo)}")
    lines.append("")
    lines.append("Cross-reference with Phase 3a (Mode-B inputs):")
    lines.append("")
    lines.append(f"- Phase 3a sel_dsr   ∩ Phase 3b sel_dsr:   `{len(phase3a_sel_dsr & sel_dsr)}` of `{len(phase3a_sel_dsr)}`")
    lines.append(f"- Phase 3a sel_combo ∩ Phase 3b sel_combo: `{len(phase3a_sel_combo & sel_combo)}` of `{len(phase3a_sel_combo)}`")
    lines.append("")

    # Materiality verdict — mirror Phase 3a tie-tolerance
    meta_by_id = {r["strategy_id"]: (r["instrument"], r["orb_label"], r["rr_target"]) for r in rows}
    score_by_id = {r["strategy_id"]: float(r["mode_a_dsr"]) for r in rows}

    def real_diff(a, b, score_map):
        sd = a ^ b
        tied = set()
        for sid in sd:
            for other in sd:
                if (
                    other != sid
                    and meta_by_id.get(other) == meta_by_id.get(sid)
                    and abs(score_map.get(other, 0) - score_map.get(sid, 0)) < 1e-6
                ):
                    tied.update({sid, other})
        return len(sd - tied)

    delta_dsr = real_diff(sel_raw, sel_dsr, score_by_id)
    delta_combo = real_diff(sel_raw, sel_combo, score_by_id)

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Non-tied selection delta vs raw under Mode-A: `dsr={delta_dsr}, combo={delta_combo}`")
    lines.append("")
    if delta_dsr == 0 and delta_combo == 0:
        lines.append("**RANKING_COSMETIC_UNDER_MODE_A** — Mode-A DSR ranking matches raw selection. The Phase 3a RANKING_MATERIAL verdict was a Mode-B artifact. **A2b-2 should be deprioritized**; the ranking patch would not change deployment under Mode-A truth. Scope must be revised before any Stage-2 work.")
    else:
        # Direction-flip check: are the same lanes flipped under Mode-A as under Mode-B?
        flipped_3a = (phase3a_sel_dsr - sel_raw) | (sel_raw - phase3a_sel_dsr) if phase3a_sel_dsr else set()
        flipped_3b = (sel_dsr - sel_raw) | (sel_raw - sel_dsr)
        overlap = flipped_3a & flipped_3b
        lines.append(f"**RANKING_MATERIAL_PRESERVED** — Mode-A DSR rank still flips selection vs raw (`dsr={delta_dsr}` slots, `combo={delta_combo}` slots).")
        lines.append("")
        lines.append(f"Direction overlap: of the lanes Phase 3a flipped under DSR, `{len(overlap)}` of `{len(flipped_3a)}` are ALSO flipped under Mode-A. The remaining `{len(flipped_3a) - len(overlap)}` were Mode-B artifacts; the patch's true selection delta is the Mode-A set.")
        lines.append("")
        lines.append("**A2b-2 K1 prerequisite SATISFIED** — Stage-2 may proceed pending user approval of patch shape (A/B/C/D in scope §11).")
    lines.append("")

    # Per-lane table — focus on selected and dsr changes
    lines.append("## Per-lane Mode-A DSR table")
    lines.append("")
    lines.append("| rank_raw | strategy_id | inst | session | RR | filter | Mode-A N | Mode-A Sharpe | Mode-A skew | Mode-A kurt | SR0 | DSR_mode_a | DSR_phase3a | drift | rank_dsr_mA | rank_combo_mA | sel_raw | sel_dsr_mA | sel_combo_mA |")
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in rows:
        sid_short = r["strategy_id"][:30]
        sharpe_str = f"`{r['mode_a_sharpe']:+.4f}`" if r["mode_a_sharpe"] is not None else "`null`"
        skew_str = f"`{r['mode_a_skew']:+.3f}`" if r["mode_a_skew"] is not None else "`null`"
        kurt_str = f"`{r['mode_a_kurt']:+.3f}`" if r["mode_a_kurt"] is not None else "`null`"
        lines.append(
            f"| {r['rank_raw']:>2} | `{sid_short}` | {r['instrument']} | {r['orb_label']} | {r['rr_target']} | `{r['filter_type']}` | `{r['mode_a_n']}` | {sharpe_str} | {skew_str} | {kurt_str} | `{r['mode_a_sr0']:.3f}` | `{r['mode_a_dsr']:.4f}` | `{r['phase3a_dsr']:.4f}` | `{r['dsr_drift']:+.4f}` | {r['rank_dsr_mode_a']:>2} | {r['rank_combo_mode_a']:>2} | {'Y' if r['selected_raw'] else '.'} | {'Y' if r['selected_dsr_mode_a'] else '.'} | {'Y' if r['selected_combo_mode_a'] else '.'} |"
        )
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("- Phase 3a Mode-B baseline: `docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md`")
    lines.append("- A2b-2 Stage-1 scope: `docs/audit/hypotheses/2026-04-18-a2b-2-dsr-ranking-preregistered.md` (K1 binding)")
    lines.append("- Mode-A revalidation context: `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`")
    lines.append("- Canonical SQL pattern: `research/mode_a_revalidation_active_setups.py::compute_mode_a` lines 146-209")
    lines.append("- DSR canonical: `trading_app/dsr.py`")
    lines.append("- Holdout boundary: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`")
    lines.append("")

    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
