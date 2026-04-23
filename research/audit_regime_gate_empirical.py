"""Empirical regime-gate bug verification (Phase 2a of A2b-1).

For each profile-eligible validated lane on the 2026-04-18 rebalance, compute:

  UNFILT        — canonical `_compute_session_regime` output.
                  E2 RR1.0 CB1 O5 pool on (instrument, session), NO filter.
                  This is what the allocator CURRENTLY reads.

  FILT_POOLED   — same E2/RR1.0/CB1/O5 pool + lane's filter applied via
                  canonical `research.filter_utils.filter_signal`. This is
                  what the proposed A2b-1 patch would compute if it keeps
                  the baseline pool and only adds filter gating.

  FILT_LANE     — lane's OWN dimensions (entry_model, rr_target, confirm_bars,
                  orb_minutes, direction) + lane's filter. Diagnostic showing
                  the lane's trailing 6mo ExpR on trades it would actually
                  have taken.

Purpose: empirically decide whether the regime-gate miscalibration is
material (sign-flips on deployed lanes) or cosmetic (sign-agreement across
the universe). Output drives the A2b-1 scope doc priority + kill criteria.

Zero new OOS reads. All three queries read the SAME 6mo trailing window
(2025-10-18 to 2026-04-18) that `_compute_session_regime` already consumed
on the 2026-04-18 rebalance.

One-shot lock: refuses to re-run if result MD exists.

Canonical delegation (per .claude/rules/institutional-rigor.md Rule 4):
  - trading_app.lane_allocator._compute_session_regime: UNFILT
  - trading_app.lane_allocator._month_range: window alignment
  - trading_app.lane_allocator.compute_lane_scores / enrich / ACCOUNT_PROFILES
  - trading_app.validated_shelf.deployable_validated_relation: lane dims
  - research.filter_utils.filter_signal: canonical filter application
  - pipeline.paths.GOLD_DB_PATH, pipeline.db_config.configure_connection
No re-encoding of filter or regime logic.

Output:
  - docs/audit/results/2026-04-18-regime-gate-empirical-verification.md
  - docs/audit/results/2026-04-18-regime-gate-empirical-per-lane.csv
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
import pandas as pd  # noqa: E402

from pipeline.db_config import configure_connection  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.filter_utils import filter_signal  # noqa: E402
from trading_app.config import ALL_FILTERS  # noqa: E402
from trading_app.lane_allocator import (  # noqa: E402
    REGIME_WINDOW_MONTHS,
    LaneScore,
    _compute_session_regime,
    _month_range,
    compute_lane_scores,
    enrich_scores_with_liveness,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES  # noqa: E402
from trading_app.validated_shelf import deployable_validated_relation  # noqa: E402

# ---------------------------------------------------------------------------
# One-shot lock
# ---------------------------------------------------------------------------

REBALANCE_DATE = date(2026, 4, 18)
PROFILE_ID = "topstep_50k_mnq_auto"
RESULT_MD = Path("docs/audit/results/2026-04-18-regime-gate-empirical-verification.md")
RESULT_CSV = Path("docs/audit/results/2026-04-18-regime-gate-empirical-per-lane.csv")

if RESULT_MD.exists():
    print(
        f"REFUSING TO RE-RUN. Result file already exists: {RESULT_MD}\n"
        f"This audit's scope is locked to the {REBALANCE_DATE} rebalance."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Canonical lane-dimension lookup via deployable_validated_relation
# ---------------------------------------------------------------------------


def load_lane_dimensions(con: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    """Return {strategy_id: {entry_model, orb_minutes, rr_target, confirm_bars, filter_type}}.

    Pulls from the canonical deployable-validated relation so every row
    matches what the allocator considered. `validated_setups` has no
    `direction` column — ORB directionality is resolved per-day at break
    time via `orb_{label}_break_dir` and handled inside each filter's
    canonical `matches_df`. No direction parameter needed downstream.
    """
    rel = deployable_validated_relation(con, alias="vs")
    rows = con.execute(
        f"""
        SELECT strategy_id, instrument, orb_label, entry_model,
               orb_minutes, rr_target, confirm_bars, filter_type
        FROM {rel}
        """
    ).fetchall()
    return {
        r[0]: {
            "strategy_id": r[0],
            "instrument": r[1],
            "orb_label": r[2],
            "entry_model": r[3],
            "orb_minutes": int(r[4]),
            "rr_target": float(r[5]),
            "confirm_bars": int(r[6]),
            "filter_type": r[7],
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Filter-aware regime queries (use canonical filter_signal — no re-encoding)
# ---------------------------------------------------------------------------


def _load_pool(
    con: duckdb.DuckDBPyConnection,
    *,
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    orb_minutes: int,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Return orb_outcomes JOIN daily_features within the trailing window.

    Returns a DataFrame carrying `pnl_r` + all `d.*` canonical feature columns
    needed by `filter_signal`. Directionality is handled inside the
    canonical filter's `matches_df` (checks `orb_{label}_break_dir`); no
    direction WHERE clause is applied here.
    """
    # NB: do NOT drop `symbol` from the projection — some canonical filters
    # (CostRatioFilter aka COST_LT*) require `symbol` in `matches_df` to
    # look up per-instrument COST_SPECS. A prior `EXCLUDE (trading_day,
    # symbol, orb_minutes)` clause here silently made COST_LT* return
    # all-False and produced spurious FILT_EMPTY verdicts. Only `o.pnl_r`
    # is selected from the outcomes side, so `d.*` alone introduces no
    # ambiguous column names.
    return con.execute(
        """
        SELECT o.pnl_r, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.entry_model = ?
          AND o.rr_target = ?
          AND o.confirm_bars = ?
          AND o.orb_minutes = ?
          AND o.outcome IN ('win', 'loss')
          AND o.trading_day >= ?
          AND o.trading_day < ?
        """,
        [
            instrument,
            orb_label,
            entry_model,
            rr_target,
            confirm_bars,
            orb_minutes,
            start,
            end,
        ],
    ).fetch_df()


def compute_filtered_regime(
    con: duckdb.DuckDBPyConnection,
    *,
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    orb_minutes: int,
    filter_key: str,
    start: date,
    end: date,
) -> tuple[float | None, int]:
    """Return (avg_pnl_r_when_filter_fires, N_when_filter_fires).

    `filter_key` must exist in `ALL_FILTERS`. If not, returns (None, 0) —
    caller decides how to treat unknown filters.
    """
    if filter_key not in ALL_FILTERS:
        return None, 0
    df = _load_pool(
        con,
        instrument=instrument,
        orb_label=orb_label,
        entry_model=entry_model,
        rr_target=rr_target,
        confirm_bars=confirm_bars,
        orb_minutes=orb_minutes,
        start=start,
        end=end,
    )
    if df.empty:
        return None, 0
    sig = filter_signal(df, filter_key, orb_label)
    fired = df.loc[sig == 1, "pnl_r"]
    n = int(fired.shape[0])
    if n == 0:
        return None, 0
    return round(float(fired.mean()), 4), n


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(unfilt: float | None, filt_pooled: float | None) -> str:
    """Diagnose whether the UNFILTERED regime agrees with the FILTERED regime.

    Codes (returned as strings so they appear in the MD table):
      AGREE_SIGN      — both positive or both negative; deployment verdict unchanged
      SIGN_FLIP       — one positive, one negative; deployment verdict would flip
      FILT_UNKNOWN    — filter not in ALL_FILTERS; cannot compute FILT
      FILT_EMPTY      — filter fires on 0 trades in the window
      UNFILT_EMPTY    — UNFILT is None (no pooled data in window)
    """
    if unfilt is None:
        return "UNFILT_EMPTY"
    if filt_pooled is None:
        return "FILT_EMPTY"
    if (unfilt > 0) == (filt_pooled > 0):
        return "AGREE_SIGN"
    return "SIGN_FLIP"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Profile: {PROFILE_ID}")
    print(f"Rebalance date: {REBALANCE_DATE}")
    start, end = _month_range(REBALANCE_DATE, REGIME_WINDOW_MONTHS)
    print(f"Regime window: {start} -> {end} ({REGIME_WINDOW_MONTHS} months)")
    print()

    # --- Allocator reproduction (retry loop for DB transient) ---
    print("Step 1: compute_lane_scores + enrich_liveness ...")
    scores: list[LaneScore] | None = None
    for attempt in range(8):
        try:
            scores = compute_lane_scores(rebalance_date=REBALANCE_DATE)
            break
        except Exception as e:  # noqa: BLE001 — canonical DB transient; retry OK
            print(f"  attempt {attempt + 1}/8 failed: {e}")
            time.sleep(4)
    if scores is None:
        print("FATAL: compute_lane_scores failed 8 attempts")
        sys.exit(1)
    try:
        enrich_scores_with_liveness(scores)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: liveness enrichment failed ({e}); continuing")
    print(f"  scored {len(scores)} lanes")

    profile = ACCOUNT_PROFILES[PROFILE_ID]
    eligible = [
        s
        for s in scores
        if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")
        and (not profile.allowed_instruments or s.instrument in profile.allowed_instruments)
        and (not profile.allowed_sessions or s.orb_label in profile.allowed_sessions)
    ]
    print(f"  {len(eligible)} profile-eligible lanes")
    print()

    # --- Per-lane regime triplet ---
    print("Step 2: per-lane regime computation (UNFILT + FILT_POOLED + FILT_LANE)")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        lane_dims = load_lane_dimensions(con)
        rows: list[dict] = []
        for s in eligible:
            dims = lane_dims.get(s.strategy_id)
            if dims is None:
                print(f"  WARN: no validated_setups row for {s.strategy_id} — skip")
                continue

            unfilt = _compute_session_regime(
                con, s.instrument, s.orb_label, REBALANCE_DATE
            )

            # FILT_POOLED: baseline pool (E2 RR1.0 CB1 O5) with lane's
            # filter overlaid. Directionality handled inside matches_df.
            filt_pool, n_pool = compute_filtered_regime(
                con,
                instrument=s.instrument,
                orb_label=s.orb_label,
                entry_model="E2",
                rr_target=1.0,
                confirm_bars=1,
                orb_minutes=5,
                filter_key=s.filter_type,
                start=start,
                end=end,
            )

            # FILT_LANE: lane's own dimensions + filter.
            filt_lane, n_lane = compute_filtered_regime(
                con,
                instrument=s.instrument,
                orb_label=s.orb_label,
                entry_model=dims["entry_model"],
                rr_target=dims["rr_target"],
                confirm_bars=dims["confirm_bars"],
                orb_minutes=dims["orb_minutes"],
                filter_key=s.filter_type,
                start=start,
                end=end,
            )

            verdict = classify(unfilt, filt_pool)
            rows.append(
                {
                    "strategy_id": s.strategy_id,
                    "instrument": s.instrument,
                    "orb_label": s.orb_label,
                    "entry_model": dims["entry_model"],
                    "orb_minutes": dims["orb_minutes"],
                    "rr_target": dims["rr_target"],
                    "confirm_bars": dims["confirm_bars"],
                    "filter_type": s.filter_type,
                    "status": s.status,
                    "unfilt_regime": unfilt,
                    "filt_pooled_regime": filt_pool,
                    "filt_pooled_n": n_pool,
                    "filt_lane_regime": filt_lane,
                    "filt_lane_n": n_lane,
                    "verdict": verdict,
                }
            )
    finally:
        con.close()

    # --- Sanity: UNFILT must equal the LaneScore.session_regime_expr it computed from ---
    mismatches = []
    for r in rows:
        want = next(s.session_regime_expr for s in eligible if s.strategy_id == r["strategy_id"])
        if r["unfilt_regime"] != want:
            mismatches.append((r["strategy_id"], want, r["unfilt_regime"]))
    if mismatches:
        print("\nSELF-CONSISTENCY FAILURE — UNFILT does not match LaneScore.session_regime_expr:")
        for sid, want, got in mismatches:
            print(f"  {sid}: expected {want} got {got}")
        sys.exit(1)
    print("  self-consistency: PASS — UNFILT reproduces LaneScore.session_regime_expr")
    print()

    # --- Summary ---
    counts = {
        "AGREE_SIGN": 0,
        "SIGN_FLIP": 0,
        "FILT_EMPTY": 0,
        "FILT_UNKNOWN": 0,
        "UNFILT_EMPTY": 0,
    }
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    print("Step 3: verdict summary")
    for v, c in counts.items():
        print(f"  {v}: {c}")
    print()

    # --- Outputs ---
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {RESULT_CSV}")

    write_result_md(rows, counts, start, end)
    print(f"Wrote {RESULT_MD}")


def write_result_md(rows: list[dict], counts: dict[str, int], start: date, end: date) -> None:
    lines: list[str] = []
    lines.append("# Phase 2a — regime-gate empirical verification")
    lines.append("")
    lines.append(f"- rebalance_date: `{REBALANCE_DATE}`")
    lines.append(f"- profile: `{PROFILE_ID}`")
    lines.append(f"- regime window: `{start}` -> `{end}` (`{REGIME_WINDOW_MONTHS}` months)")
    lines.append(f"- lanes audited: `{len(rows)}`")
    lines.append("- canonical deps: `_compute_session_regime` (UNFILT) + `filter_signal` (FILT)")
    lines.append("- scope: Phase 2a of multi-phase audit roadmap / A2b-1 scope informing")
    lines.append("- OOS consumption: zero (re-reads same window the allocator already consumed)")
    lines.append("")
    lines.append("## Verdict counts")
    lines.append("")
    lines.append("| code | count | meaning |")
    lines.append("|---|---:|---|")
    lines.append(f"| AGREE_SIGN | {counts.get('AGREE_SIGN', 0)} | UNFILT and FILT_POOLED agree on sign — deployment verdict unchanged |")
    lines.append(f"| SIGN_FLIP | {counts.get('SIGN_FLIP', 0)} | UNFILT and FILT_POOLED disagree on sign — deployment verdict flips under patch |")
    lines.append(f"| FILT_EMPTY | {counts.get('FILT_EMPTY', 0)} | filter fires on 0 trades in window — FILT regime undefined |")
    lines.append(f"| FILT_UNKNOWN | {counts.get('FILT_UNKNOWN', 0)} | filter_type not in ALL_FILTERS |")
    lines.append(f"| UNFILT_EMPTY | {counts.get('UNFILT_EMPTY', 0)} | no pooled data in window |")
    lines.append("")

    # Bug verdict summary
    flips = counts.get("SIGN_FLIP", 0)
    empties = counts.get("FILT_EMPTY", 0)
    lines.append("## Bug materiality verdict")
    lines.append("")
    if flips > 0:
        lines.append(f"**BUG_MATERIAL** — {flips} profile-eligible lane(s) would flip deployment sign under the A2b-1 patch. The filtered-regime patch is NOT cosmetic; it changes at least one current deployment decision.")
    elif empties > 0:
        lines.append(f"**BUG_LATENT** — 0 sign flips on current window, but {empties} lane(s) have FILT_EMPTY (filter fires on 0 trades in the 6mo window). Patch behavior is undefined for these lanes and needs a policy (e.g., fall back to UNFILT vs block deployment). A2b-1 scope must specify.")
    else:
        lines.append("**BUG_COSMETIC** — all lanes agree on sign between UNFILT and FILT_POOLED. A2b-1 patch would not change deployment verdicts on the 2026-04-18 rebalance. Patch value is defensive / forward-looking, not immediate.")
    lines.append("")

    # Per-lane table
    lines.append("## Per-lane regime triplet")
    lines.append("")
    lines.append(
        "| strategy_id | inst | session | E/RR/CB/Omin | filter | UNFILT | FILT_POOLED (N) | FILT_LANE (N) | verdict |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---|")
    for r in rows:
        dims = f"{r['entry_model']}/{r['rr_target']}/{r['confirm_bars']}/{r['orb_minutes']}"
        unfilt = "`null`" if r["unfilt_regime"] is None else f"`{r['unfilt_regime']:+.4f}`"
        if r["filt_pooled_regime"] is None:
            fp = "`null` (0)"
        else:
            fp = f"`{r['filt_pooled_regime']:+.4f}` ({r['filt_pooled_n']})"
        if r["filt_lane_regime"] is None:
            fl = "`null` (0)"
        else:
            fl = f"`{r['filt_lane_regime']:+.4f}` ({r['filt_lane_n']})"
        sid_short = r["strategy_id"][:28]
        lines.append(
            f"| `{sid_short}` | {r['instrument']} | {r['orb_label']} | {dims} | `{r['filter_type']}` | {unfilt} | {fp} | {fl} | `{r['verdict']}` |"
        )
    lines.append("")

    lines.append("## Self-consistency")
    lines.append("")
    lines.append("UNFILT column was computed via canonical `_compute_session_regime` and matched `LaneScore.session_regime_expr` for every audited lane (harness would HALT otherwise).")
    lines.append("")

    lines.append("## Next phase")
    lines.append("")
    lines.append("Results feed Phase 2 Stage-1 scope doc `docs/audit/hypotheses/2026-04-18-a2b-1-regime-gate-filtered-patch-preregistered.md`:")
    lines.append("")
    lines.append("- SIGN_FLIP count → whether A2b-1 is BUG_MATERIAL (high-priority) or BUG_COSMETIC (defensive).")
    lines.append("- FILT_EMPTY count → whether the patch needs a fallback policy for lanes whose filter is rare in the 6mo window.")
    lines.append("- FILT_POOLED vs FILT_LANE divergence → whether the patch should consume the baseline pool + filter (conservative) or the lane's own dims + filter (ground-truth).")
    lines.append("")

    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
