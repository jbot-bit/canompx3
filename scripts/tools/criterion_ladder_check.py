"""Criterion ladder report -- C1..C13 PASS/FAIL/N-A for one strategy_id.

Read-only. Joins ``validated_setups`` + ``experimental_strategies`` to the
canonical helpers (cost spec, holdout policy, locked criteria thresholds) and
emits one row per criterion.

Authority chain (every threshold cites its source):
- Criterion 1   -- pre_registered_criteria.md:42-53 (file existence)
- Criterion 2   -- pre_registered_criteria.md:79-92 (MinBTL bounds)
- Criterion 3   -- pre_registered_criteria.md:96-106 (BH FDR or Pathway B)
- Criterion 4   -- pre_registered_criteria.md:110-118 (Chordia t-stat)
- Criterion 5   -- pre_registered_criteria.md:122-140 + Amendment 2.1
                   (DSR informational since N_eff unresolved)
- Criterion 6   -- pre_registered_criteria.md:144-150 (WFE >= 0.50)
- Criterion 7   -- pre_registered_criteria.md:154-159 (N >= 100)
- Criterion 8   -- pre_registered_criteria.md:162-167 (2026 OOS)
- Criterion 9   -- pre_registered_criteria.md:170-176 (era stability)
- Criterion 10  -- pre_registered_criteria.md:180-192 (volume filter eligibility)
- Criterion 11  -- pre_registered_criteria.md:196-207 (account MC, offline)
- Criterion 12  -- pre_registered_criteria.md:210-218 (SR monitor, runtime)
- Criterion 13  -- pre_registered_criteria.md:222-258 (scratch policy)

Usage:
    python scripts/tools/criterion_ladder_check.py MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Locked thresholds from pre_registered_criteria.md acceptance matrix (line 262-280).
# Numerical literals MAY be cited from the criteria doc — the doc IS the canonical
# source for these. Other live values (cost, sessions) MUST be queried at runtime.
T_THRESHOLD_WITH_THEORY = 3.00  # Criterion 4
T_THRESHOLD_NO_THEORY = 3.79  # Criterion 4
DSR_THRESHOLD = 0.95  # Criterion 5
WFE_THRESHOLD = 0.50  # Criterion 6
SAMPLE_SIZE_THRESHOLD = 100  # Criterion 7
ERA_STABILITY_FLOOR_R = -0.05  # Criterion 9
ERA_STABILITY_MIN_N = 50  # Criterion 9
LOCKED_OPERATIONAL_CAP = 300  # Criterion 2

PASS = "PASS"
FAIL = "FAIL"
NA = "N/A "
WARN = "WARN"


def fmt_row(num: int, name: str, status: str, detail: str) -> str:
    return f"  C{num:<2} {status}  {name:<28} -- {detail}"


def parse_strategy(strategy_id: str) -> dict:
    from trading_app.eligibility.builder import parse_strategy_id

    return parse_strategy_id(strategy_id)


def fetch_validated(con, strategy_id: str) -> dict | None:
    cols = (
        "strategy_id, instrument, orb_label, orb_minutes, entry_model, rr_target, "
        "confirm_bars, filter_type, sample_size, expectancy_r, sharpe_ratio, "
        "fdr_significant, fdr_adjusted_p, p_value, dsr_score, sr0_at_discovery, "
        "wfe, wf_passed, oos_exp_r, era_dependent, max_year_pct, "
        "yearly_results, all_years_positive, n_trials_at_discovery, "
        "discovery_k, discovery_date, validation_pathway, status, "
        "promotion_provenance, slippage_validation_status, validation_run_id"
    )
    row = con.execute(
        f"SELECT {cols} FROM validated_setups WHERE strategy_id = ?",
        [strategy_id],
    ).fetchone()
    if row is None:
        return None
    return dict(zip([c.strip() for c in cols.split(",")], row, strict=False))


def fetch_experimental(con, strategy_id: str) -> int | None:
    """Total MGC/instrument experimental pool (used for Bailey N estimate)."""
    row = con.execute(
        """SELECT COUNT(*) FROM experimental_strategies
           WHERE instrument = (SELECT instrument FROM experimental_strategies
                               WHERE strategy_id = ? LIMIT 1)""",
        [strategy_id],
    ).fetchone()
    return int(row[0]) if row else None


def find_prereg_files(project_root: Path, instrument: str, discovery_date) -> list[Path]:
    """Locate prereg yaml files matching the discovery date and instrument.

    Pattern: docs/audit/hypotheses/YYYY-MM-DD-<instr-lower>-*.yaml
    """
    if discovery_date is None:
        return []
    if isinstance(discovery_date, str):
        ds = discovery_date
    else:
        ds = discovery_date.isoformat()
    instr_lower = instrument.lower()
    hyp_dir = project_root / "docs" / "audit" / "hypotheses"
    if not hyp_dir.exists():
        return []
    return sorted(hyp_dir.glob(f"{ds}-{instr_lower}-*.yaml"))


def chordia_t_from_expectancy(expectancy_r: float, sharpe_per_trade: float, n: int) -> float | None:
    """Approximate t-stat from per-trade Sharpe and N (Harvey-Liu Exhibit 4 form).

    t = sharpe_per_trade * sqrt(N).  Returns None if inputs incomplete.
    Note: validated_setups stores annualized Sharpe (sharpe_ann) in a different
    column; per-trade SR isn't directly stored, so we reconstruct only when
    expectancy_r and sample_size are both present and sharpe_ratio is the
    per-trade Sharpe (validator's convention -- sharpe_ratio is per-trade,
    sharpe_ann is annualized).
    """
    if expectancy_r is None or sharpe_per_trade is None or not n:
        return None
    if sharpe_per_trade == 0:
        return 0.0
    return float(sharpe_per_trade) * math.sqrt(n)


def check_criteria(strategy_id: str, con, project_root: Path) -> list[tuple[int, str, str, str]]:
    """Return list of (num, name, status, detail) for C1..C13."""
    rows: list[tuple[int, str, str, str]] = []

    parsed = parse_strategy(strategy_id)
    instrument = parsed["instrument"]
    vs = fetch_validated(con, strategy_id)

    if vs is None:
        rows.append((0, "lookup", FAIL, f"strategy_id {strategy_id!r} not in validated_setups"))
        return rows

    # ---- C1: pre-reg file present ----
    prereg_files = find_prereg_files(project_root, instrument, vs.get("discovery_date"))
    if vs.get("discovery_date") is None:
        rows.append((1, "Pre-registration", NA, "discovery_date NULL -- legacy/grandfathered row"))
    elif prereg_files:
        rows.append((1, "Pre-registration", PASS, f"{len(prereg_files)} prereg file(s) match: {prereg_files[0].name}"))
    else:
        rows.append(
            (
                1,
                "Pre-registration",
                FAIL,
                f"no prereg yaml at docs/audit/hypotheses/{vs['discovery_date']}-{instrument.lower()}-*.yaml",
            )
        )

    # ---- C2: MinBTL ----
    pool_n = fetch_experimental(con, strategy_id)
    discovery_k = vs.get("discovery_k")
    if pool_n is None and discovery_k is None:
        rows.append((2, "MinBTL (Bailey 2013)", NA, "no pool size available"))
    else:
        # Use discovery_k (frozen per-session K) if available, else total pool
        n_for_btl = discovery_k or pool_n
        clean_years = {"MNQ": 6.65, "MES": 6.65, "MGC": 2.70}.get(instrument)
        n_strict = int(math.floor(math.exp(clean_years / 2.0))) if clean_years else None
        if n_for_btl > LOCKED_OPERATIONAL_CAP:
            rows.append(
                (
                    2,
                    "MinBTL (Bailey 2013)",
                    FAIL,
                    f"N={n_for_btl} > op cap ({LOCKED_OPERATIONAL_CAP}); "
                    f"strict E=1.0 bound for {instrument} = N<={n_strict}",
                )
            )
        elif n_strict is not None and n_for_btl > n_strict:
            rows.append(
                (
                    2,
                    "MinBTL (Bailey 2013)",
                    WARN,
                    f"N={n_for_btl} <= op cap but exceeds strict E=1.0 N<={n_strict}; noise-floor disclosure required",
                )
            )
        else:
            rows.append((2, "MinBTL (Bailey 2013)", PASS, f"N={n_for_btl} <= strict bound N<={n_strict}"))

    # ---- C3: BH FDR (Pathway A) or raw p (Pathway B) ----
    pathway = vs.get("validation_pathway") or "family"
    fdr_p = vs.get("fdr_adjusted_p")
    raw_p = vs.get("p_value")
    if pathway == "individual":
        if raw_p is not None and raw_p < 0.05:
            rows.append((3, "Statistical sig (Pathway B)", PASS, f"raw p={raw_p:.4f} < 0.05"))
        else:
            rows.append((3, "Statistical sig (Pathway B)", FAIL, f"raw p={raw_p}"))
    else:
        if vs.get("fdr_significant") and fdr_p is not None and fdr_p < 0.05:
            rows.append(
                (3, "Statistical sig (Pathway A BH)", PASS, f"BH-adjusted p={fdr_p:.4f} < 0.05 (K={discovery_k})")
            )
        else:
            rows.append(
                (
                    3,
                    "Statistical sig (Pathway A BH)",
                    FAIL,
                    f"BH-adjusted p={fdr_p}, fdr_sig={vs.get('fdr_significant')}",
                )
            )

    # ---- C4: Chordia t-stat ----
    sharpe = vs.get("sharpe_ratio")
    n = vs.get("sample_size")
    t_stat = chordia_t_from_expectancy(vs.get("expectancy_r"), sharpe, n)
    if t_stat is None:
        rows.append((4, "Chordia t-stat", NA, "insufficient data to compute t"))
    else:
        if t_stat >= T_THRESHOLD_NO_THEORY:
            rows.append((4, "Chordia t-stat", PASS, f"t={t_stat:.2f} >= 3.79 (without-theory)"))
        elif t_stat >= T_THRESHOLD_WITH_THEORY:
            rows.append((4, "Chordia t-stat", WARN, f"t={t_stat:.2f} in [3.00, 3.79) -- requires theory citation"))
        else:
            rows.append((4, "Chordia t-stat", FAIL, f"t={t_stat:.2f} < 3.00 (Harvey-Liu-Zhu floor)"))

    # ---- C5: DSR (informational per Amendment 2.1) ----
    dsr = vs.get("dsr_score")
    if dsr is None:
        rows.append((5, "DSR (informational)", NA, "dsr_score NULL"))
    elif dsr > DSR_THRESHOLD:
        rows.append(
            (5, "DSR (informational)", PASS, f"DSR={dsr:.4f} > {DSR_THRESHOLD} (informational only -- Amendment 2.1)")
        )
    else:
        rows.append(
            (
                5,
                "DSR (informational)",
                WARN,
                f"DSR={dsr:.4f} <= {DSR_THRESHOLD} (informational only -- Amendment 2.1; N_eff unresolved)",
            )
        )

    # ---- C6: WFE ----
    wfe = vs.get("wfe")
    if wfe is None:
        rows.append((6, "Walk-forward efficiency", NA, "wfe NULL"))
    elif wfe >= WFE_THRESHOLD:
        rows.append((6, "Walk-forward efficiency", PASS, f"WFE={wfe:.2f} >= {WFE_THRESHOLD}"))
    else:
        rows.append((6, "Walk-forward efficiency", FAIL, f"WFE={wfe:.2f} < {WFE_THRESHOLD}"))

    # ---- C7: Sample size ----
    if n is None:
        rows.append((7, "Sample size", NA, "sample_size NULL"))
    elif n >= SAMPLE_SIZE_THRESHOLD:
        rows.append((7, "Sample size", PASS, f"N={n} >= {SAMPLE_SIZE_THRESHOLD}"))
    else:
        rows.append((7, "Sample size", FAIL, f"N={n} < {SAMPLE_SIZE_THRESHOLD}"))

    # ---- C8: 2026 OOS ----
    oos = vs.get("oos_exp_r")
    is_expr = vs.get("expectancy_r")
    if oos is None:
        rows.append((8, "2026 OOS positive", NA, "oos_exp_r NULL (Mode A or no holdout window)"))
    elif oos >= 0 and (is_expr is None or oos >= 0.40 * is_expr):
        rows.append((8, "2026 OOS positive", PASS, f"OOS ExpR={oos:.4f} (IS={is_expr})"))
    else:
        rows.append((8, "2026 OOS positive", FAIL, f"OOS ExpR={oos:.4f} (IS={is_expr}; require >= 0.40*IS)"))

    # ---- C9: Era stability ----
    era_dep = vs.get("era_dependent")
    if era_dep is None:
        rows.append((9, "Era stability", NA, "era_dependent NULL"))
    elif era_dep:
        rows.append((9, "Era stability", FAIL, "era_dependent=TRUE -- regime-gated only"))
    else:
        rows.append((9, "Era stability", PASS, f"era_dependent=FALSE; max_year_pct={vs.get('max_year_pct')}"))

    # ---- C10: Data era compatibility ----
    filter_type = vs.get("filter_type") or ""
    is_volume_filter = any(tok in filter_type for tok in ("VOL", "REL_VOL", "ORB_VOL"))
    if is_volume_filter:
        rows.append(
            (
                10,
                "Data era compatibility",
                WARN,
                f"volume filter '{filter_type}' -- requires MICRO-era trade verification",
            )
        )
    else:
        rows.append((10, "Data era compatibility", PASS, f"price-based filter '{filter_type}' -- no era restriction"))

    # ---- C11: Account Monte Carlo ----
    rows.append((11, "Account survival MC", NA, "offline gate -- run scripts/tools/account_mc.py at deployment time"))

    # ---- C12: Shiryaev-Roberts monitor ----
    rows.append((12, "SR drift monitor", NA, "runtime gate -- enabled post-deployment in monitor service"))

    # ---- C13: Scratch policy ----
    if not prereg_files:
        rows.append((13, "Scratch policy declared", FAIL, "no prereg file -- scratch_policy field cannot be verified"))
    else:
        # Search prereg files for scratch_policy field
        found = False
        for p in prereg_files:
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "scratch_policy" in txt:
                found = True
                break
        if found:
            rows.append((13, "Scratch policy declared", PASS, f"scratch_policy: declared in {prereg_files[0].name}"))
        else:
            rows.append((13, "Scratch policy declared", FAIL, "no scratch_policy field in prereg yaml"))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Criterion ladder C1..C13 for one strategy_id.")
    parser.add_argument("strategy_id", help="e.g. MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4")
    parser.add_argument("--db", default=None, help="Path to gold.db")
    args = parser.parse_args()

    import duckdb

    from pipeline.paths import GOLD_DB_PATH

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    if not Path(db_path).exists():
        raise SystemExit(f"gold.db not found at {db_path}")

    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = check_criteria(args.strategy_id, con, _PROJECT_ROOT)

    print(f"Criterion ladder: {args.strategy_id}")
    print("=" * (20 + len(args.strategy_id)))
    print()
    n_pass = n_fail = n_na = n_warn = 0
    for num, name, status, detail in rows:
        print(fmt_row(num, name, status, detail))
        if status == PASS:
            n_pass += 1
        elif status == FAIL:
            n_fail += 1
        elif status == WARN:
            n_warn += 1
        else:
            n_na += 1
    print()
    print(f"Summary: {n_pass} PASS  {n_fail} FAIL  {n_warn} WARN  {n_na} N/A")
    print(f"Date: {date.today().isoformat()}")
    print("Authority: docs/institutional/pre_registered_criteria.md (locked criteria).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
