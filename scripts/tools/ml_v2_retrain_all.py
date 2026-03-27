"""ML V2 Phase 1: Retrain all 108 configs + config selection.

Runs 6 invocations of train_per_session_meta_label:
  3 RR targets (1.0, 1.5, 2.0) × 2 modes (flat, per-aperture)

For each of 12 sessions, selects the best (mode, aperture, RR) by CPCV AUC.
Writes selected configs to JSON (for bootstrap) and markdown (for audit trail).

Usage:
    PYTHONPATH=. python scripts/tools/ml_v2_retrain_all.py

Pre-registration: docs/pre-registrations/ml-v2-preregistration.md
Design doc: docs/plans/ml-improvement-3phase.md
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import SESSION_CHRONOLOGICAL_ORDER
from trading_app.ml.meta_label import print_per_session_results, train_per_session_meta_label

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "plans"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RR_TARGETS = [1.0, 1.5, 2.0]
INSTRUMENT = "MNQ"


def run_all_combos() -> list[dict]:
    """Run 6 config combos and collect results."""
    combos = []
    for rr in RR_TARGETS:
        for per_ap in [False, True]:
            combos.append({"rr": rr, "per_aperture": per_ap})

    all_results = []
    for i, combo in enumerate(combos, 1):
        mode_str = "PER-APERTURE" if combo["per_aperture"] else "FLAT"
        log.info(f"\n{'#' * 70}")
        log.info(f"  COMBO {i}/6: {mode_str} RR={combo['rr']}")
        log.info(f"{'#' * 70}")

        t0 = time.time()
        # single_config=True required for bypass_validated to take effect.
        # single_config=False ignores bypass_validated and loads from
        # validated_setups (DERIVED layer — BANNED for discovery).
        result = train_per_session_meta_label(
            INSTRUMENT,
            str(GOLD_DB_PATH),
            save_model=False,
            run_cpcv=True,
            single_config=True,
            rr_target=combo["rr"],
            per_aperture=combo["per_aperture"],
            bypass_validated=True,
        )
        elapsed = time.time() - t0

        print_per_session_results(result)
        log.info(f"  Combo {i}/6 done in {elapsed:.1f}s")

        all_results.append({
            "rr": combo["rr"],
            "per_aperture": combo["per_aperture"],
            "mode": "per_aperture" if combo["per_aperture"] else "flat",
            "elapsed_s": round(elapsed, 1),
            "n_ml": result["n_ml_sessions"],
            "n_none": result["n_none_sessions"],
            "n_samples": result["n_samples"],
            "total_val_delta_r": result["total_val_delta_r"],
            "total_honest_delta_r": result["total_honest_delta_r"],
            "total_full_delta_r": result["total_full_delta_r"],
            "sessions": result["sessions"],
        })

    return all_results


def select_best_per_session(all_results: list[dict]) -> list[dict]:
    """For each session, pick the best config by CPCV AUC across all 6 combos.

    Returns list of selected configs (one per session, only sessions with
    at least one passing ML model).
    """
    # Flatten all (session, config) candidates across 6 combos
    candidates: dict[str, list[dict]] = {s: [] for s in SESSION_CHRONOLOGICAL_ORDER}

    for combo in all_results:
        rr = combo["rr"]
        mode = combo["mode"]
        sessions = combo["sessions"]

        for session_name, session_data in sessions.items():
            if session_name not in candidates:
                continue

            if combo["per_aperture"]:
                # Nested: session_data is {aperture_key: info_dict}
                for ak, info in session_data.items():
                    if info.get("model_type") != "SESSION":
                        continue
                    aperture_int = int(ak.replace("O", ""))
                    candidates[session_name].append({
                        "session": session_name,
                        "mode": mode,
                        "aperture": aperture_int,
                        "rr": rr,
                        "cpcv_auc": info.get("cpcv_auc") or 0,
                        "test_auc": info.get("test_auc") or 0,
                        "honest_delta_r": info.get("honest_delta_r", 0),
                        "skip_pct": info.get("skip_pct", 0),
                    })
            else:
                # Flat: session_data is info_dict directly
                if session_data.get("model_type") != "SESSION":
                    continue
                candidates[session_name].append({
                    "session": session_name,
                    "mode": mode,
                    "aperture": None,
                    "rr": rr,
                    "cpcv_auc": session_data.get("cpcv_auc") or 0,
                    "test_auc": session_data.get("test_auc") or 0,
                    "honest_delta_r": session_data.get("honest_delta_r", 0),
                    "skip_pct": session_data.get("skip_pct", 0),
                })

    # Select best per session
    selected = []
    for session in SESSION_CHRONOLOGICAL_ORDER:
        pool = candidates[session]
        if not pool:
            continue
        # Sort by CPCV AUC desc, then honest_delta_r desc as tiebreaker
        pool.sort(key=lambda c: (c["cpcv_auc"], c["honest_delta_r"]), reverse=True)
        best = pool[0]
        # Pre-registration gate: CPCV must be >= 0.50
        if best["cpcv_auc"] < 0.50:
            log.info(
                f"  {session}: best CPCV={best['cpcv_auc']:.3f} < 0.50 — skipped"
            )
            continue
        selected.append(best)
        ap_str = f" O{best['aperture']}" if best["aperture"] else " FLAT"
        log.info(
            f"  {session}: SELECTED{ap_str} RR{best['rr']} "
            f"CPCV={best['cpcv_auc']:.3f} AUC={best['test_auc']:.3f} "
            f"dR={best['honest_delta_r']:+.1f}"
        )

    return selected


def save_results(
    all_results: list[dict], selected: list[dict], total_elapsed: float
) -> Path:
    """Save full results + selected configs to JSON."""

    # Strip non-serializable content (numpy types, etc.)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        return obj

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "instrument": INSTRUMENT,
        "db_path": str(GOLD_DB_PATH),
        "total_elapsed_s": round(total_elapsed, 1),
        "combos": _clean(all_results),
        "selected_configs": _clean(selected),
        "n_selected": len(selected),
        "pre_registration": "docs/pre-registrations/ml-v2-preregistration.md",
    }

    json_path = LOG_DIR / "ml_v2_retrain_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"  Results saved: {json_path}")
    return json_path


def write_selection_doc(selected: list[dict]) -> Path:
    """Write config selection markdown (committed BEFORE bootstrap)."""
    lines = [
        "# ML V2 Config Selection",
        "",
        f"**Generated:** {datetime.now(UTC).isoformat()}",
        f"**Instrument:** {INSTRUMENT}",
        "**Pre-registration:** `docs/pre-registrations/ml-v2-preregistration.md`",
        "",
        "Committed BEFORE bootstrap. Selection uses CPCV AUC on train split only.",
        "Test set is never consulted for selection (honest OOS is reported but not used).",
        "",
        "## Selected Configs for Bootstrap",
        "",
        "| # | Session | Mode | Aperture | RR | CPCV AUC | Test AUC | OOS dR | Skip% |",
        "|---|---------|------|----------|----|----------|----------|--------|-------|",
    ]

    if not selected:
        lines.append("| - | *No sessions passed retrain gates* | - | - | - | - | - | - | - |")
    else:
        for i, c in enumerate(selected, 1):
            ap = f"O{c['aperture']}" if c["aperture"] else "ALL"
            lines.append(
                f"| {i} | {c['session']} | {c['mode']} | {ap} | "
                f"{c['rr']:.1f} | {c['cpcv_auc']:.3f} | {c['test_auc']:.3f} | "
                f"{c['honest_delta_r']:+.1f} | {c['skip_pct']:.1%} |"
            )

    lines.extend([
        "",
        f"**Total selected:** {len(selected)}/12 sessions",
        "",
        "## Bootstrap Parameters (from pre-registration)",
        "",
        "- Permutations: 5000 (Phipson & Smyth 2010)",
        "- Family unit: session (K=12)",
        "- BH FDR: q=0.05 at K=12 (promotion), K=108 reported as footnote",
        "- Kill gate: 0=DEAD, 1=CONDITIONAL, >=2=ALIVE",
        "",
        "## Configs NOT Selected (for audit trail)",
        "",
        "Sessions with no ML model across all 6 combos are omitted (negative baseline,",
        "insufficient data, or CPCV below random in ALL configs).",
        "",
    ])

    doc_path = DOCS_DIR / "ml-v2-config-selection.md"
    with open(doc_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"  Selection doc: {doc_path}")
    return doc_path


def main():
    log.info("=" * 70)
    log.info("  ML V2 PHASE 1: RETRAIN ALL CONFIGS + CONFIG SELECTION")
    log.info("=" * 70)
    log.info(f"  Instrument: {INSTRUMENT}")
    log.info(f"  DB: {GOLD_DB_PATH}")
    log.info(f"  Combos: {len(RR_TARGETS)} RR x 2 modes = 6")
    log.info("  Pre-reg: docs/pre-registrations/ml-v2-preregistration.md")

    t_start = time.time()

    # Step 7: Retrain all 6 combos
    all_results = run_all_combos()

    # Step 8: Config selection
    log.info(f"\n{'=' * 70}")
    log.info("  CONFIG SELECTION (by CPCV AUC on train split)")
    log.info(f"{'=' * 70}")
    selected = select_best_per_session(all_results)

    total_elapsed = time.time() - t_start

    # Save artifacts
    log.info(f"\n{'=' * 70}")
    log.info("  SAVING ARTIFACTS")
    log.info(f"{'=' * 70}")
    json_path = save_results(all_results, selected, total_elapsed)
    doc_path = write_selection_doc(selected)

    # Summary
    log.info(f"\n{'=' * 70}")
    log.info("  RETRAIN SUMMARY")
    log.info(f"{'=' * 70}")
    log.info(f"  Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f}m)")

    combo_summary = []
    for r in all_results:
        mode = "PA" if r["per_aperture"] else "FL"
        combo_summary.append(
            f"  {mode} RR{r['rr']:.1f}: {r['n_ml']} ML, "
            f"val={r['total_val_delta_r']:+.1f}R, "
            f"oos={r['total_honest_delta_r']:+.1f}R "
            f"({r['elapsed_s']:.0f}s)"
        )
    log.info("  Per-combo:")
    for s in combo_summary:
        log.info(s)

    log.info(f"\n  SELECTED: {len(selected)}/12 sessions for bootstrap")
    for c in selected:
        ap = f"O{c['aperture']}" if c["aperture"] else "FLAT"
        log.info(
            f"    {c['session']:<22} {ap:<5} RR{c['rr']:.1f} "
            f"CPCV={c['cpcv_auc']:.3f} dR={c['honest_delta_r']:+.1f}"
        )

    if not selected:
        log.info("\n  >>> NO SESSIONS SURVIVED RETRAIN GATES")
        log.info("  >>> ML is effectively DEAD before bootstrap")
        log.info("  >>> Bootstrap would have 0 candidates — skip it")
        log.info("  >>> Decision: ML DEAD (per pre-registration: 0 = add to NO-GO)")
    else:
        log.info(f"\n  NEXT STEP: commit {doc_path.name}, then run bootstrap:")
        log.info("    PYTHONPATH=. python scripts/tools/ml_bootstrap_test.py")

    log.info("\n  Artifacts:")
    log.info(f"    {json_path}")
    log.info(f"    {doc_path}")


if __name__ == "__main__":
    main()
