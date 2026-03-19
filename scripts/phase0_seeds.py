import sys; sys.path.insert(0, r"C:\Users\joshd\canompx3")
"""Phase 0 Step 2: Seed stability — 10 seeds x 4 instruments.
Patches RF_PARAMS.random_state at import time."""
import re, json, statistics

# Patch RF_PARAMS before importing meta_label
import trading_app.ml.config as ml_config

from trading_app.ml.meta_label import train_per_session_meta_label
from pipeline.paths import GOLD_DB_PATH

INSTRUMENTS = [
    ("MGC", {"rr_target": 2.5}),
    ("MNQ", {"rr_target": None}),
    ("MES", {"rr_target": 2.5}),
    ("M2K", {"rr_target": 1.0}),
]

results = {}
for seed in range(10):
    # Patch random_state
    ml_config.RF_PARAMS["random_state"] = seed * 7 + 13  # spread seeds
    results[seed] = {}

    for inst, kwargs in INSTRUMENTS:
        try:
            r = train_per_session_meta_label(
                inst, str(GOLD_DB_PATH),
                save_model=False,  # don't overwrite saved models
                run_cpcv=False,    # skip CPCV for speed (not needed for seed stability)
                single_config=True,
                config_selection="max_samples",
                skip_filter=True,
                per_aperture=True,
                **kwargs,
            )
            # Extract totals from results
            honest = 0.0
            full = 0.0
            n_models = 0
            for session, data in r.items():
                if isinstance(data, dict) and "model_type" in data:
                    if data.get("test_auc") is not None:
                        full += data.get("honest_delta_r", 0)
                    if data.get("model_type") == "SESSION":
                        honest += data.get("honest_delta_r", 0)
                        n_models += 1
                elif isinstance(data, dict):
                    # per-aperture: nested dict
                    for ap, apdata in data.items():
                        if isinstance(apdata, dict):
                            if apdata.get("test_auc") is not None:
                                full += apdata.get("honest_delta_r", 0)
                            if apdata.get("model_type") == "SESSION":
                                honest += apdata.get("honest_delta_r", 0)
                                n_models += 1

            results[seed][inst] = {"full": round(full, 1), "honest": round(honest, 1), "models": n_models}
            print(f"  Seed {seed} {inst}: full={full:+.1f}R honest={honest:+.1f}R models={n_models}", flush=True)
        except Exception as e:
            results[seed][inst] = {"full": None, "honest": None, "models": 0}
            print(f"  Seed {seed} {inst}: FAILED -- {e}", flush=True)

    total_full = sum(v["full"] for v in results[seed].values() if v["full"])
    total_honest = sum(v["honest"] for v in results[seed].values() if v["honest"])
    print(f"  === Seed {seed} TOTAL: full={total_full:+.1f}R honest={total_honest:+.1f}R ===", flush=True)

# Summary
print("\n" + "="*60)
print("SEED STABILITY SUMMARY (10 seeds)")
print("="*60)

totals_full = [sum(v["full"] for v in results[s].values() if v["full"]) for s in results]
totals_honest = [sum(v["honest"] for v in results[s].values() if v["honest"]) for s in results]

print(f"Full:   mean={statistics.mean(totals_full):+.1f}  std={statistics.stdev(totals_full):.1f}  min={min(totals_full):+.1f}  max={max(totals_full):+.1f}")
print(f"Honest: mean={statistics.mean(totals_honest):+.1f}  std={statistics.stdev(totals_honest):.1f}  min={min(totals_honest):+.1f}  max={max(totals_honest):+.1f}")
cv = statistics.stdev(totals_honest) / abs(statistics.mean(totals_honest)) * 100
print(f"CV (std/mean): {cv:.1f}%")
print(f"VERDICT: {'STABLE' if cv < 30 else 'MARGINAL' if cv < 50 else 'UNSTABLE'}")

for inst, _ in INSTRUMENTS:
    vals = [results[s][inst]["honest"] for s in results if results[s][inst].get("honest")]
    if len(vals) > 1:
        print(f"  {inst}: mean={statistics.mean(vals):+.1f}  std={statistics.stdev(vals):.1f}  [{min(vals):+.1f}, {max(vals):+.1f}]")
