import sys

sys.path.insert(0, r"C:\Users\joshd\canompx3")

"""Phase 0: Decontaminated seed stability (10 seeds) + bootstrap permutation (20 reps)."""
import statistics

import numpy as np

import trading_app.ml.config as ml_config
from pipeline.paths import GOLD_DB_PATH
from trading_app.ml import features as _feat_mod
from trading_app.ml.meta_label import train_per_session_meta_label

INSTRUMENTS = [
    ("MGC", {"rr_target": 2.5}),
    ("MNQ", {"rr_target": None}),
    ("MES", {"rr_target": 2.5}),
    ("M2K", {"rr_target": 1.0}),
]

def extract_deltas(r):
    """Extract honest and full delta from per-aperture results dict."""
    honest = 0.0
    full = 0.0
    n_models = 0
    for _session, data in r.items():
        if not isinstance(data, dict):
            continue
        # Check if flat (has model_type directly) or nested (per-aperture)
        if "model_type" in data:
            # Flat per-session result
            if data.get("test_auc") is not None:
                full += data.get("honest_delta_r", 0)
            if data.get("model_type") == "SESSION":
                honest += data.get("honest_delta_r", 0)
                n_models += 1
        else:
            # Per-aperture: nested dict {session: {O5: {...}, O15: {...}}}
            for _ap, apdata in data.items():
                if not isinstance(apdata, dict):
                    continue
                if apdata.get("test_auc") is not None:
                    full += apdata.get("honest_delta_r", 0)
                if apdata.get("model_type") == "SESSION":
                    honest += apdata.get("honest_delta_r", 0)
                    n_models += 1
    return honest, full, n_models


# ============================================================
# PART 1: SEED STABILITY (10 seeds)
# ============================================================
print("=" * 70)
print("DECONTAMINATED SEED STABILITY — 10 seeds")
print("=" * 70)

seed_results = {}
for seed in range(10):
    ml_config.RF_PARAMS["random_state"] = seed * 7 + 13
    seed_results[seed] = {}

    for inst, kwargs in INSTRUMENTS:
        try:
            r = train_per_session_meta_label(
                inst, str(GOLD_DB_PATH),
                save_model=False,
                run_cpcv=False,
                single_config=True,
                config_selection="max_samples",
                skip_filter=True,
                per_aperture=True,
                **kwargs,
            )
            honest, full, n_models = extract_deltas(r)
            seed_results[seed][inst] = {"honest": round(honest, 1), "full": round(full, 1), "models": n_models}
            print(f"  Seed {seed} {inst}: honest={honest:+.1f}R full={full:+.1f}R models={n_models}", flush=True)
        except Exception as e:
            seed_results[seed][inst] = {"honest": 0, "full": 0, "models": 0}
            print(f"  Seed {seed} {inst}: FAILED -- {e}", flush=True)

    total_h = sum(v["honest"] for v in seed_results[seed].values())
    total_f = sum(v["full"] for v in seed_results[seed].values())
    total_m = sum(v["models"] for v in seed_results[seed].values())
    print(f"  === Seed {seed} TOTAL: honest={total_h:+.1f}R full={total_f:+.1f}R models={total_m} ===", flush=True)

# Seed summary
print("\n" + "=" * 70)
print("SEED STABILITY SUMMARY")
print("=" * 70)

totals_h = [sum(v["honest"] for v in seed_results[s].values()) for s in seed_results]
totals_f = [sum(v["full"] for v in seed_results[s].values()) for s in seed_results]

print(f"{'Seed':>4} {'Honest':>10} {'Full':>10} {'MGC':>8} {'MNQ':>8} {'MES':>8} {'M2K':>8}")
print("-" * 62)
for s in sorted(seed_results.keys()):
    h = sum(v["honest"] for v in seed_results[s].values())
    f = sum(v["full"] for v in seed_results[s].values())
    mgc = seed_results[s].get("MGC", {}).get("honest", 0)
    mnq = seed_results[s].get("MNQ", {}).get("honest", 0)
    mes = seed_results[s].get("MES", {}).get("honest", 0)
    m2k = seed_results[s].get("M2K", {}).get("honest", 0)
    print(f"{s:>4} {h:>+10.1f} {f:>+10.1f} {mgc:>+8.1f} {mnq:>+8.1f} {mes:>+8.1f} {m2k:>+8.1f}")

print("-" * 62)
mean_h = statistics.mean(totals_h)
std_h = statistics.stdev(totals_h)
mean_f = statistics.mean(totals_f)
std_f = statistics.stdev(totals_f)
cv = std_h / abs(mean_h) * 100 if mean_h != 0 else float('inf')

print(f"{'MEAN':>4} {mean_h:>+10.1f} {mean_f:>+10.1f}")
print(f"{'STD':>4} {std_h:>10.1f} {std_f:>10.1f}")
print(f"{'MIN':>4} {min(totals_h):>+10.1f} {min(totals_f):>+10.1f}")
print(f"{'MAX':>4} {max(totals_h):>+10.1f} {max(totals_f):>+10.1f}")
print(f"\nCV (std/mean): {cv:.1f}%")

if cv < 15 and min(totals_h) > 500:
    verdict = "PASSED (CV < 15%, min > +500R)"
elif cv > 30 or min(totals_h) < 0:
    verdict = "FAILED (CV > 30% or negative seed)"
else:
    verdict = "MARGINAL"
print(f"VERDICT: {verdict}")

for inst, _ in INSTRUMENTS:
    vals = [seed_results[s].get(inst, {}).get("honest", 0) for s in seed_results]
    if any(v != 0 for v in vals) and len(vals) > 1:
        m = statistics.mean(vals)
        sd = statistics.stdev(vals)
        cv_i = sd / abs(m) * 100 if m != 0 else float('inf')
        print(f"  {inst}: mean={m:+.1f}  std={sd:.1f}  CV={cv_i:.1f}%  [{min(vals):+.1f}, {max(vals):+.1f}]")

# ============================================================
# PART 2: BOOTSTRAP PERMUTATION (20 reps)
# ============================================================
print("\n\n" + "=" * 70)
print("BOOTSTRAP PERMUTATION — 20 reps (y-label shuffle)")
print("=" * 70)
print("Training with SHUFFLED labels. If real models beat 95% of shuffled,")
print("the aggregate has genuine skill.\n")

# We need to patch the training to shuffle labels.
# Monkey-patch load_single_config_feature_matrix to shuffle y after loading.
_original_load = _feat_mod.load_single_config_feature_matrix

def _shuffled_load(*args, **kwargs):
    X, y, meta = _original_load(*args, **kwargs)
    # Shuffle y (break the relationship between features and outcomes)
    rng = np.random.RandomState(ml_config.RF_PARAMS["random_state"])
    y_shuffled = y.copy()
    y_shuffled[:] = rng.permutation(y.values)
    return X, y_shuffled, meta

null_deltas = []
for rep in range(20):
    ml_config.RF_PARAMS["random_state"] = rep * 13 + 7
    _feat_mod.load_single_config_feature_matrix = _shuffled_load

    rep_honest = 0
    for inst, kwargs in INSTRUMENTS:
        try:
            r = train_per_session_meta_label(
                inst, str(GOLD_DB_PATH),
                save_model=False,
                run_cpcv=False,
                single_config=True,
                config_selection="max_samples",
                skip_filter=True,
                per_aperture=True,
                **kwargs,
            )
            honest, _, _ = extract_deltas(r)
            rep_honest += honest
        except Exception:
            pass

    null_deltas.append(rep_honest)
    print(f"  Null rep {rep}: honest={rep_honest:+.1f}R", flush=True)

# Restore original
_feat_mod.load_single_config_feature_matrix = _original_load

print("\n" + "=" * 70)
print("BOOTSTRAP PERMUTATION SUMMARY")
print("=" * 70)
print(f"Null distribution (20 reps): mean={statistics.mean(null_deltas):+.1f}  std={statistics.stdev(null_deltas):.1f}")
print(f"  min={min(null_deltas):+.1f}  max={max(null_deltas):+.1f}")
print(f"  95th pct ≈ {sorted(null_deltas)[int(0.95 * len(null_deltas))]:+.1f}R")
print(f"\nReal model (seed 0): {totals_h[0]:+.1f}R")
print(f"Real model (mean):   {mean_h:+.1f}R")
n_above = sum(1 for d in null_deltas if d >= mean_h)
print(f"Null reps >= real mean: {n_above}/{len(null_deltas)}")
if n_above == 0:
    print(f"p-value: < {1/len(null_deltas):.3f} (0/{len(null_deltas)} null reps beat real)")
else:
    print(f"p-value: {n_above/len(null_deltas):.3f}")

if n_above == 0:
    print("VERDICT: REAL SKILL (no null rep reached real performance)")
elif n_above <= 1:
    print("VERDICT: LIKELY REAL (≤5% of null reps reach real)")
else:
    print("VERDICT: NOT SIGNIFICANT (too many null reps match real)")
