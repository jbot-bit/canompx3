---
name: ml-verify
description: >
  ML-specific verification gate. Checks lookahead, baseline type, bootstrap status,
  model config, and feature integrity. Use after any ML code change or before trusting ML results.
allowed-tools: Read, Grep, Glob, Bash
---

ML verification gate: $ARGUMENTS

Use when: "ml check", "verify ml", "is the ML clean", "lookahead check", "ml audit", "check model", "bootstrap status", before trusting any ML training result

## Gate 1: Lookahead Blacklist Integrity

```bash
python -c "
from trading_app.ml.config import LOOKAHEAD_BLACKLIST, GLOBAL_FEATURES, SESSION_FEATURE_SUFFIXES, CATEGORICAL_FEATURES, CROSS_SESSION_FEATURES, LEVEL_PROXIMITY_FEATURES

# Check no blacklisted feature is in any active list
all_active = set(GLOBAL_FEATURES) | set(SESSION_FEATURE_SUFFIXES) | set(CATEGORICAL_FEATURES) | set(CROSS_SESSION_FEATURES) | set(LEVEL_PROXIMITY_FEATURES)
overlap = all_active & LOOKAHEAD_BLACKLIST
if overlap:
    print(f'CRITICAL: {len(overlap)} blacklisted features in active lists: {overlap}')
else:
    print(f'PASS: 0 blacklisted features in active lists ({len(LOOKAHEAD_BLACKLIST)} blacklisted, {len(all_active)} active)')
"
```

**CRITICAL fail = STOP.** Blacklisted feature in active list = lookahead bias in production.

## Gate 2: Model Bundle Inspection

```bash
python -c "
import joblib, os
model_path = 'models/ml/meta_label_MNQ_hybrid.joblib'
if not os.path.exists(model_path):
    print('NO MODEL ON DISK')
else:
    b = joblib.load(model_path)
    print(f'Type: {b.get(\"model_type\")}')
    print(f'Format: {b.get(\"bundle_format\")}')
    print(f'RR lock: {b.get(\"rr_target_lock\")}')
    print(f'Trained: {b.get(\"trained_at\")}')
    print(f'Config hash: {b.get(\"config_hash\")}')
    print(f'ML sessions: {b.get(\"n_ml_sessions\")} / {b.get(\"n_none_sessions\")} none')
    print(f'Honest delta: {b.get(\"total_honest_delta_r\")}R')
    print(f'Full delta: {b.get(\"total_full_delta_r\")}R')
    uplift = b.get('total_honest_delta_r', 0) - b.get('total_full_delta_r', 0)
    if abs(uplift) > 0.5 * abs(b.get('total_full_delta_r', 1)):
        print(f'WARNING: Selection uplift {uplift:+.1f}R > 50% of full delta (White 2000)')
    # Check config hash drift
    from trading_app.ml.config import compute_config_hash
    current_hash = compute_config_hash()
    model_hash = b.get('config_hash', '')
    if current_hash != model_hash:
        print(f'DRIFT: Config hash changed! Model={model_hash}, Current={current_hash}')
        print('Model was trained with different config. Retrain required.')
    else:
        print(f'Config hash: MATCH')
"
```

## Gate 3: Baseline Type Check

**CRITICAL:** ML on portfolio-level negative baselines is DEAD (threshold artifact, p=0.35). Per-session negative baselines CAN work but REQUIRE bootstrap verification.

```bash
python -c "
import joblib, os
model_path = 'models/ml/meta_label_MNQ_hybrid.joblib'
if not os.path.exists(model_path):
    print('NO MODEL'); exit()
b = joblib.load(model_path)
sessions = b.get('sessions', {})
is_pa = b.get('bundle_format') == 'per_aperture'
for s, info in sessions.items():
    if is_pa:
        for ak, ainfo in info.items():
            if isinstance(ainfo, dict) and ainfo.get('model') is not None:
                delta = ainfo.get('honest_delta_r', 0)
                print(f'  ML: {s}/{ak} delta={delta:+.1f}R')
    else:
        if isinstance(info, dict) and info.get('model') is not None:
            delta = info.get('honest_delta_r', 0)
            print(f'  ML: {s} delta={delta:+.1f}R')
"
```

## Gate 4: Bootstrap Status

Check if bootstrap permutation tests have been run on all ML survivors:

```bash
ls -la logs/ml_bootstrap_results.log logs/ml_bootstrap_remaining.log 2>/dev/null
echo "---"
grep -E "p-value|PASS|FAIL|MARGINAL" logs/ml_bootstrap_results.log logs/ml_bootstrap_remaining.log 2>/dev/null || echo "NO BOOTSTRAP RESULTS FOUND — run scripts/tools/ml_bootstrap_test.py"
```

**Any ML survivor without bootstrap = UNVERIFIED.** Do not trust honest_delta_r without bootstrap confirmation.

## Gate 5: Feature Coverage

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
# Check VWAP/velocity coverage for key sessions
for sess in ['NYSE_OPEN', 'COMEX_SETTLE', 'US_DATA_1000']:
    col = f'orb_{sess}_vwap'
    r = con.sql(f'''
        SELECT COUNT(*) as total,
               COUNT({col}) as populated,
               ROUND(COUNT({col})*100.0/COUNT(*), 1) as pct
        FROM daily_features WHERE symbol = 'MNQ' AND orb_minutes = 5
    ''').fetchone()
    print(f'{sess} VWAP: {r[2]}% ({r[1]}/{r[0]})')
con.close()
"
```

## Output

```
=== ML VERIFY ===
Gate 1 (Lookahead):    [PASS/FAIL]
Gate 2 (Model Bundle): [PRESENT/MISSING/DRIFT]
Gate 3 (Baseline):     [POSITIVE/NEGATIVE+BOOTSTRAP/NEGATIVE+UNVERIFIED]
Gate 4 (Bootstrap):    [VERIFIED p=X/UNVERIFIED]
Gate 5 (Features):     [COMPLETE/PARTIAL N%]
Overall: [CLEAN / NEEDS WORK / CRITICAL]
=================
```

## Key NO-GOs (from Blueprint §5)
- ML on portfolio-level negative baselines: DEAD (p=0.35)
- ML per-session negative: ALIVE but REQUIRES bootstrap (p<0.05)
- Always report selection uplift (White 2000)
- Never trust model bundle metadata without verification (hard lesson #7)
