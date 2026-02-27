# Rebuild & Audit Playbook

Operational procedures for rebuilding the trading pipeline safely.
Created Feb 27 2026 after a validator DELETE bug wiped 5m strategies during a partial 15m/30m rebuild.

---

## 1. Full Rebuild (All Instruments)

Run each instrument sequentially. Never run two instruments against the same DB simultaneously.

```bash
# For each instrument: MGC, MNQ, MES, M2K
for INST in MGC MNQ MES M2K; do
  echo "=== Rebuilding $INST ==="

  # Step 1: Outcomes (adjust --end to latest ingested date)
  python trading_app/outcome_builder.py --instrument $INST --force \
    --start 2021-02-05 --end 2026-02-04

  # Step 2: Discovery (all apertures: 5m, 15m, 30m)
  python trading_app/strategy_discovery.py --instrument $INST

  # Step 3: Validation
  python trading_app/strategy_validator.py --instrument $INST \
    --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 \
    --no-walkforward

  # Step 4: Edge families
  python scripts/tools/build_edge_families.py --instrument $INST
done
```

### Post-Rebuild Audit Gates (MANDATORY)

All three must pass before updating docs or committing:

```bash
# Gate 1: Data integrity (17 checks, exit code 0 required)
python scripts/tools/audit_integrity.py

# Gate 2: Drift detection (36 checks, exit code 0 required)
python pipeline/check_drift.py

# Gate 3: Test suite (all tests pass)
python -m pytest tests/ -x -q
```

---

## 2. Single-Instrument Rebuild

Same as full rebuild but for one instrument only. The scoped DELETE fix (Feb 27 2026) ensures that validating one instrument's apertures won't wipe another instrument's data.

```bash
INST=MGC  # Change as needed

python trading_app/outcome_builder.py --instrument $INST --force \
  --start 2021-02-05 --end 2026-02-04
python trading_app/strategy_discovery.py --instrument $INST
python trading_app/strategy_validator.py --instrument $INST \
  --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 \
  --no-walkforward
python scripts/tools/build_edge_families.py --instrument $INST

# Run all 3 audit gates after
python scripts/tools/audit_integrity.py
python pipeline/check_drift.py
python -m pytest tests/ -x -q
```

---

## 3. Partial Rebuild (Single Aperture) -- DANGER

**WARNING:** Running discovery for only one aperture (e.g., `--orb-minutes 15`) and then validating is now SAFE due to the scoped DELETE fix. The validator only deletes validated_setups rows for apertures it actually processed.

However, **always run all 3 apertures through discovery** when possible. Partial discovery means partial validation, which means incomplete portfolio assembly downstream.

If you must run a partial rebuild:
1. Run discovery with the specific `--orb-minutes` flag
2. Run validation (it will only process the discovered apertures)
3. Run audit gates to verify other apertures are intact
4. Re-run edge families for the full instrument (edge families span apertures)

---

## 4. Doc Update Checklist

After any rebuild that changes strategy counts, update these files:

| File | What to Update |
|------|---------------|
| `TRADING_RULES.md` | Validated count, FDR count, edge family count (line ~55) |
| `ROADMAP.md` | Validated active count (multiple locations) |
| `README.md` | Validated count, drift check count |
| `CLAUDE.md` | Drift check count in Key Commands and Guardrails sections |
| `MEMORY.md` | Current State section strategy counts |
| `.claude/commands/health-check.md` | Drift check count if changed |

**Drift Check #36 will catch stale doc counts automatically.** Run `python pipeline/check_drift.py` to see which docs need updating.

---

## 5. Post-Rebuild Audit Gates

Three mandatory gates after ANY rebuild:

### Gate 1: Data Integrity (`audit_integrity.py`)
```bash
python scripts/tools/audit_integrity.py
```
Checks: outcome coverage, session integrity, E0 contamination, old session names, orphan strategies, duplicate IDs, win rate sanity, negative expectancy, row counts.

**Exit code 0 = PASS. Exit code 1 = violations found.**

### Gate 2: Drift Detection (`check_drift.py`)
```bash
python pipeline/check_drift.py
```
36 static analysis checks including doc-stats consistency (Check #36).

### Gate 3: Test Suite
```bash
python -m pytest tests/ -x -q
```

### All Three via Health Check
```bash
python pipeline/health_check.py
```
Runs drift + integrity + tests + other checks in one command.

---

## 6. Emergency Recovery

### Symptom: Missing aperture (e.g., 5m strategies gone, 15m/30m present)
**Root cause (pre-fix):** Validator DELETE wiped all apertures but only re-inserted what it processed.
**Fix (post Feb 27):** Validator now scopes DELETEs to processed apertures only.
**Recovery:** Re-run discovery + validation for the missing aperture's instrument.

### Symptom: Drift Check #36 fails (doc counts don't match DB)
**Cause:** Docs updated without running a full rebuild, or rebuild completed without doc updates.
**Fix:** Query DB for ground truth, update docs per checklist in Section 4.
```bash
python -c "
import duckdb
con = duckdb.connect('gold.db', read_only=True)
print('Validated active:', con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE status='active'\").fetchone()[0])
print('FDR significant:', con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE status='active' AND fdr_significant\").fetchone()[0])
print('Edge families:', con.execute('SELECT COUNT(*) FROM edge_families').fetchone()[0])
con.close()
"
```

### Symptom: E0 contamination detected
**Cause:** E0 rows re-introduced (should never happen post-purge).
**Fix:** Drift Check #35 catches this. Re-run outcome_builder with `--force` for affected instrument, then full discovery + validation chain.

### Symptom: Old session names in DB
**Cause:** Stale data from pre-session-overhaul (pre Feb 24 2026).
**Fix:** Full outcome rebuild with `--force` for affected instrument.
