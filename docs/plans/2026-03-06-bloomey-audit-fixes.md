# Bloomey Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix the two actionable issues from the Bloomberg-style project overview audit.

**Architecture:** Config change (health_check.py timeout) + dashboard regen. No schema changes, no new modules.

**Tech Stack:** Python, DuckDB (read-only for dashboard)

---

### Task 1: Bump health check test timeout from 300s to 600s

The test suite takes ~500s sequential. The health check subprocess timeout of 300s causes a false FAIL.

**Files:**
- Modify: `pipeline/health_check.py:113`
- Test: `tests/test_pipeline/test_health_check.py` (if exists, verify timeout is referenced)

**Step 1: Read current code and confirm the line**

Verify `pipeline/health_check.py` line 113 has `timeout=300`.

**Step 2: Edit the timeout**

```python
# Change line 113 from:
timeout=300,
# To:
timeout=600,
```

**Step 3: Run drift checks**

Run: `python pipeline/check_drift.py`
Expected: All checks pass, 0 drift

**Step 4: Run health check to verify it no longer times out**

Run: `python pipeline/health_check.py`
Expected: Tests section shows PASS (or at least no timeout error). Note: this takes ~10 minutes.

**Step 5: Commit**

```bash
git add pipeline/health_check.py
git commit -m "fix: bump health check test timeout 300s → 600s to match actual suite runtime"
```

---

### Task 2: Regenerate dashboard HTML

The static dashboard.html is 18 days stale (last generated Feb 16, data through Feb 20).

**Files:**
- Modify: `dashboard.html` (auto-generated output)

**Step 1: Run dashboard generator**

Run: `python pipeline/dashboard.py`
Expected: Generates `dashboard.html` with current data. No errors.

**Step 2: Verify the output file was updated**

Check that `dashboard.html` has a recent timestamp and reflects data through 2026-02-20.

**Step 3: Commit**

```bash
git add dashboard.html
git commit -m "chore: regen dashboard.html — data through 2026-02-20"
```

---

### Verification: Live portfolio rr_target resolution (NO FIX NEEDED)

`rr_target=None` in `LIVE_PORTFOLIO` specs is **by design** — resolved at build time via `family_rr_locks` JOIN in `live_config.py:build_live_portfolio()`.

**Already guarded by:**
- Drift check #58: `check_live_config_spec_validity`
- Drift check #59: `check_family_rr_locks_coverage`
- Drift check #60: `family_rr_locks` JOIN key completeness
- Drift check #61: RR resolution paths must JOIN `family_rr_locks`
- Test: `tests/test_ui/test_session_helpers.py:223` asserts `rr_target is not None` post-build
- Test: `tests/test_trading_app/test_live_config.py` — full `family_rr_locks` JOIN coverage

**No action required.**
