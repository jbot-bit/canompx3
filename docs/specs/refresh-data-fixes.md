# refresh_data.py — Two Fixes

## Fix 1: Databento exclusive-end date math bug

**File:** `scripts/tools/refresh_data.py` line 223
**Bug:** `yesterday = date.today() - timedelta(days=1)` is passed as Databento API `end` param.
Databento uses **exclusive end** semantics — `end=Apr 1` fetches through Mar 31 only.
When there's a 1-day gap, `start == end` and Databento rejects the request (422).

**Fix:** Separate display date from API end date:
```python
yesterday = date.today() - timedelta(days=1)
api_end = date.today()  # exclusive — fetches through yesterday
```
Pass `api_end` to `download_dbn()`, keep `yesterday` for display/gap calc.

## Fix 2: Outcome builder runs weekly, not daily

**File:** `scripts/tools/refresh_data.py` lines 167-180
**Problem:** Daily refresh calls `outcome_builder --force` with no date bounds.
This rebuilds ALL ~3000 days of outcomes to add 1 new day (~7.5 min, blocks DB).

**Fix:** Add `--full-rebuild` flag. Default behavior = date-bounded outcome build (last 10 trading days).
Full rebuild only when `--full-rebuild` is passed.

### Code changes in `refresh_data.py`:

**main() — add arg:**
```python
parser.add_argument("--full-rebuild", action="store_true",
                    help="Full outcome rebuild (default: incremental last 10 days)")
```

**run_build_steps() — pass through:**
```python
def run_build_steps(instrument: str, start: date, end: date, full_rebuild: bool = False) -> bool:
```

**Outcome step — conditional:**
```python
# Step 3: Build O5 outcomes
if full_rebuild:
    print("  Full O5 outcome rebuild ...")
    cmd = [sys.executable, "-m", "trading_app.outcome_builder",
           "--instrument", instrument, "--force", "--orb-minutes", "5"]
else:
    # Incremental: only rebuild recent days (trade resolution window)
    lookback = start - timedelta(days=14)  # 10 trading days ≈ 14 calendar
    print(f"  Incremental O5 outcomes from {lookback} ...")
    cmd = [sys.executable, "-m", "trading_app.outcome_builder",
           "--instrument", instrument, "--force",
           "--start", str(lookback), "--end", str(end),
           "--orb-minutes", "5"]
```

**daily_refresh.bat — weekly full rebuild:**
```bat
REM Daily: fast incremental (download + ingest + features + recent outcomes)
python -m scripts.tools.refresh_data 2>&1 >> logs\daily_refresh.log

REM Weekly full rebuild (Sundays only):
for /f %%d in ('powershell -c "(Get-Date).DayOfWeek"') do (
    if "%%d"=="Sunday" (
        python -m scripts.tools.refresh_data --full-rebuild 2>&1 >> logs\weekly_rebuild.log
    )
)
```

### Expected timing:
- Daily (incremental): ~15 sec total (download + ingest + bars + features + 10-day outcomes)
- Weekly (full rebuild): ~8 min (full outcome rebuild for all instruments)
