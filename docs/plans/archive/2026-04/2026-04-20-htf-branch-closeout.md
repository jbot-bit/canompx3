---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# 2026-04-20 HTF Branch Closeout

**Purpose:** close the current HTF branch with an honest statement of what was fixed, what was verified, where the simple HTF profit thesis failed, and what would justify reopening HTF work.

---

## 1. Integrity verdict

### What was broken

- `pipeline/check_drift.py` was failing on `MGC 2026-04-17` because the `O5`
  row carried `NULL` `prev_week_*` / `prev_month_*` fields while sibling rows
  on the same day had valid values.
- Canonical DB query showed this was a **single-row stale miss**, not a
  system-wide HTF aggregation failure.

### Root cause

- The root-cause class was already fixed in canonical code by commit
  `234c7d0d`:
  - `pipeline.build_daily_features()` now seeds the HTF post-pass with prior
    history via `_load_htf_seed_rows(...)`
  - this prevents narrow incremental builds from outputting `NULL`
    `prev_week_*` / `prev_month_*`
- The lingering `MGC 2026-04-17 O5` row was historical residue from the
  pre-fix state, not evidence that the current aggregator is still wrong.

### What was repaired

- Re-ran:

```bash
./.venv-wsl/bin/python scripts/backfill_htf_levels.py --symbols MGC
```

- Result:
  - `MGC: 1120 rows, 1114 non-null prev_week_high`
  - the stale row was repaired in `gold.db`

### Guardrail added

- New drift check: **HTF fields consistent across apertures for each
  trading_day × symbol**
- Rationale:
  - `prev_week_*` / `prev_month_*` are orb-agnostic
  - a day cannot honestly carry different HTF values across O5/O15/O30
  - this catches partial stale / partial repair states directly

---

## 2. Profit verdict

### Verified scans re-run on current canonical DB

```bash
./.venv-wsl/bin/python research/verify_htf_fire_rate.py
./.venv-wsl/bin/python research/htf_path_a_prev_week_v1_scan.py
./.venv-wsl/bin/python research/htf_path_a_prev_month_v1_scan.py
./.venv-wsl/bin/python research/htf_path_a_overlap_decomposition.py
```

### Structural fire-rate result

- All 6 prev-week v1 family cells are in the acceptable fire-rate band on the
  pre-holdout horizon:
  - MNQ: `12.59%`, `14.12%`, `16.73%` long-fire rates across scanned sessions
  - MES: `10.56%`, `12.54%`, `15.47%`
  - short-fire rates all `5.37%` to `8.13%`
- So the v1 family is **testable**, not dead because of rarity.

### Family result

- `prev-week v1`: **FAMILY KILL**
  - `PASS=0`, `PARK=0`, `KILL=24`
- `prev-month v1`: **FAMILY KILL**
  - `PASS=0`, `PARK=0`, `KILL=24`

### Interpretation

- The simple thesis
  - "if price breaks through prev-week / prev-month high-low in break
    direction, taking that ORB trade is better"
  - does **not** survive the repo's actual gates
- This is not a metadata conclusion. It is verified from the re-run scan
  scripts against canonical `daily_features + orb_outcomes`.

### Overlap decomposition

- `MES EUROPE_FLOW long RR2.0` prev-month wrong-sign result is mostly
  **overlap-driven** by the prev-week family:
  - overlap subset `N=146`, `mean=-0.353`, `t=-4.018`
  - non-overlap subset `N=195`, `mean=-0.119`, `t=-1.384`
- `MES TOKYO_OPEN long RR2.0` prev-month wrong-sign result is
  **non-overlap-driven**:
  - overlap subset `N=121`, `mean=-0.193`, `t=-1.863`
  - non-overlap subset `N=188`, `mean=-0.275`, `t=-3.367`

So even the wrong-sign residuals do not justify "HTF works." They justify
only a narrow observational shadow on specifically pre-registered residual
anomalies.

---

## 3. Decision

### Closed

- **Closed:** simple HTF level-break-as-filter family
  - `prev_week_high/low` break-aligned take filters
  - `prev_month_high/low` break-aligned take filters

### Not closed

- **Not closed:** broader HTF / S-R question
  - first-touch / touched-to-date state
  - distance-to-HTF-level conditioning
  - inside-vs-outside HTF range conditioning
  - rolling-high / rolling-low structures
  - market-profile / auction-theory structures

### Why

- The killed family asked the wrong profit question:
  - "does simple alignment with a weekly/monthly break improve the ORB trade?"
- The remaining plausible questions are more specific:
  - "is the level *fresh*?"
  - "is price already outside the HTF range before the ORB?"
  - "is the ORB breaking *into* a nearby HTF obstacle or away from it?"

Those are different mechanisms. They are not license to reopen the killed
family.

---

## 4. Reopen criteria

Reopen HTF only if at least one of these is true:

1. A new pre-reg uses a **structurally different** mechanism than the killed
   v1 break-through predicate.
2. New pipeline features are added that materially change the information set:
   first-touch / touched-to-date, rolling highs/lows, distance-to-level.
3. New literature grounding is added for level-based theory strong enough to
   justify a different prior and evaluation framing.

Until then, HTF should not consume discovery budget ahead of other mechanism
families.
