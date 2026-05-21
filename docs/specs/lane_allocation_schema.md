# Lane Allocation JSON — Schema (Multi-Profile Aware)

**Status:** Active — replaces single-profile flat shape (legacy).
**Owner:** `trading_app/lane_allocator.py` (writer), `trading_app/prop_profiles.py` (reader).
**Migration stage:** Stage 1a — dual-write (new + legacy). Stage 1d removes legacy.

---

## 1. Filesystem layout

### New (canonical)

```
docs/runtime/lane_allocation/
├── <profile_id_1>.json
├── <profile_id_2>.json
└── ...
```

One file per `AccountProfile.profile_id`. The directory is created lazily by the writer.

### Legacy (transitional)

```
docs/runtime/lane_allocation.json
```

A single file holding ONE profile's allocation at a time. Profile-mismatch fail-closed: a reader requesting a different `profile_id` receives an empty tuple.

**Stage 1a contract:** the writer emits BOTH the new path AND the legacy path (single-profile overwrite semantics on legacy preserved for backward-compat). The reader prefers the new path and falls back to the legacy path on miss.

**Stage 1d removes the legacy writer + reader fallback.** Anything still reading `lane_allocation.json` directly at that point is broken — drift check enforces.

---

## 2. File contents (per-profile JSON)

Identical shape under both paths. The only structural difference between new and legacy is the filesystem location — the JSON body itself is unchanged.

```json
{
  "rebalance_date": "YYYY-MM-DD",
  "trailing_window_months": 12,
  "profile_id": "<profile_id>",
  "lanes": [ /* LaneEntry */ ],
  "paused": [ /* BlockedEntry */ ],
  "stale": [ /* BlockedEntry */ ],
  "displaced": [ /* DisplacedEntry */ ],
  "all_scores_count": <int>
}
```

### Top-level fields

| Field | Type | Notes |
|---|---|---|
| `rebalance_date` | ISO `YYYY-MM-DD` | Allocator run date; consumed by `check_allocation_staleness` (WARN >35d, BLOCK >60d) |
| `trailing_window_months` | int | Canonical value from `lane_allocator.DEPLOY_WINDOW_MONTHS` |
| `profile_id` | str | MUST match the `AccountProfile.profile_id` and the filename stem under the new path |
| `lanes` | list[LaneEntry] | Lanes the bot SHALL trade (`status` ∈ {`DEPLOY`, `PROVISIONAL`}) |
| `paused` | list[BlockedEntry] | `status == "PAUSE"` (filtered through C8 + Chordia gates) |
| `stale` | list[BlockedEntry] | `status == "STALE"` |
| `displaced` | list[DisplacedEntry] | Candidates rejected by correlation gate; informational |
| `all_scores_count` | int | Count of gate-passing scores before greedy allocation |

### LaneEntry

| Field | Type | Notes |
|---|---|---|
| `strategy_id` | str | Canonical strategy id (parsed by `trading_app.eligibility.builder.parse_strategy_id`) |
| `instrument` | str | `MNQ` / `MES` / `MGC` (see `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`) |
| `orb_label` | str | Session label (see `pipeline.dst.SESSION_CATALOG`) |
| `orb_minutes` | int | 5 / 15 / 30 |
| `rr_target` | float | 1.0 / 1.5 / 2.0 typical |
| `filter_type` | str | Filter class name |
| `annual_r` | float | Annualized R estimate |
| `trailing_expr` | float | Trailing-window ExpR |
| `trailing_n` | int | Trailing-window trade count |
| `trailing_wr` | float | Trailing-window win rate |
| `months_negative` | int | Count of negative months in trailing window |
| `session_regime` | str | `HOT` / `COLD` / `FLAT` (computed from `session_regime_expr`) |
| `status` | str | `DEPLOY` / `PROVISIONAL` / `PAUSE` / `STALE` |
| `status_reason` | str | Human-readable rationale |
| `chordia_verdict` | str / null | Gate state — drift check refuses lanes whose verdict is not `PASS_*` |
| `chordia_audit_age_days` | int / null | Doctrine-freshness gate input |
| `c8_oos_status` | str / null | `PASSED` / `NULL` (Phase-4 grandfather) — drift check refuses other values |
| `avg_orb_pts` | float | Optional — present only when `orb_size_stats` supplied at write time |
| `p90_orb_pts` | float | Optional — same condition |

### BlockedEntry

| Field | Type | Notes |
|---|---|---|
| `strategy_id` | str | |
| `status` | str | `PAUSE` / `STALE` |
| `reason` | str | `status_reason` from the LaneScore |
| `chordia_verdict` | str / null | |
| `chordia_audit_age_days` | int / null | |
| `c8_oos_status` | str / null | |

### DisplacedEntry

Free-form dict constructed by the correlation-gate writer. Shape preserved verbatim from `displaced` arg.

---

## 3. Writer contract (Stage 1a)

`trading_app.lane_allocator.save_allocation()` MUST:

1. Build the JSON body once.
2. Write the body to `docs/runtime/lane_allocation/<profile_id>.json` (creating the directory if absent).
3. Also write the body to `docs/runtime/lane_allocation.json` (legacy path — single file, overwritten on every call).
4. Return the legacy path (caller-compat: existing callers print the returned path; that surface stays stable until Stage 1d).

Failure of EITHER write is fail-closed: the function raises. Partial writes are not permitted.

---

## 4. Reader contract (Stage 1a)

`trading_app.prop_profiles.load_allocation_lanes(profile_id)` (and sibling loaders) MUST:

1. Try `docs/runtime/lane_allocation/<profile_id>.json` first.
2. If that file is missing, fall back to `docs/runtime/lane_allocation.json` AND only return its contents when `data.profile_id == profile_id` (preserves Stage 0 fail-closed guard).
3. Return `()` / empty equivalent if both paths miss or the file is corrupt.

Path-override args (`allocation_path` kwarg) continue to take precedence — they bypass the new-path lookup and read the supplied file directly with the same profile-mismatch guard.

---

## 5. Drift-check coverage

- Existing drift checks (read legacy path) — UNCHANGED in Stage 1a; the legacy file is still authoritative.
- Stage 1b adds parity check: for every profile present under the new path, contents MUST equal the legacy file when the legacy file's `profile_id` matches.
- Stage 1d removes the legacy writer + parity check; new path becomes sole authority.

---

## 6. Concurrent-write safety

`shared-state-commit-guard.py` governs `docs/runtime/` writes. With one file per profile, parallel rebalance runs on different profiles no longer collide on the same path — the per-instance commit-guard hits stop firing as cross-profile false positives. They continue to fire (correctly) on same-profile concurrent runs.

The legacy `lane_allocation.json` remains a single shared file in Stage 1a; concurrent rebalance runs CAN still collide on it. Stage 1d removes the collision surface.

---

## 7. Authority

- This spec governs the JSON shape and filesystem layout.
- `trading_app/lane_allocator.py` is the canonical writer; `trading_app/prop_profiles.py` is the canonical reader.
- Changes to the shape MUST update this spec in the same PR (canonical-inline-copy parity rule applies — any drift check or test referencing the literal shape must update with this file).
