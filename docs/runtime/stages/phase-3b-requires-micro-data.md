---
task: Phase 3b — requires_micro_data attribute on StrategyFilter classes
mode: IMPLEMENTATION
slug: phase-3b-requires-micro-data
agent: claude-code
created: 2026-04-08T04:40:00Z
updated: 2026-04-08T04:40:00Z
plan: docs/plans/2026-04-07-canonical-data-redownload.md
parent_branch: main @ a3b4f32
parent_commit: Phase 3a data_era foundation (b032a03)
---

## Purpose

Phase 3b wires the `data_era` foundation from 3a into the filter layer.
Every `StrategyFilter` subclass declares whether it needs REAL micro
contract data to produce valid signals. Stage 3d will use this to enforce
era discipline: any `validated_setups` row with a filter that
`requires_micro_data == True` must have all its trades on or after
`micro_launch_day(instrument)`.

Zero behavior change to discovery/live/backtest at Phase 3b — just adds
the self-description attribute. Stage 3c consumes it for rebuild scope
filtering; Stage 3d consumes it for drift enforcement.

## Design — `@property` approach

Initial plan said "add `requires_micro_data: bool` attribute to each
StrategyFilter class" — suggesting a dataclass field. However:

1. `StrategyFilter` is a frozen dataclass with non-default fields in
   subclasses. Adding a default field in the base class breaks positional
   construction unless every subclass redeclares.
2. `CompositeFilter` wraps `base: StrategyFilter` and
   `overlay: StrategyFilter` — its `requires_micro_data` must be DYNAMIC
   (True iff any leaf requires it). A static field cannot express this.
3. The existing pattern on `StrategyFilter` is method overrides
   (`describe()`, `matches_row()`, `matches_df()`) — `@property` override
   is consistent with that pattern.

Decision: **`@property requires_micro_data` on base class returning False,
overridden in volume-based subclasses and CompositeFilter.**

```python
@dataclass(frozen=True)
class StrategyFilter:
    filter_type: str
    description: str

    @property
    def requires_micro_data(self) -> bool:
        """Override in volume/OI-based subclasses. Default False (price-based)."""
        return False

@dataclass(frozen=True)
class VolumeFilter(StrategyFilter):
    min_rel_vol: float = 1.2
    lookback_days: int = 20

    @property
    def requires_micro_data(self) -> bool:
        return True

@dataclass(frozen=True)
class OrbVolumeFilter(StrategyFilter):
    min_volume: float

    @property
    def requires_micro_data(self) -> bool:
        return True

# CombinedATRVolumeFilter subclasses VolumeFilter → inherits True (no override)

@dataclass(frozen=True)
class CompositeFilter(StrategyFilter):
    base: StrategyFilter
    overlay: StrategyFilter

    @property
    def requires_micro_data(self) -> bool:
        return self.base.requires_micro_data or self.overlay.requires_micro_data
```

## Expected classification

| Filter class | requires_micro_data | Reason |
|---|---|---|
| `NoFilter` | False | Pass-through |
| `OrbSizeFilter` | False | Price (ORB high-low points) |
| `CostRatioFilter` | False | Price (ORB size / cost model) |
| **`VolumeFilter`** | **True** | rel_vol from bars_1m volume |
| **`CombinedATRVolumeFilter`** | **True** (inherited) | Volume + ATR percentile |
| **`OrbVolumeFilter`** | **True** | Aggregate ORB-window volume |
| `CrossAssetATRFilter` | False | Cross-instrument price ATR |
| `OwnATRPercentileFilter` | False | Own price ATR |
| `OvernightRangeFilter` | False | Price range |
| `OvernightRangeAbsFilter` | False | Price range |
| `PrevDayRangeNormFilter` | False | Price range |
| `GapNormFilter` | False | Price gap |
| `DirectionFilter` | False | Direction only |
| `CalendarSkipFilter` | False | Calendar |
| `DayOfWeekSkipFilter` | False | Calendar |
| `ATRVelocityFilter` | False | Price ATR |
| `DoubleBreakFilter` | False | Price break |
| `BreakSpeedFilter` | False | Break timing |
| `BreakBarContinuesFilter` | False | Price continuation |
| `PitRangeFilter` | False | Price range |
| `CompositeFilter` | **dynamic** | OR of components |

**Only 3 classes need explicit override:** `VolumeFilter`, `OrbVolumeFilter`,
`CompositeFilter`. `CombinedATRVolumeFilter` inherits from VolumeFilter and
gets True automatically.

## Scope Lock

- trading_app/config.py

(Tests in `tests/` are in `SAFE_DIRS` — not scope-gated.)

## Blast Radius

**Direct edit:** 1 production file (trading_app/config.py, NEVER_TRIVIAL).

**Additive only — zero existing behavior changes:**
- `@property requires_micro_data` returns False by default → every existing
  filter's behavior unchanged
- Three overrides return True for volume-based filters → makes implicit
  semantic explicit, no runtime impact until consumers (3d) wire it up
- `CompositeFilter` override computes dynamically → consistent with existing
  composite semantics (`matches_row = base AND overlay`)

**Downstream callers — none today:**
- Grep `requires_micro_data` in `trading_app/` and `pipeline/` → 0 hits
- No existing consumer. Stage 3b adds the attribute; Stage 3c/3d add
  consumers.

**Drift check coverage:**
- Existing check 86 (`check_filter_self_description_coverage`) iterates
  `ALL_FILTERS` and calls `.describe()` — it does NOT currently call
  `.requires_micro_data`. Stage 3b leaves check 86 untouched.
- New drift check (optional, defer to 3d): "every ALL_FILTERS entry has
  a `requires_micro_data` attribute that returns a bool without raising".

**Companion tests:**
- `tests/test_trading_app/test_config.py` — extend with `TestRequiresMicroData`
  class covering all ~20 filter classes + CompositeFilter dynamic case.

**Mode A holdout integrity:** Pure code + tests. No data writes, no
discovery, no validated_setups changes. Holdout not at risk.

**Out of scope (deferred):**
- Stage 3c: rebuild orb_outcomes / daily_features from new bars_1m
- Stage 3d: drift check enforcing era discipline on validated_setups
- Consumer wire-up (strategy_discovery, strategy_validator, etc)

## TDD Sequence

1. **Red:** write `TestRequiresMicroData` in `test_config.py` covering
   every filter class → run → all fail (attribute doesn't exist).
2. **Green:** add `@property requires_micro_data` to base `StrategyFilter`
   returning False. Override on `VolumeFilter`, `OrbVolumeFilter`,
   `CompositeFilter`. Run tests → all pass.
3. **Verify:** `pytest tests/test_trading_app/test_config.py` →
   green; `pytest tests/test_pipeline/ tests/test_trading_app/` → full
   pass; `python pipeline/check_drift.py` → 85/0/7 unchanged;
   `python scripts/tools/audit_behavioral.py` → 7/7.
4. **Self-review:** Bloomey check. Verify no dead code, no silent
   failures, composite dynamic logic correct.
5. **Commit + close stage.**

## Done Criteria

- [ ] `StrategyFilter.requires_micro_data` exists (property, returns False)
- [ ] `VolumeFilter.requires_micro_data` overrides to True
- [ ] `OrbVolumeFilter.requires_micro_data` overrides to True
- [ ] `CombinedATRVolumeFilter.requires_micro_data` inherits True (no override)
- [ ] `CompositeFilter.requires_micro_data` computes dynamically from components
- [ ] Every other concrete subclass defaults to False (no override needed)
- [ ] `TestRequiresMicroData` covers all ~20 filter classes + composite case
- [ ] `pytest tests/test_trading_app/ tests/test_pipeline/` → all pass
- [ ] `python pipeline/check_drift.py` → 85/0/7 unchanged
- [ ] `python scripts/tools/audit_behavioral.py` → 7/7 clean
- [ ] One commit with descriptive message
