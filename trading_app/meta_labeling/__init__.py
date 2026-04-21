"""Meta-labeling research package."""

from trading_app.meta_labeling.dataset import (
    PHASE1_V1_FAMILY,
    DatasetBundle,
    MetaLabelFamilySpec,
    build_family_dataset,
    summarize_dataset,
)
from trading_app.meta_labeling.monotonic import (
    TOKYO_OPEN_MONOTONIC_V1,
    FittedMonotonicAllocator,
    MonotonicAllocatorConfig,
    MonotonicFeatureSpec,
    apply_monotonic_allocator,
    fit_monotonic_allocator,
    run_monotonic_walkforward,
)

__all__ = [
    "DatasetBundle",
    "FittedMonotonicAllocator",
    "MetaLabelFamilySpec",
    "MonotonicAllocatorConfig",
    "MonotonicFeatureSpec",
    "PHASE1_V1_FAMILY",
    "TOKYO_OPEN_MONOTONIC_V1",
    "apply_monotonic_allocator",
    "build_family_dataset",
    "fit_monotonic_allocator",
    "run_monotonic_walkforward",
    "summarize_dataset",
]
