"""Shared input-validation helper for Phase 6e detectors.

Rationale: institutional-rigor Rule #3 ("refactor when you see a pattern
of bugs"). Extracted after three input-validation patch cycles on the
4 scalar-threshold detectors (strict-<, NaN, None).

Contract: detectors treat None and NaN as MISSING and return []. Upstream
data corruption is a distinct alert class (at PerformanceMonitor /
paper_trades layer), not an Alert 1-7 signal.

Infinity is intentionally passed through as a valid extreme. A rolling
expectancy of -inf (hypothetical extreme regime collapse) MUST fire an
EXPR DRIFT critical, not be suppressed.
"""

import math


def has_missing_input(*values: float | None) -> bool:
    for v in values:
        if v is None:
            return True
        if math.isnan(v):
            return True
    return False
