"""
CUSUM (Cumulative Sum) control chart for detecting strategy performance drift.

Reference: arXiv:1509.01570 — "Real-time financial surveillance via quickest
change-point detection methods".

One CUSUMMonitor per strategy. Raises alarm when cumulative downward deviation
from the expected R per trade exceeds `threshold` standard deviations.

threshold=4.0 ≈ 4σ of accumulated drift before alarm — conservative enough to
avoid false positives over ~100-trade windows while catching genuine regime change.
"""

from dataclasses import dataclass, field


@dataclass
class CUSUMMonitor:
    expected_r: float  # Expected R per trade from backtest (e.g. 0.30)
    std_r: float  # Std deviation of R outcomes (use 1.0 as conservative default)
    threshold: float  # Alarm threshold in std units (e.g. 4.0)

    cusum_pos: float = field(default=0.0, init=False)
    cusum_neg: float = field(default=0.0, init=False)
    alarm_triggered: bool = field(default=False, init=False)
    n_trades: int = field(default=0, init=False)

    def update(self, actual_r: float) -> bool:
        """
        Process one completed trade result.

        Returns True if alarm just triggered for the first time.
        After alarm is triggered, returns False on all subsequent calls
        (alarm stays set until clear() is called).
        """
        self.n_trades += 1
        if self.std_r <= 0:
            return False  # Cannot compute z-score — degenerate distribution
        z = (actual_r - self.expected_r) / self.std_r
        self.cusum_neg = min(0.0, self.cusum_neg + z)  # tracks persistent losses
        self.cusum_pos = max(0.0, self.cusum_pos + z)  # tracks persistent gains

        if -self.cusum_neg > self.threshold and not self.alarm_triggered:
            self.alarm_triggered = True
            return True
        return False

    def clear(self) -> None:
        """Reset CUSUM state after investigation/acknowledgement."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.alarm_triggered = False

    @property
    def drift_severity(self) -> float:
        """How many std devs below expected (positive = underperforming)."""
        return -self.cusum_neg
