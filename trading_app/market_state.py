"""
Market state: single shared object describing current market context.

Built from DB for historical replay, or updated progressively from live bars.
All modules (execution_engine, paper_trader, risk_manager) share this object.

Read-only consumer of the database -- never writes to DB.

Usage:
    state = MarketState.from_trading_day(date(2025, 6, 15), Path("gold.db"))
    print(state.signals)
    print(state.strategy_scores)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb



# =========================================================================
# ORB session labels (ordered by time of day in Brisbane)
# =========================================================================

ORB_LABELS = ["0900", "1000", "1100", "1800", "2300", "0030"]

# Session ordering for cascade logic: which sessions come before which
SESSION_ORDER = {label: i for i, label in enumerate(ORB_LABELS)}


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class OrbSnapshot:
    """State of a single ORB for today."""
    label: str
    high: float | None = None
    low: float | None = None
    size: float | None = None
    break_dir: str | None = None
    break_ts: datetime | None = None
    complete: bool = False
    outcome: str | None = None  # "win", "loss", None (pending/no trade)


@dataclass
class SessionSignals:
    """Cross-session intelligence derived from today's outcomes."""
    prior_outcomes: dict[str, str] = field(default_factory=dict)
    reversal_active: bool = False
    chop_detected: bool = False
    continuation: bool = False
    cascade_wr: float | None = None


@dataclass
class RegimeContext:
    """Current regime vs full-period performance deltas."""
    label: str = ""
    start_date: date | None = None
    end_date: date | None = None
    deltas: dict[str, float] = field(default_factory=dict)


@dataclass
class MarketState:
    """Single shared object describing current market context.

    This object is a READ-ONLY consumer of the database.
    It never writes to any table.
    """

    # Identity
    trading_day: date | None = None
    current_ts: datetime | None = None

    # ORB snapshots (built progressively through the day)
    orbs: dict[str, OrbSnapshot] = field(default_factory=dict)

    # Cross-session intelligence
    signals: SessionSignals = field(default_factory=SessionSignals)

    # Regime
    regime: RegimeContext | None = None

    # Volatility / momentum
    rsi_14: float | None = None
    session_highs: dict[str, float] = field(default_factory=dict)
    session_lows: dict[str, float] = field(default_factory=dict)

    # Strategy scoring (hypothesis-driven)
    strategy_scores: dict[str, float] = field(default_factory=dict)

    # Portfolio state (updated by execution engine)
    daily_pnl_r: float = 0.0
    daily_trade_count: int = 0
    active_trade_ids: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------

    @classmethod
    def from_trading_day(
        cls,
        trading_day: date,
        db_path: Path | str,
        orb_minutes: int = 5,
        cascade_table: dict | None = None,
        visible_sessions: set[str] | None = None,
    ) -> "MarketState":
        """Build MarketState from DB for a given day (historical replay).

        Loads daily_features for the trading day.
        Does NOT write to the database.

        Args:
            visible_sessions: If provided, only ORBs whose label is in this
                set will have their ``outcome`` populated.  Structural fields
                (high/low/size/complete/break_dir/break_ts) are always
                populated once the ORB window has completed.  Use an empty set
                at replay start and add labels as trade outcomes resolve
                (TP/SL/EOD) to prevent lookahead.  ``None`` (default) means
                all outcomes are visible (post-hoc analysis mode).
        """
        state = cls(trading_day=trading_day)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            # Load daily_features for this day
            row = con.execute("""
                SELECT rsi_14_at_0900,
                       session_asia_high, session_asia_low,
                       session_london_high, session_london_low,
                       session_ny_high, session_ny_low,
                       orb_0900_high, orb_0900_low, orb_0900_size,
                       orb_0900_break_dir, orb_0900_break_ts, orb_0900_outcome,
                       orb_1000_high, orb_1000_low, orb_1000_size,
                       orb_1000_break_dir, orb_1000_break_ts, orb_1000_outcome,
                       orb_1100_high, orb_1100_low, orb_1100_size,
                       orb_1100_break_dir, orb_1100_break_ts, orb_1100_outcome,
                       orb_1800_high, orb_1800_low, orb_1800_size,
                       orb_1800_break_dir, orb_1800_break_ts, orb_1800_outcome,
                       orb_2300_high, orb_2300_low, orb_2300_size,
                       orb_2300_break_dir, orb_2300_break_ts, orb_2300_outcome,
                       orb_0030_high, orb_0030_low, orb_0030_size,
                       orb_0030_break_dir, orb_0030_break_ts, orb_0030_outcome
                FROM daily_features
                WHERE symbol = 'MGC'
                  AND trading_day = ?
                  AND orb_minutes = ?
            """, [trading_day, orb_minutes]).fetchone()

            if row is None:
                return state

            # Unpack columns by position
            state.rsi_14 = row[0]

            # Session highs/lows
            session_names = ["asia", "london", "ny"]
            for i, name in enumerate(session_names):
                high_val = row[1 + i * 2]
                low_val = row[2 + i * 2]
                if high_val is not None:
                    state.session_highs[name] = high_val
                if low_val is not None:
                    state.session_lows[name] = low_val

            # ORB snapshots (6 ORBs, 6 columns each, starting at index 7)
            for j, label in enumerate(ORB_LABELS):
                base = 7 + j * 6
                orb_high = row[base]
                orb_low = row[base + 1]
                orb_size = row[base + 2]
                break_dir = row[base + 3]
                break_ts = row[base + 4]
                outcome = row[base + 5]

                # Mask outcome for sessions not yet resolved during
                # replay (prevents lookahead leak).  Structural fields
                # are always visible once the ORB window completes.
                if visible_sessions is not None and label not in visible_sessions:
                    outcome = None

                state.orbs[label] = OrbSnapshot(
                    label=label,
                    high=orb_high,
                    low=orb_low,
                    size=orb_size,
                    break_dir=break_dir,
                    break_ts=break_ts,
                    complete=orb_high is not None,
                    outcome=outcome,
                )

            # Load regime deltas if regime table exists
            _load_regime_context(con, state)

        finally:
            con.close()

        # Compute cross-session signals
        state.update_signals(cascade_table)

        return state

    # -----------------------------------------------------------------
    # Update methods
    # -----------------------------------------------------------------

    def update_orb(self, label: str, high: float, low: float, size: float,
                   break_dir: str | None = None, break_ts: datetime | None = None,
                   complete: bool = False, outcome: str | None = None) -> None:
        """Update ORB snapshot from live or replay data."""
        self.orbs[label] = OrbSnapshot(
            label=label,
            high=high,
            low=low,
            size=size,
            break_dir=break_dir,
            break_ts=break_ts,
            complete=complete,
            outcome=outcome,
        )

    def update_signals(self, cascade_table: dict | None = None) -> None:
        """Recompute cross-session signals after an ORB outcome is known.

        Detects:
          - reversal: 0900 loss + 1000 opposite-direction break
          - chop: 0900 + 1000 both losses
          - continuation: 0900 win + 1000 same-direction break
        """
        self.signals = SessionSignals()

        # Build prior_outcomes from completed ORBs
        for label in ORB_LABELS:
            orb = self.orbs.get(label)
            if orb and orb.outcome:
                self.signals.prior_outcomes[label] = orb.outcome

        orb_0900 = self.orbs.get("0900")
        orb_1000 = self.orbs.get("1000")

        if orb_0900 and orb_1000 and orb_0900.outcome and orb_1000.break_dir:
            # Determine direction relationship
            same_dir = (
                orb_0900.break_dir is not None
                and orb_0900.break_dir.upper() == orb_1000.break_dir.upper()
            )

            if orb_0900.outcome == "loss" and not same_dir:
                self.signals.reversal_active = True

            if orb_0900.outcome == "loss" and orb_1000.outcome == "loss":
                self.signals.chop_detected = True

            if orb_0900.outcome == "win" and same_dir:
                self.signals.continuation = True

        # Look up cascade win rate from pre-computed table
        if cascade_table and orb_0900 and orb_0900.outcome and orb_1000:
            dir_relation = "same" if (
                orb_0900.break_dir and orb_1000.break_dir
                and orb_0900.break_dir.upper() == orb_1000.break_dir.upper()
            ) else "opposite"
            key = ("0900", orb_0900.outcome, dir_relation)
            entry = cascade_table.get(key)
            if entry:
                self.signals.cascade_wr = entry.get("1000_wr")

    def score_strategies(self, strategies: list, weights=None) -> dict[str, float]:
        """Score each strategy given current context.

        Uses the scoring module for computation. Stores results in
        self.strategy_scores and returns them.

        Args:
            strategies: List of PortfolioStrategy objects
            weights: Optional ScoringWeights override

        Returns:
            {strategy_id: context_score}
        """
        from trading_app.scoring import score_strategy, ScoringWeights

        if weights is None:
            weights = ScoringWeights()

        scores = {}
        for strategy in strategies:
            scores[strategy.strategy_id] = score_strategy(strategy, self, weights)

        self.strategy_scores = scores
        return scores


# =========================================================================
# Internal helpers
# =========================================================================

def _load_regime_context(con: duckdb.DuckDBPyConnection, state: MarketState) -> None:
    """Load regime deltas from regime_strategies if available."""
    try:
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
    except Exception:
        return

    if "regime_strategies" not in tables or "experimental_strategies" not in tables:
        return

    try:
        rows = con.execute("""
            SELECT rs.strategy_id,
                   rs.expectancy_r - es.expectancy_r AS delta_exp_r
            FROM regime_strategies rs
            JOIN experimental_strategies es
              ON rs.strategy_id = es.strategy_id
            WHERE rs.expectancy_r IS NOT NULL
              AND es.expectancy_r IS NOT NULL
        """).fetchall()

        if rows:
            state.regime = RegimeContext(
                label="regime",
                deltas={r[0]: r[1] for r in rows},
            )
    except Exception:
        pass
