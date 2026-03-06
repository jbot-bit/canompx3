"""Discipline Coach data layer — JSONL I/O, enums, pattern computation.

Append-only JSONL storage for trade debriefs and discipline state events.
No database writes, no schema migrations. Pure file I/O.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
DEBRIEFS_PATH = _DATA_DIR / "trade_debriefs.jsonl"
STATE_PATH = _DATA_DIR / "discipline_state.jsonl"

ADHERENCE_VALUES = ("followed", "modified", "overrode", "off_plan")

DEVIATION_TRIGGERS = (
    "chart_pattern",
    "narrative",
    "felt_reversal",
    "chasing_loss",
    "fomo_late",
    "sized_up",
    "other",
)


def append_debrief(record: dict, *, path: Path = DEBRIEFS_PATH) -> None:
    """Append a debrief record to the JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_debriefs(*, path: Path = DEBRIEFS_PATH) -> list[dict]:
    """Load all debrief records from JSONL. Returns [] if file missing."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def append_discipline_event(event_type: str, extra: dict | None = None, *, path: Path = STATE_PATH) -> None:
    """Append a discipline state event (cooling, commitment, etc.)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **(extra or {}),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _load_signals(signals_path: Path) -> list[dict]:
    """Load signal records from live_signals.jsonl."""
    if not signals_path.exists():
        return []
    records = []
    for line in signals_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def get_pending_debriefs(*, signals_path: Path, debriefs_path: Path = DEBRIEFS_PATH) -> list[dict]:
    """Find exit signals that have no matching debrief.

    Key: (strategy_id, exit_ts) — a debrief matches if its signal_exit_ts
    equals the exit signal's ts.
    """
    signals = _load_signals(signals_path)
    exits = [s for s in signals if s.get("type") in ("SIGNAL_EXIT", "ORDER_EXIT")]
    if not exits:
        return []

    debriefs = load_debriefs(path=debriefs_path)
    debriefed_keys = {(d.get("strategy_id"), d.get("signal_exit_ts")) for d in debriefs}

    pending = []
    for ex in exits:
        key = (ex.get("strategy_id"), ex.get("ts"))
        if key not in debriefed_keys:
            pending.append(ex)
    return pending


def compute_adherence_stats(*, path: Path = DEBRIEFS_PATH, session: str | None = None) -> dict:
    """Compute adherence stats from debrief records."""
    records = load_debriefs(path=path)
    if session:
        records = [r for r in records if session in r.get("strategy_id", "")]

    if not records:
        return {
            "total": 0,
            "followed": 0,
            "adherence_rate": 0.0,
            "avg_r_followed": 0.0,
            "avg_r_deviated": 0.0,
            "deviation_cost_dollars": 0.0,
        }

    followed = [r for r in records if r.get("adherence") == "followed"]
    deviated = [r for r in records if r.get("adherence") in ("modified", "overrode", "off_plan")]

    followed_rs = [r.get("pnl_r", 0) or 0 for r in followed]
    deviated_rs = [r.get("pnl_r", 0) or 0 for r in deviated]
    dev_cost = sum(r.get("deviation_cost_dollars", 0) or 0 for r in deviated)

    return {
        "total": len(records),
        "followed": len(followed),
        "adherence_rate": len(followed) / len(records) if records else 0.0,
        "avg_r_followed": sum(followed_rs) / len(followed_rs) if followed_rs else 0.0,
        "avg_r_deviated": sum(deviated_rs) / len(deviated_rs) if deviated_rs else 0.0,
        "deviation_cost_dollars": dev_cost,
    }


def get_latest_letter(session: str, *, path: Path = DEBRIEFS_PATH) -> dict | None:
    """Get the most recent letter_to_future_self for a session."""
    records = load_debriefs(path=path)
    letters = [r for r in records if r.get("letter_to_future_self") and session in r.get("strategy_id", "")]
    if not letters:
        return None
    latest = max(letters, key=lambda r: r.get("ts", ""))
    return {
        "text": latest["letter_to_future_self"],
        "ts": latest.get("ts"),
        "strategy_id": latest.get("strategy_id"),
    }


# -- Cooling period --------------------------------------------------------

COOLING_SECONDS = 90


def trigger_cooling(
    session_state: dict,
    *,
    pnl_r: float,
    consecutive_losses: int,
    session_pnl_r: float,
    state_path: Path = STATE_PATH,
) -> None:
    """Activate cooling period after a losing trade."""
    until = datetime.now(timezone.utc) + timedelta(seconds=COOLING_SECONDS)
    session_state["cooling_until"] = until.isoformat()
    append_discipline_event(
        "cooling_triggered",
        {
            "pnl_r": pnl_r,
            "consecutive_losses": consecutive_losses,
            "session_pnl_r": session_pnl_r,
            "cooldown_seconds": COOLING_SECONDS,
        },
        path=state_path,
    )


def is_cooling_active(session_state: dict) -> bool:
    """Check if cooling period is still active."""
    until_str = session_state.get("cooling_until")
    if not until_str:
        return False
    until = datetime.fromisoformat(until_str)
    return datetime.now(timezone.utc) < until


def cooling_remaining_seconds(session_state: dict) -> float:
    """Seconds remaining in cooling period. 0 if not active."""
    until_str = session_state.get("cooling_until")
    if not until_str:
        return 0.0
    until = datetime.fromisoformat(until_str)
    remaining = (until - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, remaining)


def override_cooling(session_state: dict, *, state_path: Path = STATE_PATH) -> None:
    """Override cooling period (soft mode). Logs the override event."""
    remaining = cooling_remaining_seconds(session_state)
    session_state.pop("cooling_until", None)
    append_discipline_event(
        "cooling_overridden",
        {"remaining_seconds": round(remaining, 1)},
        path=state_path,
    )
