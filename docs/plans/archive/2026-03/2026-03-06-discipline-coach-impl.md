---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Discipline Coach MVP — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Build three behavioral components (post-trade debrief, cooling period, pre-session priming) as a pure UI layer on top of the existing trading copilot, with no changes to orchestrator or execution engine.

**Architecture:** Two new files (`ui/discipline_data.py` for JSONL I/O and pattern computation, `ui/discipline.py` for Streamlit UI components) integrated into `ui/copilot.py` at three hook points: after signal log (debriefs), before signal cards (cooling check), and in approaching/alert states (priming). Storage is append-only JSONL in `data/`.

**Tech Stack:** Python 3.13, Streamlit (already in project), JSON/JSONL, pytest

---

### Task 0: Data layer — JSONL I/O and data models

**Files:**
- Create: `ui/discipline_data.py`
- Create: `tests/test_ui/test_discipline_data.py`

**Step 1: Write the failing tests**

```python
# tests/test_ui/test_discipline_data.py
"""Tests for discipline data layer — JSONL I/O, enums, pattern computation."""
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path


def test_adherence_enum_values():
    from ui.discipline_data import ADHERENCE_VALUES
    assert set(ADHERENCE_VALUES) == {"followed", "modified", "overrode", "off_plan"}


def test_deviation_trigger_values():
    from ui.discipline_data import DEVIATION_TRIGGERS
    assert "narrative" in DEVIATION_TRIGGERS
    assert "chasing_loss" in DEVIATION_TRIGGERS
    assert len(DEVIATION_TRIGGERS) == 7  # 6 named + "other"


def test_append_debrief_creates_file(tmp_path):
    from ui.discipline_data import append_debrief
    f = tmp_path / "debriefs.jsonl"
    record = {
        "ts": "2026-03-06T23:15:00Z",
        "trading_day": "2026-03-06",
        "instrument": "MGC",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "adherence": "followed",
        "emotional_temp": 0.5,
    }
    append_debrief(record, path=f)
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["adherence"] == "followed"


def test_append_debrief_appends(tmp_path):
    from ui.discipline_data import append_debrief
    f = tmp_path / "debriefs.jsonl"
    for i in range(3):
        append_debrief({"ts": f"2026-03-0{i+1}T00:00:00Z", "adherence": "followed"}, path=f)
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 3


def test_load_debriefs_empty(tmp_path):
    from ui.discipline_data import load_debriefs
    f = tmp_path / "debriefs.jsonl"
    assert load_debriefs(path=f) == []


def test_load_debriefs_returns_records(tmp_path):
    from ui.discipline_data import load_debriefs, append_debrief
    f = tmp_path / "debriefs.jsonl"
    append_debrief({"ts": "2026-03-06T00:00:00Z", "strategy_id": "X"}, path=f)
    records = load_debriefs(path=f)
    assert len(records) == 1
    assert records[0]["strategy_id"] == "X"


def test_append_discipline_event(tmp_path):
    from ui.discipline_data import append_discipline_event
    f = tmp_path / "state.jsonl"
    append_discipline_event("cooling_triggered", {"tilt_score": 65}, path=f)
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event"] == "cooling_triggered"
    assert "ts" in parsed


def test_get_pending_debriefs_finds_unmatched(tmp_path):
    from ui.discipline_data import get_pending_debriefs
    signals_file = tmp_path / "signals.jsonl"
    debriefs_file = tmp_path / "debriefs.jsonl"
    # Write an exit signal
    signal = {
        "ts": "2026-03-06T23:15:00Z",
        "instrument": "MGC",
        "type": "SIGNAL_EXIT",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "price": 3245.50,
    }
    signals_file.write_text(json.dumps(signal) + "\n")
    pending = get_pending_debriefs(signals_path=signals_file, debriefs_path=debriefs_file)
    assert len(pending) == 1
    assert pending[0]["strategy_id"] == "MGC_CME_REOPEN_E2_CB1_G4_RR2.5"


def test_get_pending_debriefs_excludes_debriefed(tmp_path):
    from ui.discipline_data import get_pending_debriefs, append_debrief
    signals_file = tmp_path / "signals.jsonl"
    debriefs_file = tmp_path / "debriefs.jsonl"
    signal = {
        "ts": "2026-03-06T23:15:00Z",
        "instrument": "MGC",
        "type": "SIGNAL_EXIT",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "price": 3245.50,
    }
    signals_file.write_text(json.dumps(signal) + "\n")
    # Debrief already exists keyed by strategy_id + exit ts
    append_debrief({
        "ts": "2026-03-06T23:16:00Z",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "signal_exit_ts": "2026-03-06T23:15:00Z",
        "adherence": "followed",
    }, path=debriefs_file)
    pending = get_pending_debriefs(signals_path=signals_file, debriefs_path=debriefs_file)
    assert len(pending) == 0


def test_compute_adherence_stats(tmp_path):
    from ui.discipline_data import compute_adherence_stats, append_debrief
    f = tmp_path / "debriefs.jsonl"
    for adh in ["followed", "followed", "overrode"]:
        append_debrief({"adherence": adh, "pnl_r": 1.0 if adh == "followed" else -1.0,
                        "deviation_cost_dollars": 0 if adh == "followed" else 200,
                        "instrument": "MGC", "ts": "2026-03-06T00:00:00Z"}, path=f)
    stats = compute_adherence_stats(path=f)
    assert stats["total"] == 3
    assert stats["followed"] == 2
    assert stats["adherence_rate"] == pytest.approx(2/3)
    assert stats["deviation_cost_dollars"] == 200


def test_get_latest_letter(tmp_path):
    from ui.discipline_data import get_latest_letter, append_debrief
    f = tmp_path / "debriefs.jsonl"
    append_debrief({
        "ts": "2026-03-05T00:00:00Z",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "letter_to_future_self": "Stick to the plan.",
        "adherence": "overrode",
    }, path=f)
    append_debrief({
        "ts": "2026-03-06T00:00:00Z",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "letter_to_future_self": None,
        "adherence": "followed",
    }, path=f)
    letter = get_latest_letter(session="CME_REOPEN", path=f)
    assert letter is not None
    assert letter["text"] == "Stick to the plan."
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui/test_discipline_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.discipline_data'`

**Step 3: Write minimal implementation**

```python
# ui/discipline_data.py
"""Discipline Coach data layer — JSONL I/O, enums, pattern computation.

Append-only JSONL storage for trade debriefs and discipline state events.
No database writes, no schema migrations. Pure file I/O.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

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


# ── JSONL I/O ────────────────────────────────────────────────────────────────


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


def append_discipline_event(
    event_type: str, extra: dict | None = None, *, path: Path = STATE_PATH
) -> None:
    """Append a discipline state event (cooling, commitment, etc.)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **(extra or {}),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ── Signal parsing ───────────────────────────────────────────────────────────


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


def get_pending_debriefs(
    *, signals_path: Path, debriefs_path: Path = DEBRIEFS_PATH
) -> list[dict]:
    """Find exit signals that have no matching debrief.

    Key: (strategy_id, exit_ts) — a debrief matches if its signal_exit_ts
    equals the exit signal's ts.
    """
    signals = _load_signals(signals_path)
    exits = [
        s for s in signals
        if s.get("type") in ("SIGNAL_EXIT", "ORDER_EXIT")
    ]
    if not exits:
        return []

    debriefs = load_debriefs(path=debriefs_path)
    debriefed_keys = {
        (d.get("strategy_id"), d.get("signal_exit_ts"))
        for d in debriefs
    }

    pending = []
    for ex in exits:
        key = (ex.get("strategy_id"), ex.get("ts"))
        if key not in debriefed_keys:
            pending.append(ex)
    return pending


# ── Pattern computation ──────────────────────────────────────────────────────


def compute_adherence_stats(
    *, path: Path = DEBRIEFS_PATH, session: str | None = None
) -> dict:
    """Compute adherence stats from debrief records.

    Returns dict with: total, followed, adherence_rate,
    avg_r_followed, avg_r_deviated, deviation_cost_dollars.
    """
    records = load_debriefs(path=path)
    if session:
        records = [r for r in records if session in r.get("strategy_id", "")]

    if not records:
        return {
            "total": 0, "followed": 0, "adherence_rate": 0.0,
            "avg_r_followed": 0.0, "avg_r_deviated": 0.0,
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


def get_latest_letter(
    session: str, *, path: Path = DEBRIEFS_PATH
) -> dict | None:
    """Get the most recent letter_to_future_self for a session.

    Returns {"text": str, "ts": str, "strategy_id": str} or None.
    """
    records = load_debriefs(path=path)
    letters = [
        r for r in records
        if r.get("letter_to_future_self")
        and session in r.get("strategy_id", "")
    ]
    if not letters:
        return None
    latest = max(letters, key=lambda r: r.get("ts", ""))
    return {
        "text": latest["letter_to_future_self"],
        "ts": latest.get("ts"),
        "strategy_id": latest.get("strategy_id"),
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui/test_discipline_data.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add ui/discipline_data.py tests/test_ui/test_discipline_data.py
git commit -m "feat(discipline): add data layer — JSONL I/O, enums, pattern computation"
```

---

### Task 1: Cooling period state and logic

**Files:**
- Modify: `ui/discipline_data.py`
- Modify: `tests/test_ui/test_discipline_data.py`

**Step 1: Write the failing tests**

Add to `tests/test_ui/test_discipline_data.py`:

```python
def test_trigger_cooling_sets_until(tmp_path):
    from ui.discipline_data import trigger_cooling, is_cooling_active
    state = {}  # simulates st.session_state
    trigger_cooling(state, pnl_r=-1.0, consecutive_losses=2, session_pnl_r=-2.0,
                    state_path=tmp_path / "state.jsonl")
    assert "cooling_until" in state
    assert is_cooling_active(state)


def test_cooling_expires():
    from ui.discipline_data import is_cooling_active
    from datetime import datetime, timezone, timedelta
    state = {"cooling_until": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    assert not is_cooling_active(state)


def test_override_cooling_logs_event(tmp_path):
    from ui.discipline_data import trigger_cooling, override_cooling
    import json
    state_path = tmp_path / "state.jsonl"
    state = {}
    trigger_cooling(state, pnl_r=-1.0, consecutive_losses=1, session_pnl_r=-1.0,
                    state_path=state_path)
    override_cooling(state, state_path=state_path)
    assert "cooling_until" not in state
    lines = state_path.read_text().strip().split("\n")
    events = [json.loads(ln) for ln in lines]
    override_events = [e for e in events if e["event"] == "cooling_overridden"]
    assert len(override_events) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui/test_discipline_data.py::test_trigger_cooling_sets_until -v`
Expected: FAIL — `ImportError: cannot import name 'trigger_cooling'`

**Step 3: Add cooling functions to `ui/discipline_data.py`**

```python
# ── Cooling period ───────────────────────────────────────────────────────────

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
```

Also add `from datetime import timedelta` to the imports.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui/test_discipline_data.py -v`
Expected: All 15 tests PASS

**Step 5: Commit**

```bash
git add ui/discipline_data.py tests/test_ui/test_discipline_data.py
git commit -m "feat(discipline): add cooling period state logic"
```

---

### Task 2: Debrief card UI component

**Files:**
- Create: `ui/discipline.py`
- Create: `tests/test_ui/test_discipline_ui.py`

**Step 1: Write the failing test**

```python
# tests/test_ui/test_discipline_ui.py
"""Tests for discipline UI components — debrief card, cooling screen, priming."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_render_pending_debriefs_no_exits(tmp_path):
    """No exit signals → no debrief cards rendered."""
    from ui.discipline import render_pending_debriefs
    signals_path = tmp_path / "signals.jsonl"
    debriefs_path = tmp_path / "debriefs.jsonl"
    # Empty signals file
    signals_path.write_text("")
    with patch("ui.discipline.st") as mock_st:
        render_pending_debriefs(signals_path=signals_path, debriefs_path=debriefs_path)
        mock_st.form.assert_not_called()


def test_render_pending_debriefs_shows_form(tmp_path):
    """Exit signal without debrief → form rendered."""
    from ui.discipline import render_pending_debriefs
    signals_path = tmp_path / "signals.jsonl"
    debriefs_path = tmp_path / "debriefs.jsonl"
    signal = {
        "ts": "2026-03-06T23:15:00Z",
        "instrument": "MGC",
        "type": "SIGNAL_EXIT",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "price": 3245.50,
    }
    signals_path.write_text(json.dumps(signal) + "\n")
    with patch("ui.discipline.st") as mock_st:
        mock_form = MagicMock()
        mock_st.form.return_value.__enter__ = MagicMock(return_value=mock_form)
        mock_st.form.return_value.__exit__ = MagicMock(return_value=False)
        render_pending_debriefs(signals_path=signals_path, debriefs_path=debriefs_path)
        mock_st.form.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui/test_discipline_ui.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.discipline'`

**Step 3: Write the debrief card UI**

```python
# ui/discipline.py
"""Discipline Coach UI components — debrief card, cooling screen, pre-session priming.

Pure Streamlit rendering. Reads live_signals.jsonl (written by orchestrator),
writes to data/trade_debriefs.jsonl and data/discipline_state.jsonl.
No orchestrator or execution engine changes.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from ui.discipline_data import (
    ADHERENCE_VALUES,
    DEVIATION_TRIGGERS,
    DEBRIEFS_PATH,
    STATE_PATH,
    append_debrief,
    get_pending_debriefs,
    is_cooling_active,
    cooling_remaining_seconds,
    override_cooling,
    trigger_cooling,
    compute_adherence_stats,
    get_latest_letter,
    append_discipline_event,
)

# Signals file — same path as copilot.py
_SIGNALS_FILE = Path(__file__).parent.parent / "live_signals.jsonl"


# ── Debrief card ─────────────────────────────────────────────────────────────


def render_pending_debriefs(
    *,
    signals_path: Path = _SIGNALS_FILE,
    debriefs_path: Path = DEBRIEFS_PATH,
) -> None:
    """Render debrief forms for any exit signals without a matching debrief."""
    pending = get_pending_debriefs(
        signals_path=signals_path, debriefs_path=debriefs_path
    )
    if not pending:
        return

    st.markdown("**Post-Trade Debrief**")

    for exit_signal in pending:
        strategy_id = exit_signal.get("strategy_id", "unknown")
        exit_ts = exit_signal.get("ts", "")
        exit_price = exit_signal.get("price", "")
        instrument = exit_signal.get("instrument", "")

        form_key = f"debrief_{strategy_id}_{exit_ts}"

        with st.form(key=form_key):
            st.markdown(f"**{strategy_id}** exited @ {exit_price}")

            # Layer 2: Adherence classification
            adherence = st.radio(
                "How did you execute?",
                options=list(ADHERENCE_VALUES),
                format_func=lambda x: x.replace("_", " ").title(),
                horizontal=True,
                key=f"adh_{form_key}",
            )

            # Layer 3: Deviation trigger (conditional — shown always in form,
            # but only saved when adherence != followed)
            deviation_trigger = st.selectbox(
                "What caused the deviation?",
                options=[None] + list(DEVIATION_TRIGGERS),
                format_func=lambda x: (x or "—").replace("_", " ").title(),
                key=f"dev_{form_key}",
            )

            # Layer 4: Emotional temperature
            emotional_temp = st.slider(
                "Emotional temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="0.0 = calm, 1.0 = hot",
                key=f"emo_{form_key}",
            )

            # Layer 5: Letter to future self (conditional)
            letter = st.text_area(
                "Letter to future self (optional)",
                placeholder="What do you want to remind yourself next time?",
                key=f"letter_{form_key}",
            )

            submitted = st.form_submit_button("Save Debrief")

            if submitted:
                record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "trading_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "instrument": instrument,
                    "strategy_id": strategy_id,
                    "signal_exit_ts": exit_ts,
                    "exit_price": exit_price,
                    "adherence": adherence,
                    "deviation_trigger": deviation_trigger if adherence != "followed" else None,
                    "emotional_temp": emotional_temp,
                    "letter_to_future_self": letter if letter else None,
                }
                append_debrief(record, path=debriefs_path)
                st.success("Debrief saved.")
                st.rerun()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui/test_discipline_ui.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add ui/discipline.py tests/test_ui/test_discipline_ui.py
git commit -m "feat(discipline): add debrief card UI component"
```

---

### Task 3: Cooling period UI component

**Files:**
- Modify: `ui/discipline.py`
- Modify: `tests/test_ui/test_discipline_ui.py`

**Step 1: Write the failing test**

Add to `tests/test_ui/test_discipline_ui.py`:

```python
def test_check_cooling_returns_false_when_not_active():
    from ui.discipline import check_cooling
    with patch("ui.discipline.st") as mock_st:
        mock_st.session_state = {}
        assert check_cooling() is False


def test_check_cooling_returns_true_when_active():
    from ui.discipline import check_cooling
    from datetime import datetime, timezone, timedelta
    with patch("ui.discipline.st") as mock_st:
        until = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
        mock_st.session_state = {"cooling_until": until, "cooling_mode": "hard"}
        assert check_cooling() is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui/test_discipline_ui.py::test_check_cooling_returns_false_when_not_active -v`
Expected: FAIL — `ImportError: cannot import name 'check_cooling'`

**Step 3: Add cooling UI to `ui/discipline.py`**

```python
# ── Cooling period ───────────────────────────────────────────────────────────

_TRADING_QUOTES = [
    "The goal of a successful trader is to make the best trades. Money is secondary.",
    "It's not whether you're right or wrong, but how much you make when right and lose when wrong.",
    "The market can stay irrational longer than you can stay solvent.",
    "In trading, the impossible happens about twice a year.",
    "The elements of good trading are: cutting losses, cutting losses, and cutting losses.",
    "Trade what you see, not what you think.",
    "Discipline is the bridge between goals and accomplishment.",
    "The best trade is the one you didn't take.",
]


def check_cooling(
    *,
    state_path: Path = STATE_PATH,
) -> bool:
    """Check if cooling period is active. If active, render cooling screen.

    Returns True if cooling is active (caller should skip signal rendering).
    """
    if not is_cooling_active(st.session_state):
        return False

    remaining = cooling_remaining_seconds(st.session_state)
    mode = st.session_state.get("cooling_mode", "hard")

    # Progress bar
    progress = 1.0 - (remaining / 90.0)
    st.progress(min(progress, 1.0), text=f"Cooling: {int(remaining)}s remaining")

    # Cooling content
    st.markdown("**Take a breath.**")
    quote = random.choice(_TRADING_QUOTES)
    st.markdown(f"*\"{quote}\"*")
    st.caption("Wait for the signal. The plan is the edge.")

    # Soft mode: override button after 15s
    if mode == "soft" and remaining < 75:  # 90 - 15 = 75
        if st.button("Override cooling", type="secondary"):
            override_cooling(st.session_state, state_path=state_path)
            st.rerun()

    return mode == "hard" or remaining >= 75  # soft mode: block for first 15s only via return


def render_cooling_settings() -> None:
    """Render cooling mode toggle in sidebar."""
    current = st.session_state.get("cooling_mode", "hard")
    mode = st.radio(
        "Cooling mode",
        options=["hard", "soft"],
        index=0 if current == "hard" else 1,
        format_func=lambda x: f"{x.title()} (90s {'non-dismissable' if x == 'hard' else 'dismissable after 15s'})",
        key="cooling_mode_radio",
        horizontal=True,
    )
    st.session_state["cooling_mode"] = mode
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui/test_discipline_ui.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add ui/discipline.py tests/test_ui/test_discipline_ui.py
git commit -m "feat(discipline): add cooling period UI — hard/soft modes, override, quotes"
```

---

### Task 4: Pre-session priming UI component

**Files:**
- Modify: `ui/discipline.py`
- Modify: `tests/test_ui/test_discipline_ui.py`

**Step 1: Write the failing test**

Add to `tests/test_ui/test_discipline_ui.py`:

```python
def test_render_pre_session_priming_shows_commitment(tmp_path):
    from ui.discipline import render_pre_session_priming
    debriefs_path = tmp_path / "debriefs.jsonl"
    state_path = tmp_path / "state.jsonl"
    with patch("ui.discipline.st") as mock_st:
        mock_st.session_state = {}
        mock_st.button.return_value = False
        render_pre_session_priming(
            session="CME_REOPEN",
            strategies=[],
            debriefs_path=debriefs_path,
            state_path=state_path,
        )
        # Should render commitment button
        mock_st.button.assert_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ui/test_discipline_ui.py::test_render_pre_session_priming_shows_commitment -v`
Expected: FAIL — `ImportError: cannot import name 'render_pre_session_priming'`

**Step 3: Add priming UI to `ui/discipline.py`**

```python
# ── Pre-session priming ──────────────────────────────────────────────────────


def render_pre_session_priming(
    *,
    session: str,
    strategies: list,
    debriefs_path: Path = DEBRIEFS_PATH,
    state_path: Path = STATE_PATH,
) -> None:
    """Render pre-session priming card: stats, plan, commitment, letter from past self.

    Args:
        session: Session name (e.g. "CME_REOPEN")
        strategies: List of PortfolioStrategy for this session
    """
    st.markdown("**Pre-Session Priming**")

    # Pattern stats
    stats = compute_adherence_stats(path=debriefs_path, session=session)
    if stats["total"] > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Adherence", f"{stats['adherence_rate']:.0%}",
                       help=f"{stats['followed']}/{stats['total']} signals followed")
        with col2:
            st.metric("Avg R (followed)", f"{stats['avg_r_followed']:+.2f}")
        with col3:
            st.metric("Deviation cost", f"${stats['deviation_cost_dollars']:,.0f}")
    else:
        st.caption("No debrief history yet for this session.")

    # Today's plan
    if strategies:
        st.markdown("**Today's Plan**")
        for s in strategies:
            st.markdown(
                f"- **{s.instrument}** {s.entry_model} CB{s.confirm_bars} "
                f"{s.filter_type} RR{s.rr_target} ({s.orb_minutes}m ORB)"
            )
        st.caption("Action rule: Execute within 60s of signal.")

    # Commitment button
    committed_key = f"committed_{session}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    already_committed = st.session_state.get(committed_key, False)

    if already_committed:
        st.success("Committed to the plan.")
    else:
        if st.button("I commit to following the plan", type="primary", key=f"commit_{session}"):
            st.session_state[committed_key] = True
            append_discipline_event(
                "commitment",
                {"session": session},
                path=state_path,
            )
            st.rerun()

    # Letter from past self
    letter = get_latest_letter(session=session, path=debriefs_path)
    if letter:
        st.markdown("---")
        st.markdown("**Letter from your past self:**")
        st.info(f"\"{letter['text']}\"")
        ts_str = letter.get("ts", "")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                st.caption(f"Written {dt.strftime('%b %d')} after {letter.get('strategy_id', '')}")
            except ValueError:
                pass
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ui/test_discipline_ui.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add ui/discipline.py tests/test_ui/test_discipline_ui.py
git commit -m "feat(discipline): add pre-session priming — stats, plan, commitment, letters"
```

---

### Task 5: Integration into copilot.py

**Files:**
- Modify: `ui/copilot.py:1-10` (imports)
- Modify: `ui/copilot.py:170-222` (approaching/alert — add priming call)
- Modify: `ui/copilot.py:487-518` (render — add debrief + cooling)

**Step 1: Add imports to `ui/copilot.py`**

After the existing imports (line ~44), add:

```python
from ui.discipline import (
    render_pending_debriefs,
    check_cooling,
    render_pre_session_priming,
    render_cooling_settings,
)
```

**Step 2: Add priming to `_render_approaching()` and `_render_alert()`**

In `_render_approaching()` (around line 189), before `_render_briefing_cards(...)`, add:

```python
    # Pre-session priming
    briefings = _cached_briefings()
    session_strategies = [b for b in briefings if b.session == state.next_session]
    render_pre_session_priming(session=state.next_session, strategies=session_strategies)
    st.divider()
```

Same in `_render_alert()` (around line 217), before `_render_briefing_cards(...)`.

**Step 3: Add cooling check + debrief to `render()`**

In `render()` (around line 507), modify the signal log section:

```python
    # Signal log (always, if running)
    proc = st.session_state.get("live_proc")
    if proc is not None and proc.poll() is None:
        st.divider()
        # Cooling check — may block signal rendering in hard mode
        cooling_active = check_cooling()
        if not cooling_active:
            _render_signal_log()
        # Debrief cards for any unprocessed exits
        render_pending_debriefs()
```

**Step 4: Add cooling settings to sidebar**

In `render()`, before the state-dependent main area (around line 494), add:

```python
    # Sidebar — discipline settings
    with st.sidebar:
        render_cooling_settings()
```

**Step 5: Run full test suite**

Run: `pytest tests/test_ui/ -v`
Expected: All tests PASS (existing + new)

**Step 6: Commit**

```bash
git add ui/copilot.py
git commit -m "feat(discipline): integrate debrief, cooling, priming into copilot"
```

---

### Task 6: Cooling auto-trigger from exit signals

**Files:**
- Modify: `ui/discipline.py`
- Modify: `tests/test_ui/test_discipline_ui.py`

The debrief card already renders after exits, but cooling needs to auto-trigger when a losing exit is detected. This hooks into `render_pending_debriefs()`.

**Step 1: Write the failing test**

```python
def test_losing_exit_triggers_cooling(tmp_path):
    """A negative pnl exit signal should activate cooling."""
    from ui.discipline import render_pending_debriefs
    import json
    signals_path = tmp_path / "signals.jsonl"
    debriefs_path = tmp_path / "debriefs.jsonl"
    state_path = tmp_path / "state.jsonl"
    # Write an entry then exit with negative implied P&L
    entry = {
        "ts": "2026-03-06T23:00:00Z", "instrument": "MGC",
        "type": "SIGNAL_ENTRY", "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "price": 3250.0,
    }
    exit_sig = {
        "ts": "2026-03-06T23:15:00Z", "instrument": "MGC",
        "type": "SIGNAL_EXIT", "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "price": 3245.0, "pnl_r": -1.2,
    }
    signals_path.write_text(json.dumps(entry) + "\n" + json.dumps(exit_sig) + "\n")
    with patch("ui.discipline.st") as mock_st:
        mock_st.session_state = {}
        mock_st.form.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_st.form.return_value.__exit__ = MagicMock(return_value=False)
        render_pending_debriefs(
            signals_path=signals_path,
            debriefs_path=debriefs_path,
            state_path=state_path,
        )
        # Cooling should have been triggered
        assert "cooling_until" in mock_st.session_state
```

**Step 2: Modify `render_pending_debriefs()` to accept `state_path` and trigger cooling**

At the top of `render_pending_debriefs()`, after finding pending exits, check if any have negative `pnl_r` and trigger cooling if not already active:

```python
def render_pending_debriefs(
    *,
    signals_path: Path = _SIGNALS_FILE,
    debriefs_path: Path = DEBRIEFS_PATH,
    state_path: Path = STATE_PATH,
) -> None:
    # ... existing code to find pending ...

    # Auto-trigger cooling on losing exits (if not already cooling)
    for ex in pending:
        pnl_r = ex.get("pnl_r")
        if pnl_r is not None and pnl_r < 0 and not is_cooling_active(st.session_state):
            trigger_cooling(
                st.session_state,
                pnl_r=pnl_r,
                consecutive_losses=1,  # simplified for MVP
                session_pnl_r=pnl_r,
                state_path=state_path,
            )
            break  # one trigger per render cycle
```

**Step 3: Run tests**

Run: `pytest tests/test_ui/test_discipline_ui.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add ui/discipline.py tests/test_ui/test_discipline_ui.py
git commit -m "feat(discipline): auto-trigger cooling on losing exits"
```

---

### Task 7: Manual smoke test + drift check

**Files:** No new files

**Step 1: Run drift check**

```bash
python pipeline/check_drift.py
```
Expected: All checks PASS

**Step 2: Run full test suite**

```bash
pytest tests/ -x -q
```
Expected: All tests PASS, no regressions

**Step 3: Manual smoke test (optional — requires Streamlit)**

```bash
streamlit run ui/app.py
```

Verify:
- Sidebar shows cooling mode toggle (hard/soft)
- No errors on page load
- If `live_signals.jsonl` has exit records, debrief cards appear
- APPROACHING/ALERT states show priming card

**Step 4: Final commit with all files**

```bash
git add -A
git commit -m "feat: Discipline Coach MVP — debrief, cooling, priming

Three behavioral components integrated into trading copilot:
- Post-trade debrief card (adherence, deviation trigger, emotional temp, letter)
- Cooling period (hard/soft mode, 90s, override logging)
- Pre-session priming (adherence stats, plan, commitment, past letters)"
```

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `ui/discipline_data.py` | CREATE | JSONL I/O, enums, pattern computation, cooling state |
| `ui/discipline.py` | CREATE | Streamlit UI: debrief card, cooling screen, priming |
| `ui/copilot.py` | MODIFY | Import + 3 integration hooks (debrief, cooling, priming) |
| `tests/test_ui/test_discipline_data.py` | CREATE | 15 tests for data layer |
| `tests/test_ui/test_discipline_ui.py` | CREATE | 6 tests for UI components |
| `data/trade_debriefs.jsonl` | RUNTIME | Created on first debrief save |
| `data/discipline_state.jsonl` | RUNTIME | Created on first cooling/commitment |
