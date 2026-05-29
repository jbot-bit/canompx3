#!/usr/bin/env python3
"""Track D MNQ COMEX_SETTLE Gate 0 microstructure validation runner.

This runner is intentionally bounded to the preregistered Track D family:
MNQ COMEX_SETTLE O5 E2 RR1.5 CB1, historical MBP-1 only. It writes all
microstructure state to a sidecar DuckDB under research/data and never mutates
canonical gold.db, validated_setups, allocation files, live state, or
paper_trades.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.paths import GOLD_DB_PATH, PROJECT_ROOT

DATASET = "GLBX.MDP3"
DATABENTO_SYMBOL = "MNQ.FUT"
STYPE_IN = "parent"
SCHEMA = "mbp-1"
HYPOTHESIS_SLUG = "mnq_comex_settle_gate0_microstructure_v1"
SIDE_PREFIX = "track_d_gate0"
DEFAULT_DRY_RUN_COST_SAMPLE_SIZE = 5
DEFAULT_SIDECAR_DB = PROJECT_ROOT / "research" / "data" / SIDE_PREFIX / "track_d_gate0.duckdb"
DEFAULT_DATA_DIR = PROJECT_ROOT / "research" / "data" / SIDE_PREFIX / SCHEMA
RESULT_DOC = PROJECT_ROOT / "docs" / "audit" / "results" / "2026-05-29-track-d-gate0-mbp1-results.md"
HOLDOUT_DATE = pd.Timestamp("2026-01-01").date()
LOOKBACK_SECONDS = 60
POST_TOUCH_BUFFER_SECONDS = 16
EXPECTED_TOTAL = 1741
EXPECTED_IS = 1658
EXPECTED_OOS = 83


@dataclass(frozen=True)
class Gate0Window:
    window_id: str
    hypothesis_slug: str
    trading_day: str
    symbol: str
    databento_symbol: str
    stype_in: str
    schema_used: str
    orb_label: str
    orb_minutes: int
    entry_model: str
    rr_target: float
    confirm_bars: int
    break_dir: str
    entry_ts: datetime
    window_start_utc: datetime
    window_end_utc: datetime
    lookback_seconds: int
    post_touch_buffer_seconds: int
    pnl_r: float
    source_dbn_path: str | None
    metadata_cost_usd: float | None
    downloaded_at_utc: datetime | None
    git_sha: str
    created_at_utc: datetime


@dataclass(frozen=True)
class Gate0Feature:
    feature_row_id: str
    window_id: str
    trading_day: str
    symbol: str
    orb_label: str
    orb_minutes: int
    entry_model: str
    rr_target: float
    confirm_bars: int
    break_dir: str
    entry_ts: datetime
    lookback_seconds: int
    signed_ofi_60s: float | None
    signed_tbi_60s: float | None
    signed_qi_last_1s: float | None
    signed_qi_mean_10s: float | None
    spread_mean_ticks_10s: float | None
    spread_max_ticks_10s: float | None
    event_count_mbp1: int
    event_count_trade: int
    feature_version: str


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_utc_datetime(value: object) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


def _none_if_missing(value: object) -> object | None:
    return None if value is None or pd.isna(value) else value


def _window_id(trading_day: str, schema_used: str = SCHEMA) -> str:
    raw = f"{HYPOTHESIS_SLUG}|{trading_day}|{schema_used}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _feature_row_id(window_id: str) -> str:
    return hashlib.sha256(f"{window_id}|features|v1".encode()).hexdigest()[:24]


def _result_id(feature_id: str, split: str) -> str:
    return hashlib.sha256(f"{HYPOTHESIS_SLUG}|{feature_id}|{split}".encode()).hexdigest()[:24]


def build_window_from_record(record: dict, *, sha: str | None = None, created_at: datetime | None = None) -> Gate0Window:
    entry_ts = record.get("entry_ts")
    if entry_ts is None or pd.isna(entry_ts):
        raise ValueError("entry_ts is required for Track D Gate 0 windows")
    break_dir = str(record.get("break_dir") or "").lower()
    if break_dir not in {"long", "short"}:
        raise ValueError(f"break_dir must be long or short, got {break_dir!r}")

    entry_utc = _to_utc_datetime(entry_ts)
    trading_day = str(pd.Timestamp(record["trading_day"]).date())
    start = entry_utc - timedelta(seconds=LOOKBACK_SECONDS)
    end = entry_utc + timedelta(seconds=POST_TOUCH_BUFFER_SECONDS)
    if not start < entry_utc < end:
        raise ValueError("invalid Track D feature window bounds")

    return Gate0Window(
        window_id=_window_id(trading_day),
        hypothesis_slug=HYPOTHESIS_SLUG,
        trading_day=trading_day,
        symbol="MNQ",
        databento_symbol=DATABENTO_SYMBOL,
        stype_in=STYPE_IN,
        schema_used=SCHEMA,
        orb_label="COMEX_SETTLE",
        orb_minutes=5,
        entry_model="E2",
        rr_target=1.5,
        confirm_bars=1,
        break_dir=break_dir,
        entry_ts=entry_utc,
        window_start_utc=start,
        window_end_utc=end,
        lookback_seconds=LOOKBACK_SECONDS,
        post_touch_buffer_seconds=POST_TOUCH_BUFFER_SECONDS,
        pnl_r=float(record["pnl_r"]),
        source_dbn_path=_none_if_missing(record.get("source_dbn_path")),
        metadata_cost_usd=_none_if_missing(record.get("metadata_cost_usd")),
        downloaded_at_utc=(
            _to_utc_datetime(record.get("downloaded_at_utc"))
            if _none_if_missing(record.get("downloaded_at_utc")) is not None
            else None
        ),
        git_sha=sha or git_sha(),
        created_at_utc=created_at or _utc_now(),
    )


def load_manifest_rows(gold_db_path: Path = GOLD_DB_PATH) -> list[Gate0Window]:
    sql = """
        SELECT
            o.trading_day,
            o.entry_ts,
            o.pnl_r,
            d.orb_COMEX_SETTLE_break_dir AS break_dir
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.symbol = o.symbol
         AND d.trading_day = o.trading_day
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol = 'MNQ'
          AND o.orb_label = 'COMEX_SETTLE'
          AND o.orb_minutes = 5
          AND o.entry_model = 'E2'
          AND o.rr_target = 1.5
          AND o.confirm_bars = 1
          AND o.pnl_r IS NOT NULL
          AND o.entry_ts IS NOT NULL
        ORDER BY o.trading_day
    """
    con = duckdb.connect(str(gold_db_path), read_only=True)
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    sha = git_sha()
    now = _utc_now()
    return [build_window_from_record(row.to_dict(), sha=sha, created_at=now) for _, row in df.iterrows()]


def summarize_manifest(rows: list[Gate0Window]) -> dict[str, object]:
    is_count = sum(pd.Timestamp(r.trading_day).date() < HOLDOUT_DATE for r in rows)
    oos_count = len(rows) - is_count
    return {
        "total": len(rows),
        "is": is_count,
        "oos": oos_count,
        "min_day": min((r.trading_day for r in rows), default=None),
        "max_day": max((r.trading_day for r in rows), default=None),
    }


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS micro_gate0_windows (
            window_id TEXT PRIMARY KEY,
            hypothesis_slug TEXT NOT NULL,
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            databento_symbol TEXT NOT NULL,
            stype_in TEXT NOT NULL,
            schema_used TEXT NOT NULL,
            orb_label TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            entry_model TEXT NOT NULL,
            rr_target DOUBLE NOT NULL,
            confirm_bars INTEGER NOT NULL,
            break_dir TEXT NOT NULL,
            entry_ts TIMESTAMPTZ NOT NULL,
            window_start_utc TIMESTAMPTZ NOT NULL,
            window_end_utc TIMESTAMPTZ NOT NULL,
            lookback_seconds INTEGER NOT NULL,
            post_touch_buffer_seconds INTEGER NOT NULL,
            pnl_r DOUBLE NOT NULL,
            source_dbn_path TEXT,
            metadata_cost_usd DOUBLE,
            downloaded_at_utc TIMESTAMPTZ,
            git_sha TEXT NOT NULL,
            created_at_utc TIMESTAMPTZ NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS micro_mbp1_events (
            window_id TEXT NOT NULL,
            ts_event TIMESTAMPTZ NOT NULL,
            ts_recv TIMESTAMPTZ,
            instrument_id BIGINT,
            symbol TEXT,
            action TEXT,
            side TEXT,
            price DOUBLE,
            size DOUBLE,
            bid_px_00 DOUBLE,
            ask_px_00 DOUBLE,
            bid_sz_00 DOUBLE,
            ask_sz_00 DOUBLE,
            bid_ct_00 DOUBLE,
            ask_ct_00 DOUBLE,
            sequence BIGINT,
            flags BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS micro_gate0_features (
            feature_row_id TEXT PRIMARY KEY,
            window_id TEXT NOT NULL,
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_label TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            entry_model TEXT NOT NULL,
            rr_target DOUBLE NOT NULL,
            confirm_bars INTEGER NOT NULL,
            break_dir TEXT NOT NULL,
            entry_ts TIMESTAMPTZ NOT NULL,
            lookback_seconds INTEGER NOT NULL,
            signed_ofi_60s DOUBLE,
            signed_tbi_60s DOUBLE,
            signed_qi_last_1s DOUBLE,
            signed_qi_mean_10s DOUBLE,
            spread_mean_ticks_10s DOUBLE,
            spread_max_ticks_10s DOUBLE,
            event_count_mbp1 INTEGER,
            event_count_trade INTEGER,
            feature_version TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS micro_gate0_results (
            result_id TEXT PRIMARY KEY,
            hypothesis_slug TEXT NOT NULL,
            feature_id TEXT NOT NULL,
            sample_split TEXT NOT NULL,
            n_parent INTEGER NOT NULL,
            n_selected INTEGER NOT NULL,
            parent_policy_ev_r DOUBLE NOT NULL,
            selected_policy_ev_r DOUBLE NOT NULL,
            selected_trade_mean_r DOUBLE,
            delta_policy_ev_r DOUBLE,
            threshold_value DOUBLE,
            decision TEXT NOT NULL,
            created_at_utc TIMESTAMPTZ NOT NULL,
            git_sha TEXT NOT NULL
        )
        """
    )


def write_manifest(rows: list[Gate0Window], sidecar_db: Path) -> None:
    sidecar_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(sidecar_db))
    try:
        create_schema(con)
        df = pd.DataFrame([asdict(r) for r in rows])
        con.execute("DELETE FROM micro_gate0_windows WHERE hypothesis_slug = ? AND schema_used = ?", [HYPOTHESIS_SLUG, SCHEMA])
        con.register("manifest_df", df)
        con.execute("INSERT INTO micro_gate0_windows SELECT * FROM manifest_df")
    finally:
        con.close()


def read_windows(sidecar_db: Path) -> list[Gate0Window]:
    con = duckdb.connect(str(sidecar_db), read_only=True)
    try:
        df = con.execute("SELECT * FROM micro_gate0_windows ORDER BY trading_day").fetchdf()
    finally:
        con.close()
    return [build_window_from_record(row.to_dict(), sha=str(row["git_sha"]), created_at=_to_utc_datetime(row["created_at_utc"])) for _, row in df.iterrows()]


def _get_databento_client():
    try:
        import databento as db
    except ImportError as exc:
        raise ImportError("databento package required") from exc

    api_key = os.getenv("DATABENTO_API_KEY") or _load_databento_key_from_env_files()
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not found in environment")
    return db.Historical(api_key)


def _load_databento_key_from_env_files() -> str | None:
    candidates = [PROJECT_ROOT / ".env"]
    try:
        common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=PROJECT_ROOT, text=True).strip()
        common_path = Path(common)
        if not common_path.is_absolute():
            common_path = (PROJECT_ROOT / common_path).resolve()
        candidates.append(common_path.parent / ".env")
    except Exception:
        pass

    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "DATABENTO_API_KEY":
                token = value.strip().strip("\"'")
                if token:
                    os.environ.setdefault("DATABENTO_API_KEY", token)
                    return token
    return None


def _iso_z(ts: datetime) -> str:
    return _to_utc_datetime(ts).isoformat().replace("+00:00", "Z")


def _stratified_sample(rows: list[Gate0Window], n: int = DEFAULT_DRY_RUN_COST_SAMPLE_SIZE) -> list[Gate0Window]:
    if len(rows) <= n:
        return rows
    idx = {0, len(rows) - 1}
    idx.update(round(i * (len(rows) - 1) / (n - 1)) for i in range(n))
    return [rows[i] for i in sorted(idx)]


def estimate_cost_rows(rows: Iterable[Gate0Window], client: object) -> dict[str, float]:
    costs: dict[str, float] = {}
    for row in rows:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=DATABENTO_SYMBOL,
            schema=SCHEMA,
            stype_in=STYPE_IN,
            start=_iso_z(row.window_start_utc),
            end=_iso_z(row.window_end_utc),
        )
        costs[row.window_id] = float(cost)
    return costs


def update_costs(sidecar_db: Path, costs: dict[str, float]) -> None:
    con = duckdb.connect(str(sidecar_db))
    try:
        create_schema(con)
        for window_id, cost in costs.items():
            con.execute("UPDATE micro_gate0_windows SET metadata_cost_usd = ? WHERE window_id = ?", [cost, window_id])
    finally:
        con.close()


def cache_path_for_window(row: Gate0Window, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return data_dir / f"{row.trading_day}_{row.window_id}.{SCHEMA}.dbn.zst"


def pull_pending(sidecar_db: Path, *, max_cost_usd: float | None, yes: bool, dry_run: bool = False) -> dict[str, object]:
    if max_cost_usd is None:
        raise ValueError("--pull requires --max-cost-usd")
    if not yes:
        raise ValueError("--pull requires --yes")
    rows = read_windows(sidecar_db)
    missing_costs = [r.window_id for r in rows if r.metadata_cost_usd is None]
    if missing_costs:
        raise ValueError(f"{len(missing_costs)} windows missing metadata_cost_usd; run --estimate-cost first")
    pending = [r for r in rows if not cache_path_for_window(r).exists()]
    pending_cost = sum(float(r.metadata_cost_usd or 0.0) for r in pending)
    if pending_cost > max_cost_usd:
        raise ValueError(f"pending MBP-1 cost ${pending_cost:.2f} exceeds cap ${max_cost_usd:.2f}")
    if dry_run:
        return {"pending": len(pending), "pending_cost_usd": pending_cost}

    client = _get_databento_client()
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(sidecar_db))
    try:
        for row in pending:
            path = cache_path_for_window(row)
            client.timeseries.get_range(
                dataset=DATASET,
                symbols=DATABENTO_SYMBOL,
                schema=SCHEMA,
                stype_in=STYPE_IN,
                start=_iso_z(row.window_start_utc),
                end=_iso_z(row.window_end_utc),
                path=str(path),
            )
            con.execute(
                "UPDATE micro_gate0_windows SET source_dbn_path = ?, downloaded_at_utc = ? WHERE window_id = ?",
                [str(path), _utc_now(), row.window_id],
            )
    finally:
        con.close()
    return {"pulled": len(pending), "cost_usd": pending_cost}


def _normalise_event_df(df: pd.DataFrame, window_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if "ts_event" not in df.columns and df.index.name == "ts_event":
        df = df.reset_index()
    if "symbol" in df.columns:
        df = df[~df["symbol"].astype(str).str.contains("-", na=False)].copy()
        if not df.empty:
            front_symbol = df.groupby("symbol")["size"].sum().idxmax()
            df = df[df["symbol"] == front_symbol].copy()

    out = pd.DataFrame(index=df.index)
    for col in [
        "ts_event",
        "ts_recv",
        "instrument_id",
        "symbol",
        "action",
        "side",
        "price",
        "size",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "bid_ct_00",
        "ask_ct_00",
        "sequence",
        "flags",
    ]:
        out[col] = df[col] if col in df.columns else None
    out["window_id"] = window_id
    return out


def ingest_cached_dbn(sidecar_db: Path) -> int:
    import databento as db

    rows = read_windows(sidecar_db)
    ingested = 0
    con = duckdb.connect(str(sidecar_db))
    try:
        create_schema(con)
        for row in rows:
            if not row.source_dbn_path:
                continue
            path = Path(row.source_dbn_path)
            if not path.exists():
                raise FileNotFoundError(path)
            store = db.DBNStore.from_file(path)
            raw_df = store.to_df()
            event_df = _normalise_event_df(raw_df, row.window_id)
            con.execute("DELETE FROM micro_mbp1_events WHERE window_id = ?", [row.window_id])
            if event_df.empty:
                continue
            con.register("event_df", event_df)
            con.execute("INSERT INTO micro_mbp1_events SELECT * FROM event_df")
            ingested += len(event_df)
    finally:
        con.close()
    return ingested


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _signed(value: float | None, break_dir: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value) if break_dir == "long" else -float(value)


def _side_signed_size(side: object, size: object) -> float:
    side_s = str(side or "").upper()
    size_f = float(size or 0.0)
    if side_s == "B":
        return size_f
    if side_s == "A":
        return -size_f
    return 0.0


def assert_pre_entry_events(events: pd.DataFrame, entry_ts: datetime) -> None:
    if events.empty:
        return
    ts = pd.to_datetime(events["ts_event"], utc=True)
    if (ts >= pd.Timestamp(entry_ts)).any():
        raise ValueError("feature input contains ts_event >= entry_ts")


def compute_feature_for_window(events: pd.DataFrame, window: Gate0Window, *, tick_size: float = 0.25) -> Gate0Feature:
    entry_ts = _to_utc_datetime(window.entry_ts)
    start_ts = entry_ts - timedelta(seconds=window.lookback_seconds)
    if events.empty:
        scoped = events.copy()
    else:
        scoped = events.copy()
        scoped["ts_event"] = pd.to_datetime(scoped["ts_event"], utc=True)
        scoped = scoped[(scoped["ts_event"] >= pd.Timestamp(start_ts)) & (scoped["ts_event"] < pd.Timestamp(entry_ts))]
        scoped = scoped.sort_values(["ts_event", "sequence"], na_position="last")
    assert_pre_entry_events(scoped, entry_ts)

    event_count = len(scoped)
    trade_count = int((scoped.get("action", pd.Series(dtype=object)).astype(str).str.upper() == "T").sum()) if event_count else 0
    signed_ofi = None
    signed_tbi = None
    signed_qi_last = None
    signed_qi_mean = None
    spread_mean = None
    spread_max = None

    if event_count:
        bid_px = _numeric(scoped["bid_px_00"])
        ask_px = _numeric(scoped["ask_px_00"])
        bid_sz = _numeric(scoped["bid_sz_00"])
        ask_sz = _numeric(scoped["ask_sz_00"])
        prev_bid_px = bid_px.shift(1)
        prev_ask_px = ask_px.shift(1)
        prev_bid_sz = bid_sz.shift(1)
        prev_ask_sz = ask_sz.shift(1)
        valid_prev = prev_bid_px.notna() & prev_ask_px.notna()
        ofi = pd.Series(0.0, index=scoped.index)
        ofi += ((bid_px >= prev_bid_px) * bid_sz.fillna(0.0)).where(valid_prev, 0.0)
        ofi -= ((bid_px <= prev_bid_px) * prev_bid_sz.fillna(0.0)).where(valid_prev, 0.0)
        ofi -= ((ask_px <= prev_ask_px) * ask_sz.fillna(0.0)).where(valid_prev, 0.0)
        ofi += ((ask_px >= prev_ask_px) * prev_ask_sz.fillna(0.0)).where(valid_prev, 0.0)
        signed_ofi = _signed(float(ofi.sum()), window.break_dir)

        denom = bid_sz + ask_sz
        qi = ((bid_sz - ask_sz) / denom).where(denom > 0)
        if qi.notna().any():
            signed_qi_last = _signed(float(qi.dropna().iloc[-1]), window.break_dir)
        last_10s = scoped["ts_event"] >= pd.Timestamp(entry_ts - timedelta(seconds=10))
        qi_10 = qi[last_10s]
        if qi_10.notna().any():
            signed_qi_mean = _signed(float(qi_10.mean()), window.break_dir)

        spread_ticks = (ask_px - bid_px) / tick_size
        spread_10 = spread_ticks[last_10s]
        if spread_10.notna().any():
            spread_mean = float(spread_10.mean())
            spread_max = float(spread_10.max())

        trades = scoped[scoped["action"].astype(str).str.upper() == "T"]
        if not trades.empty:
            tbi_raw = sum(_side_signed_size(side, size) for side, size in zip(trades["side"], trades["size"], strict=False))
            signed_tbi = _signed(float(tbi_raw), window.break_dir)

    return Gate0Feature(
        feature_row_id=_feature_row_id(window.window_id),
        window_id=window.window_id,
        trading_day=window.trading_day,
        symbol=window.symbol,
        orb_label=window.orb_label,
        orb_minutes=window.orb_minutes,
        entry_model=window.entry_model,
        rr_target=window.rr_target,
        confirm_bars=window.confirm_bars,
        break_dir=window.break_dir,
        entry_ts=entry_ts,
        lookback_seconds=window.lookback_seconds,
        signed_ofi_60s=signed_ofi,
        signed_tbi_60s=signed_tbi,
        signed_qi_last_1s=signed_qi_last,
        signed_qi_mean_10s=signed_qi_mean,
        spread_mean_ticks_10s=spread_mean,
        spread_max_ticks_10s=spread_max,
        event_count_mbp1=event_count,
        event_count_trade=trade_count,
        feature_version="mbp1_top_of_book_v1",
    )


def compute_features(sidecar_db: Path) -> int:
    con = duckdb.connect(str(sidecar_db))
    try:
        create_schema(con)
        windows_df = con.execute("SELECT * FROM micro_gate0_windows ORDER BY trading_day").fetchdf()
        features: list[Gate0Feature] = []
        for _, row in windows_df.iterrows():
            window = build_window_from_record(row.to_dict(), sha=str(row["git_sha"]), created_at=_to_utc_datetime(row["created_at_utc"]))
            events = con.execute(
                "SELECT * FROM micro_mbp1_events WHERE window_id = ? ORDER BY ts_event, sequence",
                [window.window_id],
            ).fetchdf()
            features.append(compute_feature_for_window(events, window))
        df = pd.DataFrame([asdict(f) for f in features])
        con.execute("DELETE FROM micro_gate0_features")
        con.register("feature_df", df)
        con.execute("INSERT INTO micro_gate0_features SELECT * FROM feature_df")
        return len(features)
    finally:
        con.close()


def _selected_mask(df: pd.DataFrame, feature_id: str, thresholds: dict[str, float]) -> pd.Series:
    if feature_id == "signed_ofi_60s_high":
        return df["signed_ofi_60s"] >= thresholds[feature_id]
    if feature_id == "signed_tbi_60s_high":
        return df["signed_tbi_60s"] >= thresholds[feature_id]
    if feature_id == "signed_qi_last_1s_high":
        return df["signed_qi_last_1s"] >= thresholds[feature_id]
    if feature_id == "signed_ofi_60s_high_AND_signed_qi_last_1s_high":
        return (df["signed_ofi_60s"] >= thresholds["signed_ofi_60s_high"]) & (
            df["signed_qi_last_1s"] >= thresholds["signed_qi_last_1s_high"]
        )
    raise ValueError(feature_id)


def evaluate_sidecar(sidecar_db: Path) -> pd.DataFrame:
    con = duckdb.connect(str(sidecar_db))
    try:
        joined = con.execute(
            """
            SELECT f.*, w.pnl_r
            FROM micro_gate0_features f
            JOIN micro_gate0_windows w USING (window_id)
            ORDER BY f.trading_day
            """
        ).fetchdf()
        if joined.empty:
            raise ValueError("no feature rows available; run --features first")
        joined["trading_day"] = pd.to_datetime(joined["trading_day"]).dt.date
        is_df = joined[joined["trading_day"] < HOLDOUT_DATE].copy()
        if is_df.empty:
            raise ValueError("cannot fit thresholds without IS rows")

        thresholds = {
            "signed_ofi_60s_high": float(is_df["signed_ofi_60s"].dropna().quantile(0.75)),
            "signed_tbi_60s_high": float(is_df["signed_tbi_60s"].dropna().quantile(0.75)),
            "signed_qi_last_1s_high": float(is_df["signed_qi_last_1s"].dropna().quantile(0.75)),
        }
        if any(pd.isna(v) for v in thresholds.values()):
            raise ValueError("cannot fit thresholds: one or more IS feature columns are all null")

        results: list[dict[str, object]] = []
        feature_ids = [
            "signed_ofi_60s_high",
            "signed_tbi_60s_high",
            "signed_qi_last_1s_high",
            "signed_ofi_60s_high_AND_signed_qi_last_1s_high",
        ]
        for split, split_df in [("IS", is_df), ("OOS", joined[joined["trading_day"] >= HOLDOUT_DATE].copy())]:
            for feature_id in feature_ids:
                selected = _selected_mask(split_df, feature_id, thresholds).fillna(False)
                n_parent = len(split_df)
                n_selected = int(selected.sum())
                parent_ev = float(split_df["pnl_r"].sum() / n_parent) if n_parent else 0.0
                selected_sum = float(split_df.loc[selected, "pnl_r"].sum()) if n_selected else 0.0
                selected_policy_ev = selected_sum / n_parent if n_parent else 0.0
                selected_mean = selected_sum / n_selected if n_selected else None
                decision = "PASS_IS" if split == "IS" and selected_policy_ev > parent_ev else "FAIL_IS"
                if split == "OOS":
                    decision = "DESCRIPTIVE_POSITIVE" if selected_policy_ev > parent_ev else "DESCRIPTIVE_NONPOSITIVE"
                threshold_value = (
                    thresholds["signed_ofi_60s_high"]
                    if feature_id == "signed_ofi_60s_high_AND_signed_qi_last_1s_high"
                    else thresholds[feature_id]
                )
                results.append(
                    {
                        "result_id": _result_id(feature_id, split),
                        "hypothesis_slug": HYPOTHESIS_SLUG,
                        "feature_id": feature_id,
                        "sample_split": split,
                        "n_parent": n_parent,
                        "n_selected": n_selected,
                        "parent_policy_ev_r": parent_ev,
                        "selected_policy_ev_r": selected_policy_ev,
                        "selected_trade_mean_r": selected_mean,
                        "delta_policy_ev_r": selected_policy_ev - parent_ev,
                        "threshold_value": threshold_value,
                        "decision": decision,
                        "created_at_utc": _utc_now(),
                        "git_sha": git_sha(),
                    }
                )
        result_df = pd.DataFrame(results)
        con.execute("DELETE FROM micro_gate0_results WHERE hypothesis_slug = ?", [HYPOTHESIS_SLUG])
        con.register("result_df", result_df)
        con.execute("INSERT INTO micro_gate0_results SELECT * FROM result_df")
        return result_df
    finally:
        con.close()


def write_result_doc(result_df: pd.DataFrame, sidecar_db: Path, path: Path = RESULT_DOC) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Track D Gate 0 MBP-1 Results",
        "",
        f"- Sidecar DB: `{sidecar_db}`",
        f"- Hypothesis: `{HYPOTHESIS_SLUG}`",
        f"- Schema: `{SCHEMA}`",
        f"- Created: `{_utc_now().isoformat()}`",
        "",
        "## Results",
        "",
        result_df.to_markdown(index=False),
        "",
        "## Guardrails",
        "",
        "- Historical MBP-1 only.",
        "- Thresholds fit on IS rows before 2026-01-01 only.",
        "- OOS rows are descriptive.",
        "- No validated_setups, allocation, live state, or paper_trades mutation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def handle_build_manifest(args: argparse.Namespace) -> int:
    rows = load_manifest_rows(Path(args.gold_db))
    summary = summarize_manifest(rows)
    print(summary)
    if not args.dry_run:
        write_manifest(rows, Path(args.sidecar_db))
    return 0


def handle_estimate_cost(args: argparse.Namespace) -> int:
    rows = load_manifest_rows(Path(args.gold_db))
    sample_rows = _stratified_sample(rows) if args.dry_run else rows
    if not args.dry_run:
        write_manifest(rows, Path(args.sidecar_db))
    client = _get_databento_client()
    costs = estimate_cost_rows(sample_rows, client)
    print(
        {
            "schema": SCHEMA,
            "windows_estimated": len(costs),
            "sample_only": bool(args.dry_run),
            "total_cost_usd": round(sum(costs.values()), 6),
            "projected_full_cost_usd": round((sum(costs.values()) / len(costs)) * len(rows), 2) if costs else 0.0,
        }
    )
    if not args.dry_run:
        update_costs(Path(args.sidecar_db), costs)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-db", default=str(GOLD_DB_PATH))
    parser.add_argument("--sidecar-db", default=str(DEFAULT_SIDECAR_DB))
    parser.add_argument("--schema", default=SCHEMA, choices=[SCHEMA])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--yes", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-manifest", action="store_true")
    mode.add_argument("--estimate-cost", action="store_true")
    mode.add_argument("--pull", action="store_true")
    mode.add_argument("--ingest", action="store_true")
    mode.add_argument("--features", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.schema != SCHEMA:
        raise ValueError("Track D Gate 0 runner is MBP-1 only")
    if args.build_manifest:
        return handle_build_manifest(args)
    if args.estimate_cost:
        return handle_estimate_cost(args)
    if args.pull:
        print(pull_pending(Path(args.sidecar_db), max_cost_usd=args.max_cost_usd, yes=args.yes, dry_run=args.dry_run))
        return 0
    if args.ingest:
        print({"events_ingested": ingest_cached_dbn(Path(args.sidecar_db))})
        return 0
    if args.features:
        print({"features": compute_features(Path(args.sidecar_db))})
        return 0
    if args.evaluate:
        result_df = evaluate_sidecar(Path(args.sidecar_db))
        if not args.dry_run:
            write_result_doc(result_df, Path(args.sidecar_db))
        print(result_df.to_string(index=False))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
