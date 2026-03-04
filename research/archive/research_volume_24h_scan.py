#!/usr/bin/env python3
"""24-Hour Volume + Edge Scan.

Aggregates raw bars_1m volume into 15-minute Brisbane-time slots across 24 hours,
split by US and UK DST regimes.  Overlays ORB edge data from the existing time
scan CSV.  Detects volume spikes and DST-shifting events.

Output:
    research/output/volume_24h_scan.md   — full analysis report
    research/output/volume_24h_scan.csv  — one row per (instrument, bris_h, bris_m)

Usage:
    DUCKDB_PATH=C:/db/gold.db python research/research_volume_24h_scan.py
"""

import csv
import os
import statistics
import time as _time
from pathlib import Path

import duckdb

# ── Configuration ──────────────────────────────────────────────────────

DB_PATH = os.environ.get(
    "DUCKDB_PATH",
    str(Path(__file__).resolve().parent.parent / "gold.db"),
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_MD = OUTPUT_DIR / "volume_24h_scan.md"
OUTPUT_CSV = OUTPUT_DIR / "volume_24h_scan.csv"
TIME_SCAN_CSV = OUTPUT_DIR / "orb_time_scan_full.csv"

INSTRUMENTS = ["MGC", "MNQ", "MES"]

# Named sessions: (bris_h, bris_m) → label
NAMED_SESSIONS = {
    (9, 0): "0900 (CME winter)",
    (10, 0): "1000 (Tokyo)",
    (11, 0): "1100 (Singapore)",
    (11, 30): "1130 (HK/SG equity)",
    (18, 0): "1800 (London winter)",
    (23, 0): "2300 (US pre/post data)",
    (0, 30): "0030 (NYSE winter)",
    # DST-shifted positions
    (8, 0): "CME summer",
    (17, 0): "London summer",
    (23, 30): "US data winter / US equity summer",
    (22, 30): "US data summer",
}

# 96 time slots
CANDIDATE_TIMES = [(h, m) for h in range(24) for m in (0, 15, 30, 45)]


# ── Volume Query ──────────────────────────────────────────────────────

VOLUME_SQL = """
WITH bars AS (
    SELECT
        b.volume,
        EXTRACT(HOUR FROM b.ts_utc AT TIME ZONE 'Australia/Brisbane')::INT AS bris_h,
        (EXTRACT(MINUTE FROM b.ts_utc AT TIME ZONE 'Australia/Brisbane')::INT // 15) * 15 AS bris_m,
        CASE
            WHEN EXTRACT(HOUR FROM b.ts_utc AT TIME ZONE 'Australia/Brisbane') < 9
            THEN (b.ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE - INTERVAL '1 day'
            ELSE (b.ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE
        END AS trading_day
    FROM bars_1m b
    WHERE b.symbol = $1
)
SELECT
    bars.bris_h, bars.bris_m,
    COUNT(DISTINCT CASE WHEN NOT df.us_dst THEN bars.trading_day END) AS us_winter_days,
    SUM(CASE WHEN NOT df.us_dst THEN bars.volume ELSE 0 END) AS us_winter_vol,
    COUNT(DISTINCT CASE WHEN df.us_dst THEN bars.trading_day END) AS us_summer_days,
    SUM(CASE WHEN df.us_dst THEN bars.volume ELSE 0 END) AS us_summer_vol,
    COUNT(DISTINCT CASE WHEN NOT df.uk_dst THEN bars.trading_day END) AS uk_winter_days,
    SUM(CASE WHEN NOT df.uk_dst THEN bars.volume ELSE 0 END) AS uk_winter_vol,
    COUNT(DISTINCT CASE WHEN df.uk_dst THEN bars.trading_day END) AS uk_summer_days,
    SUM(CASE WHEN df.uk_dst THEN bars.volume ELSE 0 END) AS uk_summer_vol
FROM bars
JOIN daily_features df
    ON df.trading_day = bars.trading_day::DATE
    AND df.symbol = $1
    AND df.orb_minutes = 5
GROUP BY bars.bris_h, bars.bris_m
ORDER BY bars.bris_h, bars.bris_m
"""


# ── Helpers ───────────────────────────────────────────────────────────

def load_time_scan(csv_path):
    """Load time scan CSV keyed by (instrument, bris_h, bris_m)."""
    data = {}
    if not csv_path.exists():
        return data

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["instrument"], int(row["bris_h"]), int(row["bris_m"]))
            data[key] = {
                "n_trades": _int(row.get("n_trades")),
                "avg_r": _float(row.get("avg_r")),
                "n_winter": _int(row.get("n_winter")),
                "avg_r_winter": _float(row.get("avg_r_winter")),
                "wr_winter": _float(row.get("wr_winter")),
                "n_summer": _int(row.get("n_summer")),
                "avg_r_summer": _float(row.get("avg_r_summer")),
                "wr_summer": _float(row.get("wr_summer")),
            }
    return data


def _float(val):
    if val is None or str(val).strip() == "":
        return None
    try:
        v = float(val)
        # NaN check without numpy
        if v != v:
            return None
        return v
    except (ValueError, TypeError):
        return None


def _int(val):
    if val is None or str(val).strip() == "":
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def fmt_vol(val):
    """Format volume as integer with commas, or ---."""
    if val is None:
        return "---"
    return f"{int(val):,}"


def fmt_r(val):
    if val is None:
        return "---"
    return f"{val:+.4f}"


def fmt_ratio(num, denom):
    """Format ratio of two values, or --- if denom is 0/None."""
    if denom is None or denom == 0 or num is None:
        return "---"
    return f"{num / denom:.2f}"


def nearest_session(bris_h, bris_m):
    """Find nearest named session within ±1 slot (15 min)."""
    slot_min = bris_h * 60 + bris_m
    best = None
    best_dist = 999
    for (sh, sm), label in NAMED_SESSIONS.items():
        sess_min = sh * 60 + sm
        dist = abs(slot_min - sess_min)
        # Handle wrap around midnight
        dist = min(dist, 1440 - dist)
        if dist <= 15 and dist < best_dist:
            best = label
            best_dist = dist
    return best or "UNNAMED"


# ── Core Logic ────────────────────────────────────────────────────────

def query_volume(con, instrument):
    """Query volume aggregated into 15-min Brisbane-time slots.

    Returns list of dicts, one per slot, with mean daily volume per regime.
    """
    rows = con.execute(VOLUME_SQL, [instrument]).fetchall()

    results = []
    for (bris_h, bris_m, us_w_days, us_w_vol,
         us_s_days, us_s_vol, uk_w_days, uk_w_vol,
         uk_s_days, uk_s_vol) in rows:

        us_w_mean = us_w_vol / us_w_days if us_w_days and us_w_days > 0 else 0
        us_s_mean = us_s_vol / us_s_days if us_s_days and us_s_days > 0 else 0
        uk_w_mean = uk_w_vol / uk_w_days if uk_w_days and uk_w_days > 0 else 0
        uk_s_mean = uk_s_vol / uk_s_days if uk_s_days and uk_s_days > 0 else 0

        results.append({
            "bris_h": int(bris_h),
            "bris_m": int(bris_m),
            "us_winter_days": int(us_w_days or 0),
            "us_winter_vol_total": int(us_w_vol or 0),
            "us_winter_vol_mean": us_w_mean,
            "us_summer_days": int(us_s_days or 0),
            "us_summer_vol_total": int(us_s_vol or 0),
            "us_summer_vol_mean": us_s_mean,
            "uk_winter_days": int(uk_w_days or 0),
            "uk_winter_vol_total": int(uk_w_vol or 0),
            "uk_winter_vol_mean": uk_w_mean,
            "uk_summer_days": int(uk_s_days or 0),
            "uk_summer_vol_total": int(uk_s_vol or 0),
            "uk_summer_vol_mean": uk_s_mean,
        })

    return results


def detect_spikes(vol_data, vol_key, top_n=10):
    """Detect volume spikes: slots with mean daily volume > 2x median.

    Args:
        vol_data: list of slot dicts
        vol_key: which mean volume column to check (e.g. 'us_winter_vol_mean')
        top_n: number of top spikes to return

    Returns list of (bris_h, bris_m, volume, magnitude_vs_median) sorted desc.
    """
    values = [d[vol_key] for d in vol_data if d[vol_key] > 0]
    if not values:
        return []

    med = statistics.median(values)
    if med <= 0:
        return []

    spikes = []
    for d in vol_data:
        v = d[vol_key]
        if v > 2 * med:
            spikes.append((d["bris_h"], d["bris_m"], v, v / med))

    spikes.sort(key=lambda x: x[2], reverse=True)
    return spikes[:top_n]


def find_dst_shift_pairs(winter_spikes, summer_spikes, shift_slots=4):
    """Find pairs where a winter spike at T has a summer spike at T - shift_slots.

    shift_slots=4 means 1hr earlier (4 × 15min).  Also checks T + shift_slots
    for the 2300 reverse-shift case.

    Returns list of (winter_h, winter_m, summer_h, summer_m, direction).
    """
    summer_set = {(h, m) for h, m, _, _ in summer_spikes}
    pairs = []

    for wh, wm, _, _ in winter_spikes:
        w_slot = wh * 4 + wm // 15

        # Forward shift: event moves 1hr earlier in summer
        s_slot = (w_slot - shift_slots) % 96
        sh, sm = divmod(s_slot, 4)
        sm *= 15
        if (sh, sm) in summer_set:
            pairs.append((wh, wm, sh, sm, "event 1hr earlier in summer"))

        # Reverse shift: event moves 1hr later in summer (2300 case)
        s_slot_rev = (w_slot + shift_slots) % 96
        sh_r, sm_r = divmod(s_slot_rev, 4)
        sm_r *= 15
        if (sh_r, sm_r) in summer_set:
            pairs.append((wh, wm, sh_r, sm_r, "event 1hr later in summer"))

    return pairs


# ── Report Generation ─────────────────────────────────────────────────

def build_report(all_data, edge_data):
    """Build the full markdown report.

    Args:
        all_data: dict of instrument → list of slot dicts
        edge_data: dict from load_time_scan()
    """
    out = []
    out.append("# 24-Hour Volume + Edge Scan")
    out.append("")
    out.append("**Method:** Raw `bars_1m.volume` aggregated into 15-minute "
               "Brisbane-time slots, split by US and UK DST regimes.  Mean "
               "daily volume per slot = total_vol / n_trading_days.  Edge "
               "data overlaid from `orb_time_scan_full.csv`.")
    out.append("")
    out.append("**Instruments:** " + ", ".join(INSTRUMENTS))
    out.append("")

    # Day counts
    for inst in INSTRUMENTS:
        if all_data.get(inst):
            d = all_data[inst][0]
            out.append(f"- **{inst}:** {d['us_winter_days']} US-winter days, "
                       f"{d['us_summer_days']} US-summer days, "
                       f"{d['uk_winter_days']} UK-winter days, "
                       f"{d['uk_summer_days']} UK-summer days")
    out.append("")

    if edge_data:
        out.append("**Edge overlay caveat:** Time scan uses US DST for all "
                   "instruments.  For 1800/London (UK DST driver), the edge "
                   "winter/summer classification differs from the volume UK "
                   "split.")
    else:
        out.append("**Note:** Time scan CSV not found — edge overlay skipped.")
    out.append("")
    out.append("---")
    out.append("")

    # Section 1: Volume Profile Summary
    section_spikes(out, all_data)

    # Section 2: Full 24-Hour Tables
    section_full_tables(out, all_data, edge_data)

    # Section 3: DST-Confirmed Events
    section_dst_events(out, all_data, edge_data)

    # Section 4: Undiscovered Events
    section_undiscovered(out, all_data)

    # Section 5: Cross-Instrument Comparison
    section_cross_instrument(out, all_data)

    return out


def section_spikes(out, all_data):
    """Section 1: Top volume spikes per instrument, per regime."""
    out.append("## 1. Volume Profile Summary")
    out.append("")

    for inst in INSTRUMENTS:
        vol_data = all_data.get(inst, [])
        if not vol_data:
            continue

        out.append(f"### {inst}")
        out.append("")

        for regime_label, vol_key in [
            ("US-Winter", "us_winter_vol_mean"),
            ("US-Summer", "us_summer_vol_mean"),
        ]:
            spikes = detect_spikes(vol_data, vol_key)
            if not spikes:
                out.append(f"**{regime_label}:** No spikes detected.")
                out.append("")
                continue

            out.append(f"**{regime_label} — Top {len(spikes)} spikes "
                       f"(> 2x median):**")
            out.append("")
            out.append("| Rank | Time | Mean Daily Vol | x Median | Session |")
            out.append("|------|------|---------------|----------|---------|")
            for i, (h, m, vol, mag) in enumerate(spikes, 1):
                sess = nearest_session(h, m)
                out.append(f"| {i} | {h:02d}:{m:02d} | {fmt_vol(vol)} "
                           f"| {mag:.1f}x | {sess} |")
            out.append("")

    out.append("---")
    out.append("")


def section_full_tables(out, all_data, edge_data):
    """Section 2: Full 24-hour table per instrument."""
    out.append("## 2. Full 24-Hour Volume + Edge Tables")
    out.append("")

    for inst in INSTRUMENTS:
        vol_data = all_data.get(inst, [])
        if not vol_data:
            continue

        # Build lookup
        vol_by_slot = {(d["bris_h"], d["bris_m"]): d for d in vol_data}

        # Compute medians for spike flagging
        us_w_vals = [d["us_winter_vol_mean"] for d in vol_data
                     if d["us_winter_vol_mean"] > 0]
        us_s_vals = [d["us_summer_vol_mean"] for d in vol_data
                     if d["us_summer_vol_mean"] > 0]
        us_w_med = statistics.median(us_w_vals) if us_w_vals else 0
        us_s_med = statistics.median(us_s_vals) if us_s_vals else 0

        out.append(f"### {inst}")
        out.append("")
        out.append(f"Median daily volume: US-W={fmt_vol(us_w_med)}, "
                   f"US-S={fmt_vol(us_s_med)}")
        out.append("")
        out.append("| Time | US-W Vol | US-S Vol | S/W Ratio "
                   "| Edge-W | Edge-S | Spike? | Session |")
        out.append("|------|----------|----------|-----------|"
                   "--------|--------|--------|---------|")

        for bh, bm in CANDIDATE_TIMES:
            d = vol_by_slot.get((bh, bm))
            if d is None:
                continue

            us_w = d["us_winter_vol_mean"]
            us_s = d["us_summer_vol_mean"]
            ratio = fmt_ratio(us_s, us_w)

            # Edge overlay
            ekey = (inst, bh, bm)
            e = edge_data.get(ekey, {})
            edge_w = fmt_r(e.get("avg_r_winter"))
            edge_s = fmt_r(e.get("avg_r_summer"))

            # Spike flag
            spike_flags = []
            if us_w_med > 0 and us_w > 2 * us_w_med:
                spike_flags.append("US-W")
            if us_s_med > 0 and us_s > 2 * us_s_med:
                spike_flags.append("US-S")
            spike = ", ".join(spike_flags) if spike_flags else ""

            # Session label
            sess = NAMED_SESSIONS.get((bh, bm), "")

            out.append(
                f"| {bh:02d}:{bm:02d} "
                f"| {fmt_vol(us_w)} "
                f"| {fmt_vol(us_s)} "
                f"| {ratio} "
                f"| {edge_w} "
                f"| {edge_s} "
                f"| {spike} "
                f"| {sess} |"
            )

        out.append("")

    out.append("---")
    out.append("")


def section_dst_events(out, all_data, edge_data):
    """Section 3: DST-Confirmed Events — volume spike shift pairs."""
    out.append("## 3. DST-Confirmed Events")
    out.append("")
    out.append("Pairs where a volume spike in one DST regime has a "
               "corresponding spike shifted by exactly 1 hour in the other "
               "regime — confirming the spike tracks a DST-shifting event.")
    out.append("")

    for inst in INSTRUMENTS:
        vol_data = all_data.get(inst, [])
        if not vol_data:
            continue

        # US DST pairs
        us_w_spikes = detect_spikes(vol_data, "us_winter_vol_mean", top_n=20)
        us_s_spikes = detect_spikes(vol_data, "us_summer_vol_mean", top_n=20)
        us_pairs = find_dst_shift_pairs(us_w_spikes, us_s_spikes)

        # UK DST pairs
        uk_w_spikes = detect_spikes(vol_data, "uk_winter_vol_mean", top_n=20)
        uk_s_spikes = detect_spikes(vol_data, "uk_summer_vol_mean", top_n=20)
        uk_pairs = find_dst_shift_pairs(uk_w_spikes, uk_s_spikes)

        if not us_pairs and not uk_pairs:
            out.append(f"**{inst}:** No DST shift pairs detected.")
            out.append("")
            continue

        out.append(f"### {inst}")
        out.append("")

        if us_pairs:
            out.append("**US DST shifts:**")
            out.append("")
            out.append("| Winter Time | Summer Time | Shift | Edge-W (winter) "
                       "| Edge-S (summer) |")
            out.append("|-------------|-------------|-------|------------------"
                       "|-----------------|")
            for wh, wm, sh, sm, direction in us_pairs:
                ew = edge_data.get((inst, wh, wm), {})
                es = edge_data.get((inst, sh, sm), {})
                out.append(
                    f"| {wh:02d}:{wm:02d} | {sh:02d}:{sm:02d} "
                    f"| {direction} "
                    f"| {fmt_r(ew.get('avg_r_winter'))} "
                    f"| {fmt_r(es.get('avg_r_summer'))} |"
                )
            out.append("")

        if uk_pairs:
            out.append("**UK DST shifts:**")
            out.append("")
            out.append("| Winter Time | Summer Time | Shift | Edge-W (winter) "
                       "| Edge-S (summer) |")
            out.append("|-------------|-------------|-------|------------------"
                       "|-----------------|")
            for wh, wm, sh, sm, direction in uk_pairs:
                ew = edge_data.get((inst, wh, wm), {})
                es = edge_data.get((inst, sh, sm), {})
                out.append(
                    f"| {wh:02d}:{wm:02d} | {sh:02d}:{sm:02d} "
                    f"| {direction} "
                    f"| {fmt_r(ew.get('avg_r_winter'))} "
                    f"| {fmt_r(es.get('avg_r_summer'))} |"
                )
            out.append("")

    out.append("---")
    out.append("")


def section_undiscovered(out, all_data):
    """Section 4: Volume spikes that don't match any named session."""
    out.append("## 4. Undiscovered Events")
    out.append("")
    out.append("Volume spikes (> 2x median) at times that don't correspond "
               "to any named session (beyond ±1 slot).  These are candidates "
               "for new dynamic sessions or further research.")
    out.append("")

    found_any = False
    for inst in INSTRUMENTS:
        vol_data = all_data.get(inst, [])
        if not vol_data:
            continue

        unnamed = []
        for vol_key, regime in [("us_winter_vol_mean", "US-W"),
                                ("us_summer_vol_mean", "US-S")]:
            spikes = detect_spikes(vol_data, vol_key, top_n=20)
            for h, m, vol, mag in spikes:
                sess = nearest_session(h, m)
                if sess == "UNNAMED":
                    unnamed.append((h, m, regime, vol, mag))

        if unnamed:
            found_any = True
            out.append(f"### {inst}")
            out.append("")
            out.append("| Time | Regime | Mean Daily Vol | x Median |")
            out.append("|------|--------|---------------|----------|")
            for h, m, regime, vol, mag in unnamed:
                out.append(f"| {h:02d}:{m:02d} | {regime} "
                           f"| {fmt_vol(vol)} | {mag:.1f}x |")
            out.append("")

    if not found_any:
        out.append("No unnamed volume spikes detected across any instrument.")
        out.append("")

    out.append("---")
    out.append("")


def section_cross_instrument(out, all_data):
    """Section 5: Cross-instrument comparison of volume spikes."""
    out.append("## 5. Cross-Instrument Comparison")
    out.append("")

    # Collect all spike slots per instrument
    spike_sets = {}
    for inst in INSTRUMENTS:
        vol_data = all_data.get(inst, [])
        if not vol_data:
            continue
        slots = set()
        for vol_key in ["us_winter_vol_mean", "us_summer_vol_mean"]:
            for h, m, _, _ in detect_spikes(vol_data, vol_key, top_n=20):
                slots.add((h, m))
        spike_sets[inst] = slots

    if len(spike_sets) < 2:
        out.append("Insufficient instruments for comparison.")
        out.append("")
        return

    # Find universal vs instrument-specific spikes
    all_slots = set()
    for s in spike_sets.values():
        all_slots |= s

    out.append("| Time | " + " | ".join(INSTRUMENTS) + " | Coverage |")
    out.append("|------|-" + "-|-".join("---" for _ in INSTRUMENTS) + "-|----------|")

    for h, m in sorted(all_slots):
        cells = []
        count = 0
        for inst in INSTRUMENTS:
            if (h, m) in spike_sets.get(inst, set()):
                # Get the volume
                vol_data = all_data.get(inst, [])
                slot = next((d for d in vol_data
                             if d["bris_h"] == h and d["bris_m"] == m), None)
                if slot:
                    # Use max of US winter/summer as representative
                    v = max(slot["us_winter_vol_mean"],
                            slot["us_summer_vol_mean"])
                    cells.append(fmt_vol(v))
                else:
                    cells.append("spike")
                count += 1
            else:
                cells.append("---")

        coverage = f"{count}/{len(INSTRUMENTS)}"
        label = " ALL" if count == len(INSTRUMENTS) else ""
        sess = nearest_session(h, m)
        sess_note = f" ({sess})" if sess != "UNNAMED" else ""
        out.append(
            f"| {h:02d}:{m:02d}{sess_note} | "
            + " | ".join(cells)
            + f" | {coverage}{label} |"
        )

    out.append("")


# ── CSV Output ────────────────────────────────────────────────────────

def write_csv(all_data, edge_data):
    """Write full results CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "instrument", "bris_h", "bris_m", "time",
        "us_winter_days", "us_winter_vol_mean",
        "us_summer_days", "us_summer_vol_mean",
        "us_summer_winter_ratio",
        "uk_winter_days", "uk_winter_vol_mean",
        "uk_summer_days", "uk_summer_vol_mean",
        "uk_summer_winter_ratio",
        "edge_n_trades", "edge_avg_r",
        "edge_n_winter", "edge_avg_r_winter",
        "edge_n_summer", "edge_avg_r_summer",
        "session",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for inst in INSTRUMENTS:
            vol_data = all_data.get(inst, [])
            for d in vol_data:
                bh, bm = d["bris_h"], d["bris_m"]
                ekey = (inst, bh, bm)
                e = edge_data.get(ekey, {})

                us_ratio = (d["us_summer_vol_mean"] / d["us_winter_vol_mean"]
                            if d["us_winter_vol_mean"] > 0 else None)
                uk_ratio = (d["uk_summer_vol_mean"] / d["uk_winter_vol_mean"]
                            if d["uk_winter_vol_mean"] > 0 else None)

                writer.writerow({
                    "instrument": inst,
                    "bris_h": bh,
                    "bris_m": bm,
                    "time": f"{bh:02d}:{bm:02d}",
                    "us_winter_days": d["us_winter_days"],
                    "us_winter_vol_mean": f"{d['us_winter_vol_mean']:.1f}",
                    "us_summer_days": d["us_summer_days"],
                    "us_summer_vol_mean": f"{d['us_summer_vol_mean']:.1f}",
                    "us_summer_winter_ratio": (f"{us_ratio:.3f}"
                                               if us_ratio is not None
                                               else ""),
                    "uk_winter_days": d["uk_winter_days"],
                    "uk_winter_vol_mean": f"{d['uk_winter_vol_mean']:.1f}",
                    "uk_summer_days": d["uk_summer_days"],
                    "uk_summer_vol_mean": f"{d['uk_summer_vol_mean']:.1f}",
                    "uk_summer_winter_ratio": (f"{uk_ratio:.3f}"
                                               if uk_ratio is not None
                                               else ""),
                    "edge_n_trades": e.get("n_trades", ""),
                    "edge_avg_r": (f"{e['avg_r']:.4f}"
                                   if e.get("avg_r") is not None else ""),
                    "edge_n_winter": e.get("n_winter", ""),
                    "edge_avg_r_winter": (f"{e['avg_r_winter']:.4f}"
                                          if e.get("avg_r_winter") is not None
                                          else ""),
                    "edge_n_summer": e.get("n_summer", ""),
                    "edge_avg_r_summer": (f"{e['avg_r_summer']:.4f}"
                                          if e.get("avg_r_summer") is not None
                                          else ""),
                    "session": NAMED_SESSIONS.get((bh, bm), ""),
                })

    return len(all_data.get(INSTRUMENTS[0], [])) * len(INSTRUMENTS)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t_start = _time.time()
    print(f"Database: {DB_PATH}")
    print(f"Instruments: {', '.join(INSTRUMENTS)}")
    print()

    con = duckdb.connect(DB_PATH, read_only=True)

    # Sanity check: bars_1m row counts
    for inst in INSTRUMENTS:
        n = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = $1", [inst]
        ).fetchone()[0]
        print(f"  {inst}: {n:,} bars in bars_1m")

    # Load edge data
    edge_data = load_time_scan(TIME_SCAN_CSV)
    if edge_data:
        print(f"\nEdge overlay: {len(edge_data)} entries from time scan CSV")
    else:
        print("\nTime scan CSV not found — edge overlay will be skipped")

    # Query volume for each instrument
    all_data = {}
    for inst in INSTRUMENTS:
        print(f"\nQuerying {inst} volume...")
        t0 = _time.time()
        vol = query_volume(con, inst)
        elapsed = _time.time() - t0
        print(f"  {len(vol)} slots in {elapsed:.1f}s")

        if vol:
            # Quick summary
            us_w_total = sum(d["us_winter_vol_total"] for d in vol)
            us_s_total = sum(d["us_summer_vol_total"] for d in vol)
            print(f"  Total volume: US-winter={us_w_total:,}, "
                  f"US-summer={us_s_total:,}")

        all_data[inst] = vol

    con.close()

    # Build report
    print("\nBuilding report...")
    report = build_report(all_data, edge_data)

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Report: {OUTPUT_MD} ({len(report)} lines)")

    n_rows = write_csv(all_data, edge_data)
    print(f"CSV:    {OUTPUT_CSV} ({n_rows} rows)")

    elapsed_total = _time.time() - t_start
    print(f"\nDone in {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
