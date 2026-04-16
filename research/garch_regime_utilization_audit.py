"""How to utilize garch regime families: gate economics and PnL decomposition.

Purpose:
  For session families identified in the regime-family audit, quantify the
  effect of two operational uses:

    1. TAKE_HIGH_ONLY  -> trade only when garch_pct >= 70
    2. SKIP_LOW_ONLY   -> trade unless garch_pct <= 30

This is an exploitation / understanding pass, not a validation pass.
Numbers are informational for operational design and must not be treated as
promotion evidence without holdout-clean forward shadow.

Output:
  docs/audit/results/2026-04-16-garch-regime-utilization-audit.md
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from research import garch_regime_family_audit as fam

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-regime-utilization-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def family_pool(cells: list[fam.CellRecord], session: str) -> pd.DataFrame:
    rows = []
    for c in cells:
        if c.orb_label != session:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "strategy_id": c.strategy_id,
                    "instrument": c.instrument,
                    "direction": c.direction,
                    "filter_type": c.filter_type,
                    "pnl_r": c.pnl,
                    "gp": c.gp,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_gate(df: pd.DataFrame, active_mask: pd.Series, label: str) -> dict[str, object]:
    active = df.loc[active_mask, "pnl_r"].astype(float)
    skipped = df.loc[~active_mask, "pnl_r"].astype(float)
    base = df["pnl_r"].astype(float)
    wins_skipped = skipped[skipped > 0].sum()
    losses_skipped = skipped[skipped < 0].sum()
    return {
        "rule": label,
        "n_total": len(base),
        "n_active": int(active_mask.sum()),
        "fire_pct": float(active_mask.mean()),
        "base_expr": float(base.mean()),
        "active_expr": float(active.mean()) if len(active) else float("nan"),
        "base_total_r": float(base.sum()),
        "active_total_r": float(active.sum()),
        "delta_total_r": float(active.sum() - base.sum()),
        "skipped_win_r": float(wins_skipped),
        "skipped_loss_r": float(losses_skipped),
        "saved_minus_missed": float((-losses_skipped) - wins_skipped),
    }


def main() -> None:
    cells, _ = fam.build_cells()
    directional = fam.family_directional(cells)

    candidate_sessions = directional[directional["bh_dir"] == True]["session"].drop_duplicates().tolist()
    results = []
    for session in candidate_sessions:
        df = family_pool(cells, session)
        if len(df) == 0:
            continue
        take_high = summarize_gate(df, df["gp"] >= 70, "TAKE_HIGH_ONLY")
        skip_low = summarize_gate(df, df["gp"] > 30, "SKIP_LOW_ONLY")
        results.append({"session": session, **take_high})
        results.append({"session": session, **skip_low})

    res = pd.DataFrame(results)
    emit(res)


def emit(res: pd.DataFrame) -> None:
    lines = [
        "# Garch Regime Utilization Audit",
        "",
        "**Date:** 2026-04-16",
        "**Question:** if a session family shows a garch regime effect, how does that translate into actual gate economics?",
        "",
        "Two gate modes are reported:",
        "- `TAKE_HIGH_ONLY`: trade only when `garch_pct >= 70`",
        "- `SKIP_LOW_ONLY`: trade unless `garch_pct <= 30`",
        "",
        "**Important:** these are informational exploitation numbers on the current pooled family populations. They are not deployment proof.",
        "",
        "| Session | Rule | Active % | Base ExpR | Active ExpR | Base Total R | Active Total R | Delta Total R | Skipped Wins R | Skipped Losses R | Saved-Missed R |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.sort_values(["session", "rule"]).iterrows():
        lines.append(
            f"| {r['session']} | {r['rule']} | {r['fire_pct']:.1%} | {r['base_expr']:+.3f} | {r['active_expr']:+.3f} | "
            f"{r['base_total_r']:+.1f} | {r['active_total_r']:+.1f} | {r['delta_total_r']:+.1f} | "
            f"{r['skipped_win_r']:+.1f} | {r['skipped_loss_r']:+.1f} | {r['saved_minus_missed']:+.1f} |"
        )

    lines += [
        "",
        "## Reading the table",
        "",
        "- `Active ExpR` answers the quality question: how good are the trades you keep?",
        "- `Delta Total R` answers the portfolio question: do you actually make more or less total R by gating?",
        "- `Saved-Missed R = abs(skipped losses) - skipped wins` is the cleanest decomposition of what the gate is doing.",
        "- `TAKE_HIGH_ONLY` is the strict regime-only interpretation.",
        "- `SKIP_LOW_ONLY` is the softer hostile-regime filter interpretation.",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUTPUT_MD}")


if __name__ == "__main__":
    main()
