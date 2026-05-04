"""Output directory and formatting helpers for research scripts.

Replaces OUTPUT_DIR.mkdir(parents=True, exist_ok=True) boilerplate
found in 70+ scripts.
"""

from pathlib import Path

import pandas as pd

# Canonical output dir -- can be monkeypatched in tests
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def output_dir() -> Path:
    """Return research/output/ directory, creating if needed."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write DataFrame to research/output/<filename>."""
    path = output_dir() / filename
    df.to_csv(path, index=False)
    return path


def write_markdown(text: str, filename: str) -> Path:
    """Write text to research/output/<filename>."""
    path = output_dir() / filename
    path.write_text(text)
    return path


def format_stats_table(results: dict) -> str:
    """Format stats dict as markdown table.

    Input: {"label": {"n": int, "mean_r": float, "win_rate": float, "p_value": float}}
    Output: markdown table string.
    """
    lines = [
        "| Label | N | Mean R | WR | p-value |",
        "|-------|---:|-------:|----:|--------:|",
    ]
    for label, m in results.items():
        n = m.get("n", 0)
        mean_r = m.get("mean_r", 0)
        wr = m.get("win_rate", 0)
        p = m.get("p_value", None)
        p_str = f"{p:.4f}" if p is not None else "—"
        lines.append(f"| {label} | {n} | {mean_r:+.4f} | {wr:.1%} | {p_str} |")
    return "\n".join(lines)
