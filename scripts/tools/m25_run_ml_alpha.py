#!/usr/bin/env python
"""Rerun ML alpha improvement review with 128K output tokens."""
from __future__ import annotations

import glob
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.m25_audit import load_api_key, audit, ARCHITECTURE_CONTEXT

api_key = load_api_key()

# Gather ALL ML files + context
files = sorted(glob.glob(str(PROJECT_ROOT / "trading_app" / "ml" / "*.py")))
files += [
    str(PROJECT_ROOT / f)
    for f in [
        "CLAUDE.md",
        "RESEARCH_RULES.md",
        "TRADING_RULES.md",
        "trading_app/config.py",
        "pipeline/cost_model.py",
        "trading_app/execution_engine.py",
    ]
]

parts = []
for f in files:
    p = Path(f)
    if p.exists():
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if len(lines) > 2000:
            content = "\n".join(lines[:2000]) + f"\n\n... [TRUNCATED - {len(lines)} total lines]"
        rel = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p
        parts.append(f"### {rel} ({len(lines)} lines)\n```python\n{content}\n```")

file_content = "\n\n---\n\n".join(parts)

SYSTEM_PROMPT = f"""\
You are the Head of Systematic Strategy Research at a $2B macro fund. PhD in Financial Engineering, \
18 years building production ML overlays for systematic trading at Renaissance, Two Sigma, and D.E. Shaw. \
You have deployed 40+ meta-labeling systems to production. You know what separates a B-grade ML overlay \
from an A-grade one that adds 30-50% more R per year.

{ARCHITECTURE_CONTEXT}

---

**ADDITIONAL ML SYSTEM CONTEXT (READ CAREFULLY)**

This is a meta-labeling RF classifier for ORB futures breakout trading (4 instruments: MGC, MNQ, MES, M2K).
Current production: V3.1 single-config per-session hybrid. 12 sessions with models, +251.9R honest OOS.

Key facts:
- Decision point: PRE-BREAK (predict before stop order placed, features known at ORB close)
- Per-session models: one RF per session per instrument. Sessions without enough data fail-open.
- 3-way split: 60% train (RF + CPCV), 20% val (threshold optimization on total-R), 20% test (frozen OOS)
- CPCV (de Prado): 5 groups, k=2, purge=1d, embargo=1d — this IS the walk-forward equivalent
- 4 quality gates: delta_r >= 0, CPCV AUC >= 0.50, test AUC > 0.52, skip <= 85%
- Fail-open design: missing model or failure = take all trades (no skipped edge)
- rr_target REMOVED from features (was tautological, dominated 56-69% importance)
- E6 noise filter drops: orb_label one-hots, gap_type, atr_vel_regime, confirm_bars, orb_minutes
- skip_filter=True for MGC/MES/M2K (filters reduce N to ~83, making ML impossible)
- MNQ uses filtered mode (G-filter edge is real, not selection bias)

Current features (after E6 filter, per session):
- Global: atr_20, atr_vel_ratio, gap_open_points_norm, prev_day_range_norm, overnight_range_norm
- Session: orb_size_norm, orb_volume, rel_vol
- Cross-session (late sessions only): prior_sessions_broken, prior_sessions_long, prior_sessions_short
- Level proximity (late sessions only): nearest_level_to_high_R, nearest_level_to_low_R, levels_within_1R, \
levels_within_2R, orb_nested_in_prior, prior_orb_size_ratio_max
- Trade config: entry_model (one-hot)

Feature importance: overnight_range ~6.5% avg #1 global, prior_sessions_broken #1 for MES (12.2%),
levels_within_2R #2 for MES (7.2%). Features are broadly distributed (no single dominator after rr_target removal).

WHAT WE HAVE ALREADY TRIED AND KILLED:
- rr_target as feature (tautological, removed)
- GARCH volatility (no mechanism, correlated with atr_20)
- RSI (mean-reverting, wrong paradigm for breakouts)
- Day of week (0 BH FDR survivors)
- Calendar overlays NFP/OPEX/FOMC (0 BH survivors at q=0.10)
- Pre-break compression (90+ tests, 0 BH survivors)
- Break quality bars (look-ahead for E0, no signal for E1/E2)
- At-break features (break_delay, break_bar_volume — valid theory but unknown pre-break)

---

**YOUR TASK: ALPHA IMPROVEMENT REVIEW**

You are NOT auditing for bugs. The system works and passes all quality gates.
You are reviewing for UNTAPPED EDGE — concrete, implementable improvements that could add +30-100R honest OOS.

Be EXHAUSTIVE. You have unlimited output space. Go deep on every section. Show your working.
Provide specific formulas, pseudocode, implementation sketches. Do not summarize — EXPAND.

Structure your review as:

## 1. FEATURE ENGINEERING OPPORTUNITIES
What features are we MISSING that have structural economic rationale for ORB breakout prediction?
- Must be knowable at ORB close (pre-break constraint)
- Must have a structural mechanism (why would this predict breakout success?)
- Must not duplicate existing features
- Rank by expected information gain and implementation difficulty
- For each suggested feature: state the MECHANISM, the FORMULA, the expected importance range, \
and HOW to validate it without data snooping

## 2. MODEL ARCHITECTURE IMPROVEMENTS
Is RF the right model? What about:
- Gradient boosting (LightGBM/XGBoost) — would it capture non-linearities RF misses?
- Ensemble of RF + GBM — would the disagreement signal be informative?
- Threshold optimization: is total-R the right objective? What about risk-adjusted?
- Hyperparameter tuning: are we leaving edge on the table with max_depth=6, n_estimators=500?
- Per-instrument model differences: should MGC and MES have different architectures?
- Provide SPECIFIC hyperparameter recommendations with reasoning

## 3. DATA UTILIZATION
Are we wasting data? Consider:
- 60/20/20 split with per-session N=200-800 — is this optimal?
- Should we use expanding windows instead of fixed split?
- Multi-task learning: can sessions with similar features share information?
- Transfer learning from high-N sessions to low-N sessions?
- Provide SPECIFIC split ratios and minimum N calculations

## 4. CALIBRATION & DEPLOYMENT
How to extract more R from existing models:
- Probability calibration analysis
- Dynamic thresholds
- Confidence-weighted position sizing: P(win) as position size scalar?
- Portfolio-level ML
- Provide SPECIFIC calibration validation protocols and metrics

## 5. MONITORING & REGIME DETECTION
What monitoring would catch model decay earlier:
- Feature distribution drift detection — SPECIFIC tests and thresholds
- Model confidence tracking over time — SPECIFIC metrics
- Regime-conditional performance decomposition — SPECIFIC regime definitions
- Automatic retraining triggers — SPECIFIC trigger conditions

## 6. THE +50R ROADMAP
TOP 10 highest-ROI improvements with:
- Expected R gain, confidence level, implementation time
- SPECIFIC experiment pseudocode to validate each
- What could go wrong

CRITICAL: Do NOT suggest things we already tried and killed (listed above).
Do NOT suggest things already implemented (read the code).
Do NOT suggest features that require post-break information (pre-break constraint).
"""

print(f"Payload: {len(file_content) + len(SYSTEM_PROMPT):,} chars", file=sys.stderr)
print("Sending ML alpha improvement prompt to M2.5 (128K output)...", file=sys.stderr)

result = audit(file_content, SYSTEM_PROMPT, api_key, include_context=False, timeout=600.0)

ts = datetime.now().strftime("%Y%m%d_%H%M")
out_dir = PROJECT_ROOT / "research" / "output"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"m25_ml_alpha_128k_{ts}.md"
out.write_text(
    f"# M2.5 ML Alpha Improvement Review (128K output)\n"
    f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n"
    f"**Max tokens:** 131072\n\n---\n\n" + result,
    encoding="utf-8",
)
print(f"\nSaved: {out}", file=sys.stderr)
print(f"Output length: {len(result):,} chars", file=sys.stderr)
