"""Generate zero-context adversarial audit data package.

Outputs raw statistical tables for evaluation by a fresh session
with no project context, memory, or accumulated bias.
"""

import sys

import duckdb
import numpy as np
from scipy import stats

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

sys.path.insert(0, ".")
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

out = []


def p(line=""):
    out.append(line)
    print(line)


p("# Zero-Context Adversarial Audit — Raw Data Package")
p("# Generated: 2026-03-19")
p("# Context: NONE. Evaluate these numbers cold.")
p("# Instrument: Micro futures (MGC=gold, MNQ=nasdaq, MES=S&P)")
p("# Strategy: Opening Range Breakout (first 5 min of session)")
p("# Entry: E2 = stop-market at ORB boundary + 1 tick slippage")
p("# Stop: opposite ORB boundary. Target: RR * risk.")
p("# Costs: included in pnl_r (MGC $5.74, MNQ $2.74, MES $3.74 round-trip)")
p("# All p-values: one-sample t-test, H0: mean_R = 0")
p()

sessions = [r[0] for r in con.execute("SELECT DISTINCT orb_label FROM orb_outcomes ORDER BY orb_label").fetchall()]

# --- Section 1: Unfiltered baseline ---
p("## 1. UNFILTERED BASELINE (ALL ORB sizes, E2 CB1 RR2.0 O5)")
p()
p(f"{'Inst':5s} {'Session':20s} {'N':>6s} {'MeanR':>8s} {'StdR':>7s} {'WinR':>6s} {'t':>7s} {'p':>10s} {'Yrs':>4s}")
p("-" * 80)

for inst in sorted(ACTIVE_ORB_INSTRUMENTS):
    for sess in sessions:
        rows = con.execute(
            """SELECT pnl_r, EXTRACT(YEAR FROM trading_day) as yr
               FROM orb_outcomes
               WHERE symbol = ? AND orb_label = ? AND orb_minutes = 5
                 AND entry_model = 'E2' AND confirm_bars = 1 AND rr_target = 2.0
                 AND outcome IN ('win', 'loss')""",
            [inst, sess],
        ).fetchall()
        if len(rows) < 30:
            continue
        pnl = np.array([r[0] for r in rows])
        yrs = len(set(int(r[1]) for r in rows))
        t_stat, p_val = stats.ttest_1samp(pnl, 0)
        wr = np.mean(pnl > 0)
        p(
            f"{inst:5s} {sess:20s} {len(pnl):6d} {pnl.mean():+8.4f} "
            f"{pnl.std():7.4f} {wr:6.3f} {t_stat:+7.2f} {p_val:10.6f} {yrs:3d}yr"
        )
    p()

# --- Section 2: BH FDR correction ---
p()
p("## 2. BH FDR CORRECTION (all instruments x sessions x RR1.0-4.0)")
p()

all_tests = []
for inst in sorted(ACTIVE_ORB_INSTRUMENTS):
    for sess in sessions:
        for rr in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            rows = con.execute(
                """SELECT pnl_r FROM orb_outcomes
                   WHERE symbol = ? AND orb_label = ? AND orb_minutes = 5
                     AND entry_model = 'E2' AND confirm_bars = 1 AND rr_target = ?
                     AND outcome IN ('win', 'loss')""",
                [inst, sess, rr],
            ).fetchall()
            if len(rows) < 30:
                continue
            pnl = np.array([r[0] for r in rows])
            t_stat, p_val = stats.ttest_1samp(pnl, 0)
            all_tests.append(
                {
                    "inst": inst,
                    "sess": sess,
                    "rr": rr,
                    "n": len(pnl),
                    "mean_r": pnl.mean(),
                    "std_r": pnl.std(),
                    "p_raw": p_val,
                    "wr": np.mean(pnl > 0),
                }
            )

all_tests.sort(key=lambda x: x["p_raw"])
m = len(all_tests)
for i, t in enumerate(all_tests):
    t["p_adj"] = min(t["p_raw"] * m / (i + 1), 1.0)
for i in range(len(all_tests) - 2, -1, -1):
    all_tests[i]["p_adj"] = min(all_tests[i]["p_adj"], all_tests[i + 1]["p_adj"])

pos_surv = [t for t in all_tests if t["p_adj"] < 0.05 and t["mean_r"] > 0]
neg_surv = [t for t in all_tests if t["p_adj"] < 0.05 and t["mean_r"] < 0]

p(f"Total hypothesis tests: {m}")
p(f"BH FDR survivors (p_adj < 0.05): {len(pos_surv) + len(neg_surv)}")
p(f"  Positive (potential edge): {len(pos_surv)}")
p(f"  Negative (confirmed losers): {len(neg_surv)}")
p()
p("### Positive BH FDR survivors:")
p(f"{'Inst':5s} {'Session':20s} {'RR':>5s} {'N':>6s} {'MeanR':>8s} {'WinR':>6s} {'p_adj':>10s}")
p("-" * 65)
for t in sorted(pos_surv, key=lambda x: (x["inst"], -x["mean_r"])):
    p(
        f"{t['inst']:5s} {t['sess']:20s} {t['rr']:5.1f} {t['n']:6d} "
        f"{t['mean_r']:+8.4f} {t['wr']:6.3f} {t['p_adj']:10.6f}"
    )

p()
p("### Negative BH FDR survivors (confirmed money losers):")
for t in sorted(neg_surv, key=lambda x: x["mean_r"]):
    p(
        f"{t['inst']:5s} {t['sess']:20s} {t['rr']:5.1f} {t['n']:6d} "
        f"{t['mean_r']:+8.4f} {t['wr']:6.3f} {t['p_adj']:10.6f}"
    )

# --- Section 3: G4 filter impact ---
p()
p("## 3. G4 FILTER IMPACT (ORB >= 4 points, E2 CB1 RR2.0 O5)")
p()
p(f"{'Inst':5s} {'Session':20s} {'N_raw':>6s} {'R_raw':>8s} {'N_g4':>6s} {'R_g4':>8s} {'Delta':>8s} {'p_g4':>10s}")
p("-" * 78)

for inst in sorted(ACTIVE_ORB_INSTRUMENTS):
    for sess in sessions:
        try:
            raw = con.execute(
                """SELECT pnl_r FROM orb_outcomes
                   WHERE symbol = ? AND orb_label = ? AND orb_minutes = 5
                     AND entry_model = 'E2' AND confirm_bars = 1 AND rr_target = 2.0
                     AND outcome IN ('win', 'loss')""",
                [inst, sess],
            ).fetchall()
            g4 = con.execute(
                f"""SELECT o.pnl_r FROM orb_outcomes o
                    JOIN daily_features df ON o.trading_day = df.trading_day
                      AND o.symbol = df.symbol AND o.orb_minutes = df.orb_minutes
                    WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = 5
                      AND o.entry_model = 'E2' AND o.confirm_bars = 1 AND o.rr_target = 2.0
                      AND o.outcome IN ('win', 'loss')
                      AND df.orb_{sess}_size >= 4""",
                [inst, sess],
            ).fetchall()
            if len(raw) < 30 or len(g4) < 20:
                continue
            pnl_raw = np.array([r[0] for r in raw])
            pnl_g4 = np.array([r[0] for r in g4])
            _, p_val = stats.ttest_1samp(pnl_g4, 0)
            delta = pnl_g4.mean() - pnl_raw.mean()
            p(
                f"{inst:5s} {sess:20s} {len(pnl_raw):6d} {pnl_raw.mean():+8.4f} "
                f"{len(pnl_g4):6d} {pnl_g4.mean():+8.4f} {delta:+8.4f} {p_val:10.6f}"
            )
        except Exception:
            pass

# --- Section 4: Yearly consistency ---
p()
p("## 4. YEARLY CONSISTENCY (MNQ top sessions, E2 CB1 RR2.0 O5 unfiltered)")
p()

for sess in ["CME_PRECLOSE", "NYSE_OPEN", "COMEX_SETTLE", "EUROPE_FLOW", "US_DATA_1000"]:
    p(f"{sess}:")
    rows = con.execute(
        """SELECT EXTRACT(YEAR FROM trading_day) as yr, COUNT(*) as n, AVG(pnl_r) as mean_r
           FROM orb_outcomes
           WHERE symbol = 'MNQ' AND orb_label = ? AND orb_minutes = 5
             AND entry_model = 'E2' AND confirm_bars = 1 AND rr_target = 2.0
             AND outcome IN ('win', 'loss')
           GROUP BY yr ORDER BY yr""",
        [sess],
    ).fetchall()
    pos = sum(1 for r in rows if r[2] > 0)
    for r in rows:
        p(f"  {int(r[0])}: N={r[1]:4d} MeanR={r[2]:+.4f}")
    p(f"  {pos}/{len(rows)} positive years")
    p()

# --- Section 5: Validated strategies ---
p("## 5. CURRENTLY VALIDATED STRATEGIES")
p()
rows = con.execute(
    """SELECT strategy_id, expectancy_r, sharpe_ratio, sample_size, win_rate, filter_type
       FROM validated_setups WHERE status = 'active'
       ORDER BY instrument, expectancy_r DESC"""
).fetchall()
p(f"Total active: {len(rows)}")
p(f"{'Strategy':55s} {'ExpR':>7s} {'Sharpe':>7s} {'N':>5s} {'WR':>5s} {'Filter':25s}")
p("-" * 110)
for r in rows:
    p(f"{r[0]:55s} {r[1]:+7.3f} {r[2]:+7.3f} {r[3]:5d} {r[4]:5.3f} {r[5]:25s}")

# --- Section 6: Questions ---
p()
p("## 6. QUESTIONS FOR COLD EVALUATOR")
p()
p("1. Is there a real edge in ORB breakout on micro futures? Show your work.")
p("2. Do the BH FDR survivors represent genuine signal or grid-search artifact?")
p("3. The G4 filter converts MGC from -0.35R to +0.39R on N=129. Justified or overfit?")
p("4. The 11 MNQ validated strategies use composite filters (ATR70_VOL, X_MES_ATR).")
p("   Their ExpR is 0.32-0.37. A null test is determining the MNQ noise ceiling.")
p("   What threshold would YOU use to distinguish signal from noise?")
p("5. MNQ effect size is +0.12R per trade (~$5-15). Viable on $50K prop, 1 contract?")
p("6. What alternative signals would you test given only 1m OHLCV futures bar data?")
p("7. Are you confident enough in any of these numbers to risk real money? Why/why not?")

# Write to file
text = "\n".join(out)
with open("docs/plans/2026-03-19-zero-context-audit-data.md", "w") as f:
    f.write(text)

p()
p(f"=== Written {len(out)} lines to docs/plans/2026-03-19-zero-context-audit-data.md ===")
p()
p("SUMMARY:")
p(f"  Total BH FDR tests: {m}")
p(f"  Positive survivors: {len(pos_surv)}")
by_inst = {}
for t in pos_surv:
    by_inst[t["inst"]] = by_inst.get(t["inst"], 0) + 1
for inst in sorted(ACTIVE_ORB_INSTRUMENTS):
    p(f"    {inst}: {by_inst.get(inst, 0)}")
p(f"  Negative survivors: {len(neg_surv)}")

con.close()
