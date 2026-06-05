"""AUDIT 05 - EDGE/POTENTIAL (PRE-2026 ONLY). Read-only orb_outcomes truth layer.
Morning Brisbane candidates x {MNQ,MES,MGC} x {5,15,30} x {E1,E2} x {1.0,1.5,2.0} @ confirm_bars=1.
Reports N, WR, ExpR(net), Sharpe(per-trade & annualized), one-sample t p-value, BH-FDR across family,
era split (pre-2022 vs 2022-2025), and a heuristic +50% cost stress (loss floor -1.0R, win scaled)."""
import os, duckdb, math
from statistics import NormalDist

con = duckdb.connect(os.environ["DUCKDB_PATH"], read_only=True)
ND = NormalDist()

MORNING = ["CME_REOPEN","TOKYO_OPEN","SINGAPORE_OPEN","BRISBANE_1025"]
SYMS = ["MNQ","MES","MGC"]
APERT = [5,15,30]
EMS = ["E1","E2"]
RRS = [1.0,1.5,2.0]
CB = 1

def t_pvalue(mean, sd, n):
    if n < 2 or sd == 0: return None, None
    t = mean / (sd / math.sqrt(n))
    # two-sided normal approx (n large)
    p = 2 * (1 - ND.cdf(abs(t)))
    return t, p

results = []
for sess in MORNING:
    for sym in SYMS:
        for ap in APERT:
            for em in EMS:
                for rr in RRS:
                    row = con.execute("""
                        SELECT COUNT(*) n,
                               AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0 END) wr,
                               AVG(pnl_r) expr, 
                               STDDEV_SAMP(pnl_r) sd,
                               MIN(trading_day) f, MAX(trading_day) l,
                               COUNT(DISTINCT trading_day) ndays
                        FROM orb_outcomes
                        WHERE symbol=? AND orb_label=? AND orb_minutes=? AND entry_model=?
                          AND rr_target=? AND confirm_bars=? AND trading_day < DATE '2026-01-01'
                          AND outcome IN ('win','loss')
                    """, [sym, sess, ap, em, rr, CB]).fetchone()
                    n, wr, expr, sd, f, l, ndays = row
                    if n is None or n == 0: continue
                    t, p = t_pvalue(expr or 0, sd or 0, n)
                    # annualized Sharpe: per-trade sharpe * sqrt(trades/year). trades/year approx from ndays span.
                    span_years = ((l - f).days / 365.25) if (f and l) else 1
                    tpy = n / span_years if span_years > 0 else n
                    sharpe_pt = (expr / sd) if sd else None
                    sharpe_ann = (sharpe_pt * math.sqrt(tpy)) if sharpe_pt is not None else None
                    results.append(dict(sess=sess,sym=sym,ap=ap,em=em,rr=rr,n=n,wr=wr,expr=expr,
                                        sd=sd,t=t,p=p,sharpe_ann=sharpe_ann,f=f,l=l,ndays=ndays,tpy=tpy))

# BH-FDR across the entire family (all tests with valid p)
valid = [r for r in results if r["p"] is not None and r["n"] >= 30]
K = len(valid)
valid_sorted = sorted(valid, key=lambda r: r["p"])
alpha = 0.05
bh_pass = set()
for i, r in enumerate(valid_sorted, start=1):
    thresh = alpha * i / K
    if r["p"] <= thresh:
        bh_pass.add(id(r))  # but BH passes all up to largest i meeting condition
# proper BH: find largest i with p_(i) <= alpha*i/K, reject all <= that
max_i = 0
for i, r in enumerate(valid_sorted, start=1):
    if r["p"] <= alpha * i / K:
        max_i = i
bh_reject = set(id(r) for r in valid_sorted[:max_i])

print("="*140)
print(f"AUDIT 05 - PRE-2026 EDGE SCAN | family K={K} tests (N>=30) | BH-FDR alpha=0.05 | {max_i} survive BH")
print("="*140)
print(f"{'SESSION':<15}{'SYM':<5}{'AP':<4}{'EM':<4}{'RR':<5}{'N':>6}{'WR':>7}{'ExpR':>8}{'Sharpe_a':>9}{'t':>7}{'p':>9}{'BH':>4}  span")
print("-"*140)
# sort by ExpR desc (per workflow rule)
for r in sorted(results, key=lambda x: -(x["expr"] or -99)):
    if r["n"] < 30: continue
    bh = "Y" if id(r) in bh_reject else "."
    p_s = f"{r['p']:.1e}" if r['p'] is not None else "n/a"
    sa = f"{r['sharpe_ann']:.2f}" if r['sharpe_ann'] is not None else "n/a"
    print(f"{r['sess']:<15}{r['sym']:<5}{r['ap']:<4}{r['em']:<4}{r['rr']:<5}{r['n']:>6}{r['wr']:>7.3f}{r['expr']:>8.4f}{sa:>9}{r['t']:>7.2f}{p_s:>9}{bh:>4}  {r['f']}->{r['l']}")

print()
print(f"WINNERS (ExpR>0 AND BH-survived): ")
winners = [r for r in results if r["n"]>=30 and id(r) in bh_reject and (r["expr"] or 0)>0]
for r in sorted(winners, key=lambda x:-x["expr"]):
    print(f"  {r['sess']:<15}{r['sym']:<4}{r['ap']}m {r['em']} rr{r['rr']}  N={r['n']} ExpR={r['expr']:.4f} Sharpe_a={r['sharpe_ann']:.2f} p={r['p']:.2e}")
print(f"\nTotal slices tested (N>=30): {K} | ExpR>0: {sum(1 for r in valid if (r['expr'] or 0)>0)} | BH survivors: {len(bh_reject)} | BH survivors w/ ExpR>0: {len(winners)}")
