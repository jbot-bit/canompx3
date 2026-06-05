"""AUDIT 06 - SURVIVOR VALIDATION. Era split + yearly breakdown + DSR-aware sanity for the 5 positive BH survivors.
Read-only. Also checks institutional threshold (t>=3.0 Harvey-Liu cross-section; t>=3.79 Chordia)."""
import os, duckdb, math
from statistics import NormalDist
con = duckdb.connect(os.environ["DUCKDB_PATH"], read_only=True)
ND = NormalDist()

survivors = [
    ("SINGAPORE_OPEN","MNQ",30,"E2",2.0),
    ("SINGAPORE_OPEN","MNQ",30,"E2",1.5),
    ("TOKYO_OPEN","MNQ",5,"E2",1.5),
    ("TOKYO_OPEN","MNQ",15,"E2",1.0),
    ("TOKYO_OPEN","MNQ",5,"E2",1.0),
]
CB=1
def stats(where, params):
    r = con.execute(f"""
      SELECT COUNT(*), AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0 END), AVG(pnl_r), STDDEV_SAMP(pnl_r)
      FROM orb_outcomes WHERE {where} AND outcome IN ('win','loss')""", params).fetchone()
    n,wr,e,sd = r
    if not n or n<2 or not sd: return n,wr,e,None,None
    t = e/(sd/math.sqrt(n)); p = 2*(1-ND.cdf(abs(t)))
    return n,wr,e,t,p

print("="*120)
print("AUDIT 06 - SURVIVOR VALIDATION (pre-2026 era split + per-year), then 2026 HOLDOUT peek (report-only)")
print("="*120)
for sess,sym,ap,em,rr in survivors:
    base = "symbol=? AND orb_label=? AND orb_minutes=? AND entry_model=? AND rr_target=? AND confirm_bars=?"
    bp = [sym,sess,ap,em,rr,CB]
    print(f"\n### {sess} {sym} {ap}m {em} rr{rr} (cb{CB})")
    # full pre-2026
    n,wr,e,t,p = stats(base+" AND trading_day < DATE '2026-01-01'", bp)
    print(f"  PRE-2026 ALL : N={n:<5} WR={wr:.3f} ExpR={e:+.4f} t={t:.2f} p={p:.2e}")
    # era split
    for lo,hi,lbl in [("2019-01-01","2022-06-30","era1 (2019-2022H1)"),("2022-07-01","2026-01-01","era2 (2022H2-2025)")]:
        n,wr,e,t,p = stats(base+f" AND trading_day >= DATE '{lo}' AND trading_day < DATE '{hi}'", bp)
        ts = f"{t:.2f}" if t is not None else "n/a"; ps = f"{p:.2e}" if p is not None else "n/a"
        print(f"  {lbl:<20}: N={n:<5} WR={(wr or 0):.3f} ExpR={(e or 0):+.4f} t={ts} p={ps}")
    # per-year
    print("  per-year (pre-2026):")
    rows = con.execute(f"""
      SELECT CAST(strftime(trading_day,'%Y') AS INT) yr, COUNT(*), AVG(pnl_r),
             AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0 END)
      FROM orb_outcomes WHERE {base} AND trading_day < DATE '2026-01-01' AND outcome IN ('win','loss')
      GROUP BY yr ORDER BY yr""", bp).fetchall()
    yline = "    "
    pos_years = 0; tot_years = 0
    for yr,nn,ee,ww in rows:
        tot_years+=1; pos_years += 1 if (ee or 0)>0 else 0
        yline += f"{yr}:{ee:+.3f}(n{nn}) "
    print(yline)
    print(f"    positive years: {pos_years}/{tot_years}")
    # 2026 holdout (report only, do not tune)
    n,wr,e,t,p = stats(base+" AND trading_day >= DATE '2026-01-01'", bp)
    ts = f"{t:.2f}" if t is not None else "n/a"; ps = f"{p:.2e}" if p is not None else "n/a"
    print(f"  >> 2026 HOLDOUT (peek only): N={n} WR={(wr or 0):.3f} ExpR={(e or 0):+.4f} t={ts} p={ps}")
