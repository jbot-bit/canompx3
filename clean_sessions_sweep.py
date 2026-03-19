import duckdb
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/joshd/canompx3')

con = duckdb.connect('C:/Users/joshd/canompx3/gold.db', read_only=True)

CLEAN_SESSIONS = ['CME_REOPEN', 'LONDON_METALS']
INSTRUMENTS = ['MGC', 'MES', 'MNQ']

# ============================================================
# 1. FULL BREAKDOWN: session x instrument x direction x sweep
# ============================================================
print("=" * 100)
print("CLEAN SESSIONS ONLY: CME_REOPEN + LONDON_METALS")
print("Sweep alignment: PDH_swept+long OR PDL_swept+short = ALIGNED")
print("=" * 100)

for entry_model in ['E1', 'E2']:
    for rr in [2.0, 2.5, 3.0]:
        print(f"\n{'='*80}")
        print(f"ENTRY={entry_model} RR={rr} CB=1 ORB=5m")
        print(f"{'='*80}")

        all_rows = []
        for session in CLEAN_SESSIONS:
            dir_col = f"orb_{session}_break_dir"
            try:
                df = con.execute(f"""
                    WITH base AS (
                        SELECT
                            o.symbol,
                            o.pnl_r,
                            o.outcome,
                            o.trading_day,
                            o.mfe_r,
                            o.mae_r,
                            d.{dir_col} AS break_dir,
                            d.overnight_took_pdh,
                            d.overnight_took_pdl,
                            CASE
                                WHEN d.overnight_took_pdh = TRUE AND d.overnight_took_pdl = TRUE THEN 'BOTH'
                                WHEN d.overnight_took_pdh = TRUE THEN 'PDH'
                                WHEN d.overnight_took_pdl = TRUE THEN 'PDL'
                                ELSE 'NONE'
                            END AS sweep_type
                        FROM orb_outcomes o
                        JOIN daily_features d ON o.trading_day = d.trading_day
                            AND o.symbol = d.symbol
                            AND o.orb_minutes = d.orb_minutes
                        WHERE o.symbol IN ('MGC', 'MES', 'MNQ')
                            AND o.orb_label = '{session}'
                            AND o.orb_minutes = 5
                            AND o.entry_model = '{entry_model}'
                            AND o.confirm_bars = 1
                            AND o.rr_target = {rr}
                            AND o.pnl_r IS NOT NULL
                            AND d.{dir_col} IS NOT NULL
                    )
                    SELECT
                        symbol,
                        '{session}' as session,
                        break_dir,
                        CASE
                            WHEN (sweep_type = 'PDH' AND break_dir = 'long') THEN 'ALIGNED'
                            WHEN (sweep_type = 'PDL' AND break_dir = 'short') THEN 'ALIGNED'
                            WHEN (sweep_type = 'PDH' AND break_dir = 'short') THEN 'OPPOSED'
                            WHEN (sweep_type = 'PDL' AND break_dir = 'long') THEN 'OPPOSED'
                            WHEN sweep_type = 'BOTH' THEN 'BOTH_SWEPT'
                            ELSE 'NO_SWEEP'
                        END AS alignment,
                        sweep_type,
                        COUNT(*) as n,
                        ROUND(AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_pct,
                        ROUND(AVG(pnl_r), 4) as avg_r,
                        ROUND(MEDIAN(pnl_r), 4) as med_r,
                        ROUND(AVG(mfe_r), 4) as avg_mfe,
                        ROUND(AVG(mae_r), 4) as avg_mae,
                        ROUND(SUM(pnl_r), 2) as total_r
                    FROM base
                    GROUP BY symbol, session, break_dir, alignment, sweep_type
                    HAVING COUNT(*) >= 10
                    ORDER BY symbol, session, alignment, break_dir
                """).fetchdf()
                if len(df) > 0:
                    all_rows.append(df)
            except Exception as e:
                print(f"Error {session}: {e}")

        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)

            # Show ALIGNED
            aligned = combined[combined.alignment == 'ALIGNED'].sort_values('avg_r', ascending=False)
            print(f"\nALIGNED ({len(aligned)} combos):")
            if len(aligned) > 0:
                print(aligned[['symbol','session','sweep_type','break_dir','n','win_pct','avg_r','med_r','avg_mfe','total_r']].to_string(index=False))

            # Show OPPOSED
            opposed = combined[combined.alignment == 'OPPOSED'].sort_values('avg_r', ascending=False)
            print(f"\nOPPOSED ({len(opposed)} combos):")
            if len(opposed) > 0:
                print(opposed[['symbol','session','sweep_type','break_dir','n','win_pct','avg_r','med_r','avg_mfe','total_r']].to_string(index=False))

            # Show NO_SWEEP baseline
            nosweep = combined[combined.alignment == 'NO_SWEEP'].sort_values('avg_r', ascending=False)
            print(f"\nNO_SWEEP baseline ({len(nosweep)} combos):")
            if len(nosweep) > 0:
                print(nosweep[['symbol','session','break_dir','n','win_pct','avg_r','total_r']].to_string(index=False))

            # Show BOTH_SWEPT
            both = combined[combined.alignment == 'BOTH_SWEPT'].sort_values('avg_r', ascending=False)
            if len(both) > 0:
                print(f"\nBOTH_SWEPT ({len(both)} combos):")
                print(both[['symbol','session','break_dir','n','win_pct','avg_r','total_r']].to_string(index=False))

            # Summary spreads
            a_avg = aligned.avg_r.mean() if len(aligned) > 0 else 0
            o_avg = opposed.avg_r.mean() if len(opposed) > 0 else 0
            ns_avg = nosweep.avg_r.mean() if len(nosweep) > 0 else 0
            print(f"\nMean avgR -- ALIGNED: {a_avg:.4f}, OPPOSED: {o_avg:.4f}, NO_SWEEP: {ns_avg:.4f}, Spread: {a_avg-o_avg:.4f}")
        else:
            print("No data found for this combination.")

# ============================================================
# 2. YEARLY STABILITY for clean sessions
# ============================================================
print("\n\n" + "=" * 100)
print("YEARLY STABILITY: Clean Sessions Only (E1/CB1/RR2.5)")
print("=" * 100)

for inst in INSTRUMENTS:
    yearly_rows = []
    for session in CLEAN_SESSIONS:
        dir_col = f"orb_{session}_break_dir"
        try:
            df = con.execute(f"""
                WITH base AS (
                    SELECT
                        o.pnl_r,
                        EXTRACT(YEAR FROM o.trading_day) as yr,
                        d.{dir_col} AS break_dir,
                        d.overnight_took_pdh,
                        d.overnight_took_pdl,
                        CASE
                            WHEN d.overnight_took_pdh = TRUE AND d.overnight_took_pdl = TRUE THEN 'BOTH'
                            WHEN d.overnight_took_pdh = TRUE THEN 'PDH'
                            WHEN d.overnight_took_pdl = TRUE THEN 'PDL'
                            ELSE 'NONE'
                        END AS sweep_type
                    FROM orb_outcomes o
                    JOIN daily_features d ON o.trading_day = d.trading_day
                        AND o.symbol = d.symbol
                        AND o.orb_minutes = d.orb_minutes
                    WHERE o.symbol = '{inst}'
                        AND o.orb_label = '{session}'
                        AND o.orb_minutes = 5
                        AND o.entry_model = 'E1'
                        AND o.confirm_bars = 1
                        AND o.rr_target = 2.5
                        AND o.pnl_r IS NOT NULL
                        AND d.{dir_col} IS NOT NULL
                )
                SELECT
                    yr,
                    CASE
                        WHEN (sweep_type = 'PDH' AND break_dir = 'long') THEN 'ALIGNED'
                        WHEN (sweep_type = 'PDL' AND break_dir = 'short') THEN 'ALIGNED'
                        WHEN (sweep_type = 'PDH' AND break_dir = 'short') THEN 'OPPOSED'
                        WHEN (sweep_type = 'PDL' AND break_dir = 'long') THEN 'OPPOSED'
                        ELSE 'OTHER'
                    END AS alignment,
                    COUNT(*) as n,
                    ROUND(AVG(pnl_r), 4) as avg_r,
                    ROUND(SUM(pnl_r), 2) as total_r
                FROM base
                GROUP BY yr, alignment
                HAVING COUNT(*) >= 3
                ORDER BY yr, alignment
            """).fetchdf()
            if len(df) > 0:
                yearly_rows.append(df)
        except:
            continue

    if yearly_rows:
        combined = pd.concat(yearly_rows, ignore_index=True)
        agg = combined.groupby(['yr','alignment']).agg({'n':'sum','avg_r':'mean','total_r':'sum'}).reset_index()

        pivot_r = agg.pivot(index='yr', columns='alignment', values='avg_r')
        pivot_n = agg.pivot(index='yr', columns='alignment', values='n')

        print(f"\n{inst} (CME_REOPEN + LONDON_METALS only):")
        if 'ALIGNED' in pivot_r.columns and 'OPPOSED' in pivot_r.columns:
            display = pd.DataFrame()
            display['aligned_r'] = pivot_r['ALIGNED']
            display['opposed_r'] = pivot_r['OPPOSED']
            display['spread'] = display['aligned_r'] - display['opposed_r']
            if 'ALIGNED' in pivot_n.columns:
                display['aligned_n'] = pivot_n['ALIGNED']
            if 'OPPOSED' in pivot_n.columns:
                display['opposed_n'] = pivot_n['OPPOSED']
            print(display.to_string())
            pos = (display.spread > 0).sum()
            tot = len(display)
            print(f"Spread positive in {pos}/{tot} years")
        else:
            print("Missing ALIGNED or OPPOSED columns")
            print(pivot_r)
    else:
        print(f"\n{inst}: No yearly data found")

con.close()
