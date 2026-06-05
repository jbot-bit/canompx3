"""AUDIT 01 - SESSION MAP. Read-only. Canonical SESSION_CATALOG + orb_utc_window."""
from datetime import date
from zoneinfo import ZoneInfo
from pipeline.dst import (
    SESSION_CATALOG, DYNAMIC_ORB_RESOLVERS, orb_utc_window, is_us_dst, is_uk_dst,
)

BRIS = ZoneInfo("Australia/Brisbane")
PROBE_DATES = [date(2026,1,15), date(2026,6,1), date(2026,10,15), date(2026,11,15)]

print("="*100)
print("AUDIT 01 - SESSION MAP (canonical SESSION_CATALOG + orb_utc_window)")
print("="*100)
print(f"Total sessions in SESSION_CATALOG: {len(SESSION_CATALOG)}")
for d in PROBE_DATES:
    print(f"  {d}: US_DST={is_us_dst(d)!s:5}  UK_DST={is_uk_dst(d)!s:5}")
print()

hdr = f"{'SESSION':<16}{'TYPE':<9}{'GRP':<8}"
for d in PROBE_DATES:
    hdr += f"{str(d):<13}"
hdr += " EVENT"
print(hdr)
print("-"*120)

shift_flags = {}
for label, spec in SESSION_CATALOG.items():
    row = f"{label:<16}{spec['type']:<9}{spec.get('break_group',''):<8}"
    bris_times = []
    for d in PROBE_DATES:
        h, m = DYNAMIC_ORB_RESOLVERS[label](d)
        bris_times.append((h,m))
        row += f"{h:02d}:{m:02d}        "
    row += f" {spec.get('event','')}"
    print(row)
    shift_flags[label] = (len(set(bris_times))>1, sorted(set(bris_times)))

print()
print("SHIFT ANALYSIS (does Brisbane local ORB start move across the 4 probe dates?):")
print("-"*100)
for label,(shifts,distinct) in shift_flags.items():
    kind = "FIXED-BRISBANE" if not shifts else "SHIFTS w/ overseas DST"
    times = ", ".join(f"{h:02d}:{m:02d}" for h,m in distinct)
    print(f"  {label:<16} {kind:<24} times={times}")

print()
print("ALIAS CHECK:")
print("-"*100)
alias_types = [l for l,s in SESSION_CATALOG.items() if s.get('type')=='alias']
numeric = [l for l in SESSION_CATALOG if l.isdigit()]
print(f"  type=='alias' entries: {alias_types}")
print(f"  pure-numeric labels (0900/1000/1100): {numeric}")
print(f"  digit-bearing CURRENT labels (NOT legacy): US_DATA_830, US_DATA_1000, BRISBANE_1025")

print()
print("FULL UTC WINDOW SANITY (orb_utc_window, 5-min, 2026-06-01):")
print("-"*100)
d = date(2026,6,1)
for label in SESSION_CATALOG:
    try:
        us,ue = orb_utc_window(d, label, 5)
        bs = us.astimezone(BRIS)
        print(f"  {label:<16} UTC {us.strftime('%Y-%m-%d %H:%M')} -> Bris {bs.strftime('%Y-%m-%d %H:%M (%a)')}")
    except Exception as e:
        print(f"  {label:<16} ERROR: {e}")
