"""AUDIT 03 - EVENT MAPPING + DST AUDIT. Read-only. Maps mission's candidate events to Brisbane."""
from datetime import date, datetime
from zoneinfo import ZoneInfo
from pipeline.dst import SESSION_CATALOG, DYNAMIC_ORB_RESOLVERS

BRIS = ZoneInfo("Australia/Brisbane")
TOK = ZoneInfo("Asia/Tokyo")
SGP = ZoneInfo("Asia/Singapore")
HK = ZoneInfo("Asia/Hong_Kong")
SYD = ZoneInfo("Australia/Sydney")  # ASX
NY = ZoneInfo("America/New_York")
CH = ZoneInfo("America/Chicago")

def to_bris(tz, h, m, d):
    return datetime(d.year,d.month,d.day,h,m,tzinfo=tz).astimezone(BRIS)

print("="*100)
print("AUDIT 03a - CANDIDATE MARKET EVENT -> BRISBANE LOCAL (true exchange TZs)")
print("="*100)
# (name, tz, hour, min, note)
events = [
    ("ASX cash open",        SYD, 10, 0,  "Sydney 10:00 (ASX has DST: AEST/AEDT!)"),
    ("Tokyo open",           TOK, 9, 0,   "JST 09:00 (no DST)"),
    ("HK pre-open auction",  HK, 9, 0,    "HKT 09:00 order input"),
    ("HK pre-open match",    HK, 9, 20,   "HKT 09:20 matching"),
    ("HK cash open",         HK, 9, 30,   "HKT 09:30 continuous"),
    ("Singapore SGX open",   SGP, 9, 0,   "SGT 09:00"),
    ("CME futures reopen",   CH, 17, 0,   "CT 17:00 (Globex)"),
    ("US data 8:30",         NY, 8, 30,   "ET 08:30"),
    ("NYSE open 9:30",       NY, 9, 30,   "ET 09:30"),
]
probe = [date(2026,1,15), date(2026,6,1), date(2026,11,15)]
print(f"  {'EVENT':<22}{'NOTE':<40}", end="")
for d in probe: print(f"{str(d)[5:]:<10}", end="")
print()
print("-"*100)
for name, tz, h, m, note in events:
    line = f"  {name:<22}{note:<40}"
    for d in probe:
        b = to_bris(tz,h,m,d)
        line += f"{b.strftime('%H:%M'):<10}"
    print(line)

print()
print("="*100)
print("AUDIT 03b - DST: which Brisbane-morning slots does the CATALOG actually anchor?")
print("="*100)
print("Brisbane-morning band = 08:00..12:00 local. Sessions whose ORB start lands there:")
for d in probe:
    print(f"\n  --- {d} (US_DST handled by resolver) ---")
    hits = []
    for label in SESSION_CATALOG:
        hh, mm = DYNAMIC_ORB_RESOLVERS[label](d)
        if 8 <= hh < 12:
            hits.append((hh, mm, label))
    for hh, mm, label in sorted(hits):
        print(f"    {hh:02d}:{mm:02d}  {label}")

print()
print("="*100)
print("AUDIT 03c - KEY DST OBSERVATIONS")
print("="*100)
print("""
 - ASX (Sydney) observes DST: AEDT (UTC+11) Oct-Apr, AEST (UTC+10) Apr-Oct.
   Brisbane is fixed UTC+10. So ASX 10:00 open = 09:00 Brisbane in summer (AEDT),
   10:00 Brisbane in winter (AEST). This is a SHIFTING event with NO catalog session.
 - Tokyo 09:00 JST = 10:00 Brisbane year-round (JST no DST). Catalog TOKYO_OPEN = 10:00 OK.
 - Singapore 09:00 SGT = 11:00 Brisbane year-round (SGT no DST). Catalog SINGAPORE_OPEN = 11:00 OK.
 - HK 09:00 HKT = 11:00 Brisbane (HKT=SGT=UTC+8). HK cash open 09:30 = 11:30 Brisbane. NO catalog session at 11:30.
 - SINGAPORE_OPEN at 11:00 is labelled 'SGX/HKEX open' but HKEX continuous is 09:30 HKT = 11:30 Bris,
   NOT 09:00. The 11:00 slot = SGX open + HK PRE-open auction input, not HK continuous open.
""")
