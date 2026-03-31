"""Compare TopStep account sizes for max extraction."""

worst_day = 323  # all 8 sessions lose at 1 ct, S0.75 risk

tiers = [
    ("50K", 2000, 50, 50),  # (name, dd, micro_max, reset_cost)
    ("100K", 3000, 100, 100),
    ("150K", 4500, 150, 150),
]

print("TOPSTEP ACCOUNT SIZE COMPARISON (5 accounts each)")
print("=" * 80)
print()

for name, dd, max_micro, reset_cost in tiers:
    safe_cts = int(dd * 0.50 / worst_day)
    send_cts = int(dd * 0.80 / worst_day)
    actual_max = min(send_cts, max_micro)

    annual_safe = 62.5 * safe_cts * 250 * 5 * 0.85 * 0.50
    annual_send = 62.5 * actual_max * 250 * 5 * 0.85 * 0.50

    print(f"{name} (DD=${dd:,}, max {max_micro} micro, ~${reset_cost} reset):")
    print(
        f"  Safe  ({safe_cts} cts, {worst_day * safe_cts / dd * 100:.0f}% DD worst): ${annual_safe:>10,.0f}/yr realistic"
    )
    print(
        f"  Send  ({actual_max} cts, {worst_day * actual_max / dd * 100:.0f}% DD worst): ${annual_send:>10,.0f}/yr realistic"
    )

    # Scaling milestones to unlock
    if name == "50K":
        print("  Unlock: +$2K profit -> 50 micro")
    elif name == "100K":
        print("  Unlock: +$3K profit -> 100 micro")
    else:
        print("  Unlock: +$4.5K profit -> 150 micro")

    # ROI per reset dollar
    roi = annual_send / (reset_cost * 5)
    print(f"  ROI on 5 resets: {roi:.0f}x")
    print()

print("VERDICT:")
print("-" * 80)
print()
print("  Account   Send Cts   Realistic/yr   Reset Cost   ROI")
for name, dd, max_micro, reset_cost in tiers:
    actual_max = min(int(dd * 0.80 / worst_day), max_micro)
    annual = 62.5 * actual_max * 250 * 5 * 0.85 * 0.50
    roi = annual / (reset_cost * 5)
    print(f"  {name:>6}   {actual_max:>4} cts    ${annual:>10,.0f}     ${reset_cost * 5:>5}       {roi:.0f}x")

print()
print("ANSWER: 150K wins on raw profit. 50K wins on ROI per dollar risked.")
print("But since you can risk blowing and resets are cheap relative to profit,")
print("150K at max contracts = most money.")
print()

# What about mixing? 3x 150K + 2x 50K?
dd150 = 4500
cts150 = min(int(dd150 * 0.80 / worst_day), 150)
dd50 = 2000
cts50 = min(int(dd50 * 0.80 / worst_day), 50)

mix = 62.5 * cts150 * 250 * 3 * 0.85 * 0.50 + 62.5 * cts50 * 250 * 2 * 0.85 * 0.50
pure150 = 62.5 * cts150 * 250 * 5 * 0.85 * 0.50
pure50 = 62.5 * cts50 * 250 * 5 * 0.85 * 0.50

print("PORTFOLIO MIX OPTIONS:")
print(f"  5x 50K:           ${pure50:>10,.0f}/yr  (reset cost: $250)")
print(f"  5x 150K:          ${pure150:>10,.0f}/yr  (reset cost: $750)")
print(f"  3x 150K + 2x 50K: ${mix:>10,.0f}/yr  (reset cost: $550)")
