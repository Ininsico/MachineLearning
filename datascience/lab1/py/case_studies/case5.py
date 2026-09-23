for i in range(1, 6):
    print(f"\n--- Employee {i} ---")
    name = input(f"Name (default Employee {i}): ").strip() or f"Employee {i}"
    basic = float(input("Basic Salary (default 50000): ").strip() or 50000)
    years = int(input("Years of Experience (default 3): ").strip() or 3)

    if years < 2:
        allowance_rate = 0.05
    elif years <= 5:
        allowance_rate = 0.10
    else:
        allowance_rate = 0.15

    if basic > 100000:
        bonus_rate = 0.10
    else:
        bonus_rate = 0.05

    allowance = basic * allowance_rate
    bonus = basic * bonus_rate
    gross = basic + allowance + bonus

    print(f"Employee Name: {name}")
    print(f"Basic Salary: Rs. {basic:,.2f}")
    print(f"Experience: {years} year(s)")
    print(f"Allowance ({allowance_rate * 100:.0f}%): Rs. {allowance:,.2f}")
    print(f"Bonus ({bonus_rate * 100:.0f}%): Rs. {bonus:,.2f}")
    print(f"Gross Salary: Rs. {gross:,.2f}")
