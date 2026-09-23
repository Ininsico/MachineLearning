years = [2020, 2021, 2024, 2100, 2000]

for year in years:
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        status = "Leap year"
    else:
        status = "Not a leap year"

    print(f"{year}: {status}")
