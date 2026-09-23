print("1. Celsius to Fahrenheit")
print("2. Fahrenheit to Celsius")

choice = input("Enter your choice (1-2, default 1): ").strip() or "1"

if choice == "1":
    celsius = float(input("Enter temperature in Celsius (default 25): ").strip() or 25)
    fahrenheit = (celsius * 9 / 5) + 32
    print(f"{celsius:.2f} C = {fahrenheit:.2f} F")
elif choice == "2":
    fahrenheit = float(input("Enter temperature in Fahrenheit (default 77): ").strip() or 77)
    celsius = (fahrenheit - 32) * 5 / 9
    print(f"{fahrenheit:.2f} F = {celsius:.2f} C")
else:
    print("Invalid choice.")
