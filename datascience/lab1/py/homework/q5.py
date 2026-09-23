n = int(input("Enter a number (default 5): ").strip() or 5)

if n < 0:
    print("Factorial is not defined for negative numbers.")
else:
    factorial = 1
    for i in range(1, n + 1):
        factorial *= i

    print(f"{n}! = {factorial}")
