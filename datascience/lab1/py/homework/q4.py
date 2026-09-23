number = int(input("Enter a number (default 7): ").strip() or 7)

for i in range(1, 11):
    print(f"{number} x {i} = {number * i}")
