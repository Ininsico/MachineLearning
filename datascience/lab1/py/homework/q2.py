a = float(input("Enter the first number (default 12): ").strip() or 12)
b = float(input("Enter the second number (default 45): ").strip() or 45)
c = float(input("Enter the third number (default 33): ").strip() or 33)

if a >= b and a >= c:
    largest = a
elif b >= a and b >= c:
    largest = b
else:
    largest = c

print(f"Numbers entered: {a:g}, {b:g}, {c:g}")
print(f"The largest number is: {largest:g}")
