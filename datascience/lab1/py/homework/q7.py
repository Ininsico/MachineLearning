n = int(input("How many terms? (default 10): ").strip() or 10)

a = 0
b = 1

print("Fibonacci Series:", end=" ")
for i in range(n):
    print(a, end=" ")
    a, b = b, a + b
print()
