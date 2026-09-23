n = int(input("Enter a number (default 29): ").strip() or 29)

is_prime = True

if n < 2:
    is_prime = False
else:
    i = 2
    while i * i <= n:
        if n % i == 0:
            is_prime = False
            break
        i += 1

if is_prime:
    print(f"{n} is a prime number.")
else:
    print(f"{n} is not a prime number.")
