balance = 50000.00

while True:
    print("\n--- ATM Menu ---")
    print("1. Check Balance")
    print("2. Deposit")
    print("3. Withdraw")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ").strip() or "4"

    if choice == "1":
        print(f"Current Balance: Rs. {balance:.2f}")

    elif choice == "2":
        amount = float(input("Enter deposit amount: ").strip() or 0)
        balance += amount
        print(f"Deposited: Rs. {amount:.2f}")
        print(f"New Balance: Rs. {balance:.2f}")

    elif choice == "3":
        amount = float(input("Enter withdrawal amount: ").strip() or 0)
        if amount > balance:
            print("Insufficient balance.")
            print(f"Withdrawal of Rs. {amount:.2f} denied.")
        else:
            balance -= amount
            print(f"Withdrawn: Rs. {amount:.2f}")
            print(f"New Balance: Rs. {balance:.2f}")

    elif choice == "4":
        print("Thank you for using the ATM. Goodbye!")
        break

    else:
        print("Invalid choice. Please enter a number from 1 to 4.")
