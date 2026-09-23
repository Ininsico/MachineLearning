PIN <- 1234
MAX_ATTEMPTS <- 3
balance <- 50000

authenticated <- FALSE

for (attempt in 1:MAX_ATTEMPTS) {
  raw <- readline(sprintf("Enter PIN (attempt %d of %d): ", attempt, MAX_ATTEMPTS))
  entered <- if (nchar(raw) == 0) PIN else as.integer(raw)

  if (entered == PIN) {
    authenticated <- TRUE
    break
  }

  cat("Incorrect PIN.\n")
}

if (!authenticated) {
  cat("Account locked. Too many incorrect attempts.\n")
} else {
  cat("Login successful. Welcome!\n")

  repeat {
    cat("\n--- ATM Menu ---\n")
    cat("1. Check Balance\n")
    cat("2. Deposit\n")
    cat("3. Withdraw\n")
    cat("4. Exit\n")

    choice <- readline("Enter your choice (1-4): ")
    if (nchar(choice) == 0) {
      choice <- "4"
    }

    if (choice == "1") {
      cat(sprintf("Current Balance: Rs. %.2f\n", balance))

    } else if (choice == "2") {
      raw <- readline("Enter deposit amount: ")
      amount <- if (nchar(raw) == 0) 0 else as.numeric(raw)
      balance <- balance + amount
      cat(sprintf("Deposited: Rs. %.2f\n", amount))
      cat(sprintf("New Balance: Rs. %.2f\n", balance))

    } else if (choice == "3") {
      raw <- readline("Enter withdrawal amount: ")
      amount <- if (nchar(raw) == 0) 0 else as.numeric(raw)
      if (amount > balance) {
        cat("Insufficient balance.\n")
        cat(sprintf("Withdrawal of Rs. %.2f denied.\n", amount))
      } else {
        balance <- balance - amount
        cat(sprintf("Withdrawn: Rs. %.2f\n", amount))
        cat(sprintf("New Balance: Rs. %.2f\n", balance))
      }

    } else if (choice == "4") {
      cat("Thank you for using the ATM. Goodbye!\n")
      break

    } else {
      cat("Invalid choice. Please enter a number from 1 to 4.\n")
    }
  }
}
