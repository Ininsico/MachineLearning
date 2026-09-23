raw <- readline("Enter a number (default 5): ")
n <- if (nchar(raw) == 0) 5 else as.integer(raw)

if (n < 0) {
  cat("Factorial is not defined for negative numbers.\n")
} else {
  factorial_value <- 1
  for (i in seq_len(n)) {
    factorial_value <- factorial_value * i
  }

  cat(sprintf("%d! = %g\n", n, factorial_value))
}
