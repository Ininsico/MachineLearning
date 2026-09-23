raw <- readline("Enter a number (default 29): ")
n <- if (nchar(raw) == 0) 29 else as.integer(raw)

is_prime <- TRUE

if (n < 2) {
  is_prime <- FALSE
} else {
  i <- 2
  while (i * i <= n) {
    if (n %% i == 0) {
      is_prime <- FALSE
      break
    }
    i <- i + 1
  }
}

if (is_prime) {
  cat(sprintf("%d is a prime number.\n", n))
} else {
  cat(sprintf("%d is not a prime number.\n", n))
}
