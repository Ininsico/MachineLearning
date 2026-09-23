raw <- readline("Enter the first number (default 12): ")
a <- if (nchar(raw) == 0) 12 else as.numeric(raw)

raw <- readline("Enter the second number (default 45): ")
b <- if (nchar(raw) == 0) 45 else as.numeric(raw)

raw <- readline("Enter the third number (default 33): ")
c <- if (nchar(raw) == 0) 33 else as.numeric(raw)

if (a >= b && a >= c) {
  largest <- a
} else if (b >= a && b >= c) {
  largest <- b
} else {
  largest <- c
}

cat(sprintf("Numbers entered: %g, %g, %g\n", a, b, c))
cat(sprintf("The largest number is: %g\n", largest))
