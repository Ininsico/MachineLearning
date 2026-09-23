raw <- readline("How many terms? (default 10): ")
n <- if (nchar(raw) == 0) 10 else as.integer(raw)

a <- 0
b <- 1

cat("Fibonacci Series:")
for (i in seq_len(n)) {
  cat(sprintf(" %g", a))
  temp <- a + b
  a <- b
  b <- temp
}
cat("\n")
