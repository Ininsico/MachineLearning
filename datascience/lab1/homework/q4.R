raw <- readline("Enter a number (default 7): ")
number <- if (nchar(raw) == 0) 7 else as.numeric(raw)

for (i in 1:10) {
  cat(sprintf("%g x %d = %g\n", number, i, number * i))
}
