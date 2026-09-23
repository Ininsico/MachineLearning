cat("1. Celsius to Fahrenheit\n")
cat("2. Fahrenheit to Celsius\n")

raw <- readline("Enter your choice (1-2, default 1): ")
choice <- if (nchar(raw) == 0) "1" else raw

if (choice == "1") {
  raw <- readline("Enter temperature in Celsius (default 25): ")
  celsius <- if (nchar(raw) == 0) 25 else as.numeric(raw)
  fahrenheit <- (celsius * 9 / 5) + 32
  cat(sprintf("%.2f C = %.2f F\n", celsius, fahrenheit))

} else if (choice == "2") {
  raw <- readline("Enter temperature in Fahrenheit (default 77): ")
  fahrenheit <- if (nchar(raw) == 0) 77 else as.numeric(raw)
  celsius <- (fahrenheit - 32) * 5 / 9
  cat(sprintf("%.2f F = %.2f C\n", fahrenheit, celsius))

} else {
  cat("Invalid choice.\n")
}
