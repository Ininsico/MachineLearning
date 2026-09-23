raw <- readline("Enter units consumed (default 250): ")
units <- if (nchar(raw) == 0) 250 else as.numeric(raw)

if (units <= 100) {
  rate <- 10
} else if (units <= 200) {
  rate <- 15
} else if (units <= 300) {
  rate <- 20
} else {
  rate <- 25
}

bill <- units * rate
service_charge <- 500
total <- bill + service_charge

cat(sprintf("Units Consumed: %g\n", units))
cat(sprintf("Rate Applied: Rs. %g/unit\n", rate))
cat(sprintf("Energy Bill: Rs. %.2f\n", bill))
cat(sprintf("Service Charge: Rs. %.2f\n", service_charge))
cat(sprintf("Total Payable: Rs. %.2f\n", total))
