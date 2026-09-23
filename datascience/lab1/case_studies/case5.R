fmt_money <- function(x) formatC(x, format = "f", digits = 2, big.mark = ",")

for (i in 1:5) {
  cat(sprintf("\n--- Employee %d ---\n", i))

  raw <- readline(sprintf("Name (default Employee %d): ", i))
  name <- if (nchar(raw) == 0) paste("Employee", i) else raw

  raw <- readline("Basic Salary (default 50000): ")
  basic <- if (nchar(raw) == 0) 50000 else as.numeric(raw)

  raw <- readline("Years of Experience (default 3): ")
  years <- if (nchar(raw) == 0) 3 else as.numeric(raw)

  if (years < 2) {
    allowance_rate <- 0.05
  } else if (years <= 5) {
    allowance_rate <- 0.10
  } else {
    allowance_rate <- 0.15
  }

  if (basic > 100000) {
    bonus_rate <- 0.10
  } else {
    bonus_rate <- 0.05
  }

  allowance <- basic * allowance_rate
  bonus <- basic * bonus_rate
  gross <- basic + allowance + bonus

  cat(sprintf("Employee Name: %s\n", name))
  cat(sprintf("Basic Salary: Rs. %s\n", fmt_money(basic)))
  cat(sprintf("Experience: %g year(s)\n", years))
  cat(sprintf("Allowance (%.0f%%): Rs. %s\n", allowance_rate * 100, fmt_money(allowance)))
  cat(sprintf("Bonus (%.0f%%): Rs. %s\n", bonus_rate * 100, fmt_money(bonus)))
  cat(sprintf("Gross Salary: Rs. %s\n", fmt_money(gross)))
}
