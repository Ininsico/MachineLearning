years <- c(2020, 2021, 2024, 2100, 2000)

for (year in years) {
  if ((year %% 4 == 0 && year %% 100 != 0) || year %% 400 == 0) {
    status <- "Leap year"
  } else {
    status <- "Not a leap year"
  }

  cat(sprintf("%d: %s\n", year, status))
}
