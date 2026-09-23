marks <- c(45, 67, 89, 34, 76, 90, 55, 72, 61, 28)

total <- 0
passed <- 0
failed <- 0
highest <- marks[1]
lowest <- marks[1]

for (i in seq_along(marks)) {
  mark <- marks[i]
  total <- total + mark

  if (mark > highest) {
    highest <- mark
  }
  if (mark < lowest) {
    lowest <- mark
  }

  if (mark >= 50) {
    passed <- passed + 1
    status <- "Pass"
  } else {
    failed <- failed + 1
    status <- "Fail"
  }

  cat(sprintf("Student %d: %d - %s\n", i, mark, status))
}

average <- total / length(marks)

cat(sprintf("\nTotal Marks: %d\n", total))
cat(sprintf("Average Marks: %.2f\n", average))
cat(sprintf("Highest Marks: %d\n", highest))
cat(sprintf("Lowest Marks: %d\n", lowest))
cat(sprintf("Passed: %d\n", passed))
cat(sprintf("Failed: %d\n", failed))
