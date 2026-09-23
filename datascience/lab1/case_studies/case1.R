assignment <- 80
midterm <- 70
final_exam <- 75

final_score <- (assignment * 0.15) + (midterm * 0.25) + (final_exam * 0.60)

cat(sprintf("Assignment: %d\n", assignment))
cat(sprintf("Midterm: %d\n", midterm))
cat(sprintf("Final Exam: %d\n", final_exam))
cat(sprintf("Final Score: %.2f\n", final_score))

if (final_score >= 90) {
  grade <- "A"
} else if (final_score >= 80) {
  grade <- "B"
} else if (final_score >= 70) {
  grade <- "C"
} else if (final_score >= 60) {
  grade <- "D"
} else {
  grade <- "F"
}

cat(sprintf("Grade: %s\n", grade))
