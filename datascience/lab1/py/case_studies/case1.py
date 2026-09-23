assignment = 80
midterm = 70
final_exam = 75

final_score = (assignment * 0.15) + (midterm * 0.25) + (final_exam * 0.60)

print(f"Assignment: {assignment}")
print(f"Midterm: {midterm}")
print(f"Final Exam: {final_exam}")
print(f"Final Score: {final_score:.2f}")

if final_score >= 90:
    grade = "A"
elif final_score >= 80:
    grade = "B"
elif final_score >= 70:
    grade = "C"
elif final_score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Grade: {grade}")
