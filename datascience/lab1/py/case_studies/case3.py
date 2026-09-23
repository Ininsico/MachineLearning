marks = [45, 67, 89, 34, 76, 90, 55, 72, 61, 28]

total = 0
passed = 0
failed = 0
highest = marks[0]
lowest = marks[0]

for index, mark in enumerate(marks, start=1):
    total += mark

    if mark > highest:
        highest = mark
    if mark < lowest:
        lowest = mark

    if mark >= 50:
        passed += 1
        status = "Pass"
    else:
        failed += 1
        status = "Fail"

    print(f"Student {index}: {mark} - {status}")

average = total / len(marks)

print(f"\nTotal Marks: {total}")
print(f"Average Marks: {average:.2f}")
print(f"Highest Marks: {highest}")
print(f"Lowest Marks: {lowest}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
