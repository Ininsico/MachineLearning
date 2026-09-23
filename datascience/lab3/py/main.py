from pathlib import Path

import pandas as pd

file_path = Path(__file__).resolve().parents[1] / "data" / "main.xlsx"
students = pd.read_excel(file_path, sheet_name="Sheet1")

print(students.head())
print(students.info())
print(students.shape)

marks = students.drop(columns=["S#", "Reg. #", "Name"]).apply(pd.to_numeric, errors="coerce")
print(marks.mean())
