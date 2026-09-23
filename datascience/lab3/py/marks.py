from pathlib import Path
import pandas as pd

file_path = Path(__file__).resolve().parents[1] / "data" / "student_marks.json"

students = pd.read_json(file_path)
print(students.head())
print(list(students.columns))
print(students.dtypes)
print(students['Marks'].mean())

csv_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "students.csv")
print(csv_df['Marks'].mean())