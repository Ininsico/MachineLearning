# 19.	Identify the delimiter/structure of each file: csv = comma, tsv = tab,
# 	txt = fixed width (columns occupy character positions 1-12, 13-28, 29-36, 37-42),
# 	xlsx = workbook whose records live in the "Data" sheet only.
# 20.	Acquire each file in Python.
# 21.	Compare row counts and column names.
# 22.	Check whether the four files represent the same number of records.
# 23.	Calculate the average Marks field from each imported object.
# 24.	Show what an incorrect separator or sheet selection does to the results.

from pathlib import Path

import pandas as pd

lab = Path(__file__).resolve().parents[1]
data = lab / "data"

csv_df = pd.read_csv(data / "students.csv")
tsv_df = pd.read_csv(data / "students.tsv", sep="\t")
txt_df = pd.read_fwf(data / "students.txt", widths=[12, 16, 8, 6])
xlsx_df = pd.read_excel(data / "students.xlsx", sheet_name="Data")

frames = {"csv": csv_df, "tsv": tsv_df, "txt": txt_df, "xlsx": xlsx_df}

for name, df in frames.items():
    print(f"  {name:<5} rows={len(df):<3} columns={list(df.columns)}")
print("  same row count:", len({len(df) for df in frames.values()}) == 1)
print("  same columns:  ", len({tuple(df.columns) for df in frames.values()}) == 1)

print("23. average Marks")
for name, df in frames.items():
    print(f"  {name:<5} avg Marks = {df['Marks'].mean():.2f}")

print("24. incorrect separator / sheet selection")
wrong = {
    "tsv read with comma": pd.read_csv(data / "students.tsv"),
    "txt read with comma": pd.read_csv(data / "students.txt"),
    "xlsx wrong sheet ('Notes')": pd.read_excel(data / "students.xlsx", sheet_name="Notes"),
}
for name, df in wrong.items():
    print(f"  {name:<28} rows={len(df):<3} columns={list(df.columns)}")
    print(f"  {'':<28} avg Marks -> ", end="")
    try:
        print(df["Marks"].mean())
    except KeyError:
        print("KeyError: no 'Marks' column")
