from pathlib import Path

import pandas as pd

lab = Path(__file__).resolve().parents[1]
file_path = lab / "data" / "main.xlsx"
headerless = {"Sheet3"}

data = {}
for sheet in ["Sheet1", "midterm result", "Final", "Sheet3"]:
    if sheet in headerless:
        data[sheet] = pd.read_excel(file_path, sheet_name=sheet, header=None)
    else:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = df.dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        data[sheet] = df

for sheet, df in data.items():
    tsv_path = lab / "results" / f"main_{sheet.replace(' ', '_').lower()}.tsv"
    df.to_csv(tsv_path, sep="\t", index=False, header=sheet not in headerless)
    print(f"{tsv_path.name}  {df.shape}")

with pd.ExcelWriter(lab / "results" / "main_clean.xlsx", engine="openpyxl") as writer:
    for sheet, df in data.items():
        df.to_excel(writer, sheet_name=sheet, index=False, header=sheet not in headerless)
    print("main_clean.xlsx", list(data))
