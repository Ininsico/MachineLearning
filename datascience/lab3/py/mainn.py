from pathlib import Path

import pandas as pd

file_path = Path(__file__).resolve().parents[1] / "data" / "main.txt"

df = pd.read_csv(file_path)
print(df.head())