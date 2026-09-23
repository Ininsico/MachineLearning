from pathlib import Path

import pandas as pd

filepath = Path(__file__).resolve().parents[3] / "spaceexploration" / "data" / "neo.csv"
data = pd.read_csv(filepath)

print(data.head())
print(data.tail())
print(data.columns)
print(data.shape)