# 10.	Download netflix_titles.csv from Kaggle and place it in C:/DataScienceLab/Data/Netflix/.
# 11.	Load the file in R and Python.
# 12.	Display the first 10 records.
# 13.	Find the dimensions of the dataset.
# 14.	List all column names.
# 15.	Count missing values in every column.
# 16.	Count how many rows represent Movies and how many represent TV Shows.
# 17.	Find the minimum and maximum release year.
# 18.	Save only the acquired raw dataset as netflix_acquired_copy.csv.

from pathlib import Path

import pandas as pd

lab = Path(__file__).resolve().parents[1]
netflix = pd.read_csv(lab / "data" / "netflix_titles.csv")

print(netflix.head(10))
print(netflix.shape)
print(list(netflix.columns))
print(netflix.isnull().sum())
print(netflix['type'].value_counts())
print(netflix['release_year'].min())
print(netflix['release_year'].max())
netflix.to_csv(lab / "results" / "netflix_acquired_copy.csv", index=False)
