# 1.	Download the dataset and identify the downloaded file name.
# 2.	Load it in both R and Python using a local file path.
# 3.	Display the first 10 rows.
# 4.	Report the number of rows and columns.
# 5.	Display all column names.
# 6.	Determine the data type of each column in R and Python.
# 7.	Calculate the mean mathematics, reading and writing scores.
# 8.	Check whether any missing values exist.
# 9.	Save the acquired dataset as a new CSV named student_performance_copy.csv.

from pathlib import Path

import pandas as pd

lab = Path(__file__).resolve().parents[1]
data = pd.read_csv(lab / "data" / "StudentsPerformance.csv")

print(data.head(10))

print(data.shape)

print(data.dtypes)

print(data[['math score', 'reading score', 'writing score']].mean())

print(data.isnull().sum())
data.to_csv(lab / "results" / "student_performance_copy.csv", index=False)