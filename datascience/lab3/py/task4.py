import pandas as pd

data_py = pd.read_csv("http://localhost:8000/neo.csv")

print(data_py.head(10))
print(data_py.shape)
print(list(data_py.columns))
print(data_py.isnull().sum())