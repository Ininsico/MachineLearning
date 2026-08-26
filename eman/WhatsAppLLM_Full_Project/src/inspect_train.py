import pickle

with open("data/train.pkl", "rb") as f:
    data = pickle.load(f)

print("TYPE:", type(data))
print("LENGTH:", len(data))

if isinstance(data, dict):
    print("KEYS:", data.keys())
else:
    print("FIRST ITEM TYPE:", type(data[0]))