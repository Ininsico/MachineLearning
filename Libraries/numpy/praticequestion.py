import numpy as np

# Sample data: [area, bedrooms, age, price]
data = np.array([
    [2100, 3, 20, 500000],
    [1600, 2, np.nan, 350000],
    [2400, 4, 15, 600000],
    [np.nan, 3, 10, 450000]
])
#replacing nan values with column mean
col_mean = np.nanmean(data, axis=0)
for i in range(data.shape[1]):
    data[np.isnan(data[:, i]), i] = col_mean[i]
    
# 2. Min-Max scaling (except price column)
features = data[:,: -1]
min = np.min(features,axis=0)
max = np.max(features,axis=0)
normalized_featuers = (features-min)/(max-min)
print("Processed Data:\n", normalized_featuers)