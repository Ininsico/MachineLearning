from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np 

# Load dataset
housing = fetch_california_housing()
X, Y = housing.data, housing.target

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print("RMSE (manual):", rmse)
print("R² score:", model.score(X_test, Y_test))