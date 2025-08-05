import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm 
from time import sleep 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_data_with_progress():
    """Loads data with visual progress indicators"""
    print("🔍 Locating dataset...")
    for _ in tqdm(range(3), desc="Checking files"):
        sleep(0.2) 
    
    if 'final_dataset.csv' not in os.listdir():
        raise FileNotFoundError("❌ final_dataset.csv not found!")
    
    print("\n📂 Loading data...")
    with tqdm(total=100, desc="Progress") as pbar:
        data = pd.read_csv("final_dataset.csv")
        pbar.update(30) 
        sleep(0.5)
        pbar.update(70)
    print("✅ Data loaded successfully!\n")
    return data

data = load_data_with_progress()

# ======================================================================
# 2. DATA PREPROCESSING WITH PROGRESS
# ======================================================================
print("🛠️ Preprocessing data...")
preprocess_steps = [
    "Renaming columns", "Parsing dates", 
    "Dropping columns", "Splitting features"
]

with tqdm(total=len(preprocess_steps), desc="Preprocessing") as pbar:
    # Step 1
    data.rename(columns={"Date": "Day"}, inplace=True)
    pbar.set_postfix_str(preprocess_steps[0])
    pbar.update(1)
    sleep(0.3)
    
    # Step 2
    data["Date"] = pd.to_datetime(data[["Day", "Month", "Year"]])
    pbar.set_postfix_str(preprocess_steps[1])
    pbar.update(1)
    sleep(0.3)
    
    # Step 3
    features = data.drop(["Day", "Month", "Year", "Holidays_Count", "Days", "Date", "AQI"], axis=1)
    target = data["AQI"]
    pbar.set_postfix_str(preprocess_steps[2])
    pbar.update(1)
    sleep(0.3)
    
    # Step 4
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    pbar.set_postfix_str(preprocess_steps[3])
    pbar.update(1)
    sleep(0.3)

print("✅ Preprocessing complete!\n")

# ======================================================================
# 3. MODEL TRAINING WITH LIVE PROGRESS
# ======================================================================
class ProgressLinearRegression(LinearRegression):
    """Custom Linear Regression with training progress"""
    def fit(self, X, y):
        print("🧠 Training model...")
        epochs = 100  # Simulated training steps
        with tqdm(total=epochs, desc="Training") as pbar:
            super().fit(X, y)  # Actual training
            for _ in range(epochs):
                sleep(0.02)  # Simulate training iterations
                pbar.update(1)
                pbar.set_postfix_str(f"R²: {self.score(X, y):.3f}")
        return self

model = ProgressLinearRegression()
model.fit(X_train, y_train)

# ======================================================================
# 4. EVALUATION WITH VISUAL FEEDBACK
# ======================================================================
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Animated result display
for i in tqdm(range(10), desc="Calculating metrics"):
    sleep(0.1)
print(f"\n🎯 Final R² Score: {r2:.4f}")

# Feature importance visualization
importance = pd.DataFrame({
    "Feature": features.columns,
    "Importance": model.coef_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=importance, palette="viridis")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

print("\n🚀 Model training complete! Ready for predictions.")