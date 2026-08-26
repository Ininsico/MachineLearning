import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])
    df.columns = ['Stress', 'K', 'Cycles', 'Crack']
    return df

def preprocess_data(df):
    X = df[['Stress', 'K', 'Cycles']]
    y = df['Crack']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, X
