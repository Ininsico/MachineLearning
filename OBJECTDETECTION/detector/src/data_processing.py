import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_initial_data():
    data = {
        'Temperature': [70, 80, 85, 90, 75, 95],
        'Vibration': [2.5, 3.0, 4.5, 5.0, 3.5, 6.0],
        'Failure': [0, 0, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)

def expand_dataset(df_initial, target_size=50):
    np.random.seed(42)
    current_size = len(df_initial)
    needed = target_size - current_size
    
    temp_synth = np.random.uniform(65, 100, needed)
    vib_synth = np.random.uniform(2.0, 7.0, needed)
    
    z = 0.5 * (temp_synth - 82) + 2.5 * (vib_synth - 4.2) + np.random.normal(0, 1.2, needed)
    prob = 1 / (1 + np.exp(-z))
    failure_synth = (prob > 0.5).astype(int)
    
    df_synth = pd.DataFrame({
        'Temperature': temp_synth,
        'Vibration': vib_synth,
        'Failure': failure_synth
    })
    
    return pd.concat([df_initial, df_synth]).reset_index(drop=True)

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
