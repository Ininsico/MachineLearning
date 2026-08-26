import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import joblib
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    os.makedirs('models', exist_ok=True)
    rows = 10000
    np.random.seed(42)
    start_date = datetime.datetime(2023, 1, 1)
    
    data = []
    for i in tqdm(range(rows), desc="Generating Data"):
        data.append({
            'product_id': f'P{np.random.randint(1000, 9999)}',
            'category': np.random.choice(['Electronics', 'Groceries', 'Clothing']),
            'unit_price': np.random.uniform(5.0, 500.0),
            'cost_per_unit': np.random.uniform(2.0, 400.0),
            'daily_volume': np.random.randint(1, 1000),
            'timestamp': start_date + datetime.timedelta(days=i % 180),
            'is_promotion': np.random.choice(['yes', 'no'], p=[0.2, 0.8]),
            'customer_type': np.random.choice(['retail', 'wholesale'])
        })
    df = pd.DataFrame(data)
    
    pattern_map = {0: 'High_Velocity', 1: 'Low_Velocity', 2: 'Seasonal', 3: 'Erratic'}
    df['consumption_pattern'] = np.random.choice(list(pattern_map.keys()), rows)
    df['next_30day_profit'] = (df['unit_price'] - df['cost_per_unit']) * df['daily_volume'] * 30 * np.random.uniform(0.8, 1.2, rows)
    
    cat_cols = ['category', 'is_promotion', 'customer_type']
    num_cols = ['unit_price', 'cost_per_unit', 'daily_volume']
    
    X = df[cat_cols + num_cols]
    y_clf = df['consumption_pattern']
    y_reg = df['next_30day_profit']
    
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols), 
        ('cat', cat_pipe, cat_cols)
    ])
    
    X_trans = preprocessor.fit_transform(X)
    preprocessor.feature_names_in_ = np.array(X.columns)
    
    with tqdm(total=1, desc="Saving Preprocessor") as pbar:
        joblib.dump(preprocessor, 'models/preprocessor.joblib')
        joblib.dump(pattern_map, 'models/pattern_map.joblib')
        pbar.update(1)
    
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, objective='multi:softprob', random_state=42)
    with tqdm(total=100, desc="Training Classifier") as pbar:
        clf.fit(X_trans, y_clf)
        pbar.update(100)
        
    reg = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    with tqdm(total=100, desc="Training Regressor") as pbar:
        reg.fit(X_trans, y_reg)
        pbar.update(100)
    
    with tqdm(total=2, desc="Saving Feature Importances") as pbar:
        plt.figure()
        plt.bar(num_cols + cat_cols, clf.feature_importances_)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('models/clf_importance.png')
        pbar.update(1)
        
        plt.figure()
        plt.bar(num_cols + cat_cols, reg.feature_importances_)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('models/reg_importance.png')
        pbar.update(1)
    
    init_types = [('float_input', FloatTensorType([None, X_trans.shape[1]]))]
    
    with tqdm(total=2, desc="Exporting ONNX Models") as pbar:
        clf_onnx = onnxmltools.convert_xgboost(clf, initial_types=init_types, target_opset=12)
        with open('models/classifier.onnx', 'wb') as f:
            f.write(clf_onnx.SerializeToString())
        pbar.update(1)
            
        reg_onnx = onnxmltools.convert_xgboost(reg, initial_types=init_types, target_opset=12)
        with open('models/regressor.onnx', 'wb') as f:
            f.write(reg_onnx.SerializeToString())
        pbar.update(1)

if __name__ == '__main__':
    main()
