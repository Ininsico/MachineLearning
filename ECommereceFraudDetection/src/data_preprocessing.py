"""
Task 01: Data Preprocessing
Handles missing values, duplicates, and data cleaning
"""

import pandas as pd
import numpy as np
import os

def load_data():
    """Load the dataset from CSV file"""
    print("Loading dataset...")
    
    # Create sample data if file doesn't exist (for testing)
    if not os.path.exists('data/raw/ecommerce_fraud_data.csv'):
        print("Creating sample dataset for demonstration...")
        create_sample_data()
    
    df = pd.read_csv('data/raw/ecommerce_fraud_data.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def create_sample_data():
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'user_id': np.arange(1000, 1000 + n_samples),
        'account_age_days': np.random.randint(1, 1000, n_samples),
        'total_transactions_user': np.random.randint(1, 200, n_samples),
        'avg_amount_user': np.round(np.random.uniform(10, 500, n_samples), 2),
        'amount': np.round(np.random.exponential(100, n_samples), 2),
        'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France'], n_samples),
        'bin_country': np.random.choice(['USA', 'UK', 'Canada', 'Other'], n_samples),
        'channel': np.random.choice(['web', 'app'], n_samples, p=[0.6, 0.4]),
        'merchant_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Other'], n_samples),
        'promo_used': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'avs_match': np.random.choice(['Y', 'N'], n_samples, p=[0.8, 0.2]),
        'cvv_result': np.random.choice(['M', 'N', 'P'], n_samples, p=[0.85, 0.1, 0.05]),
        'three_ds_flag': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'shipping_distance_km': np.round(np.random.uniform(1, 1000, n_samples), 2),
        'transaction_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    # Add some missing values
    df = pd.DataFrame(data)
    for col in ['amount', 'shipping_distance_km', 'avg_amount_user']:
        idx = np.random.choice(df.index, size=int(n_samples*0.05), replace=False)
        df.loc[idx, col] = np.nan
    
    # Add some duplicates
    duplicates = df.sample(20, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    df.to_csv('data/raw/ecommerce_fraud_data.csv', index=False)
    print("Sample dataset created successfully!")

def handle_missing_values(df):
    """Handle missing values using appropriate techniques"""
    print("\nHandling missing values...")
    
    # Check for missing values
    missing = df.isnull().sum()
    print("Missing values before handling:")
    print(missing[missing > 0])
    
    # Numerical columns: Fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical columns: Fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("\nMissing values after handling:")
    print(df.isnull().sum().sum(), "missing values remaining")
    return df

def remove_duplicates(df):
    """Remove duplicate records"""
    print("\nRemoving duplicates...")
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    final_rows = df.shape[0]
    print(f"Removed {initial_rows - final_rows} duplicate rows")
    return df

def data_type_conversion(df):
    """Convert data types where necessary"""
    print("\nConverting data types...")
    
    # Convert transaction_time to datetime
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    
    # Create hour feature from transaction time
    if 'transaction_time' in df.columns:
        df['transaction_hour'] = df['transaction_time'].dt.hour
    
    return df

def save_cleaned_data(df):
    """Save the cleaned dataset"""
    print("\nSaving cleaned data...")
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    
    # Create summary statistics
    summary = {
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'fraud_rate': df['is_fraud'].mean() * 100,
        'duplicates_removed': 0  # We'll calculate this properly
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('data/processed/data_summary.csv', index=False)
    
    print("Cleaned data saved to 'data/processed/cleaned_data.csv'")
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Fraud percentage: {summary['fraud_rate']:.2f}%")

def main():
    """Main function to run all preprocessing steps"""
    print("\n" + "="*60)
    print("TASK 01: DATA PREPROCESSING")
    print("="*60)
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Show initial info
    print("\nInitial Dataset Info:")
    print(df.info())
    
    # Perform preprocessing steps
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = data_type_conversion(df)
    
    # Save cleaned data
    save_cleaned_data(df)
    
    return df

if __name__ == "__main__":
    main()