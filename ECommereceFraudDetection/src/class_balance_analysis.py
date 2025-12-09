"""
Task 03: Class Balance Analysis
Analyzes class distributions for different label types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_cleaned_data():
    """Load the cleaned dataset"""
    print("Loading cleaned data...")
    df = pd.read_csv('data/processed/cleaned_data.csv')
    return df

def analyze_continuous_labels(df):
    """Analyze continuous labels distribution"""
    print("\n1. Analyzing Continuous Labels...")
    
    # Continuous labels in the dataset
    cont_labels = ['amount', 'account_age_days', 'shipping_distance_km', 
                   'avg_amount_user', 'total_transactions_user']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(cont_labels[:6]):  # Plot first 6
        if col in df.columns:
            axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}', fontsize=12)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            
            # Calculate statistics
            mean = df[col].mean()
            median = df[col].median()
            std = df[col].std()
            axes[idx].axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
            axes[idx].axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/continuous_labels.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Continuous labels analysis saved to 'results/plots/continuous_labels.png'")

def analyze_binary_labels(df):
    """Analyze binary labels (is_fraud)"""
    print("\n2. Analyzing Binary Labels...")
    
    # Check if is_fraud exists
    if 'is_fraud' not in df.columns:
        print("Binary label 'is_fraud' not found in dataset!")
        return
    
    # Calculate class distribution
    fraud_dist = df['is_fraud'].value_counts()
    fraud_percentage = df['is_fraud'].value_counts(normalize=True) * 100
    
    print(f"Fraud Distribution:\n{fraud_dist}")
    print(f"\nFraud Percentage:\n{fraud_percentage}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    bars = ax1.bar(['Non-Fraud (0)', 'Fraud (1)'], fraud_dist.values, 
                   color=['green', 'red'], alpha=0.7)
    ax1.set_title('Fraud Cases Distribution', fontsize=14)
    ax1.set_ylabel('Count')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Pie chart
    colors = ['lightgreen', 'lightcoral']
    explode = (0, 0.1)  # explode the fraud slice
    ax2.pie(fraud_dist.values, explode=explode, labels=['Non-Fraud', 'Fraud'],
            autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
    ax2.set_title('Fraud Percentage Distribution', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results/plots/binary_labels.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Class imbalance analysis
    imbalance_ratio = fraud_dist[0] / fraud_dist[1] if fraud_dist[1] > 0 else float('inf')
    print(f"\nClass Imbalance Ratio (Non-Fraud:Ffraud): {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print("WARNING: Severe class imbalance detected!")
        print("Impact: Model may become biased towards majority class.")
        print("Solution: Consider using techniques like SMOTE, class weights, or undersampling.")
    else:
        print("Dataset is relatively balanced for binary classification.")

def analyze_multiclass_labels(df):
    """Analyze multi-class labels"""
    print("\n3. Analyzing Multi-class Labels...")
    
    # Multi-class labels in the dataset
    multiclass_labels = ['country', 'bin_country', 'channel', 'merchant_category', 'cvv_result']
    
    # Create directory for multiclass plots
    os.makedirs('results/plots/multiclass', exist_ok=True)
    
    for label in multiclass_labels:
        if label in df.columns:
            print(f"\nAnalyzing: {label}")
            
            # Get value counts
            value_counts = df[label].value_counts()
            n_classes = len(value_counts)
            
            print(f"Number of classes: {n_classes}")
            print(f"Class distribution:\n{value_counts.head(10)}")  # Show top 10
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar chart (top 10 classes)
            top_n = min(10, n_classes)
            top_classes = value_counts.head(top_n)
            
            bars = ax1.bar(range(top_n), top_classes.values, color='skyblue', alpha=0.7)
            ax1.set_title(f'Top {top_n} {label} Classes', fontsize=14)
            ax1.set_xlabel(label)
            ax1.set_ylabel('Count')
            ax1.set_xticks(range(top_n))
            ax1.set_xticklabels(top_classes.index, rotation=45, ha='right')
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Pie chart (if not too many classes)
            if n_classes <= 10:
                ax2.pie(value_counts.values, labels=value_counts.index,
                       autopct='%1.1f%%', startangle=90, 
                       textprops={'fontsize': 8})
                ax2.set_title(f'{label} Distribution', fontsize=14)
            else:
                # For too many classes, show text instead
                ax2.text(0.5, 0.5, f'{n_classes} classes\n(Too many for pie chart)',
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12)
                ax2.set_title(f'{label} has {n_classes} classes', fontsize=14)
                ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'results/plots/multiclass/{label}_distribution.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Check for class imbalance
            if n_classes > 0:
                max_count = value_counts.max()
                min_count = value_counts.min()
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                if imbalance_ratio > 10:
                    print(f"WARNING: Class imbalance detected in {label}!")
                    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
                else:
                    print(f"Classes in {label} are relatively balanced.")

def main():
    """Main function to run all class balance analysis"""
    print("\n" + "="*60)
    print("TASK 03: CLASS BALANCE ANALYSIS")
    print("="*60)
    
    # Create directories
    os.makedirs('results/plots', exist_ok=True)
    
    # Load cleaned data
    df = load_cleaned_data()
    
    # Run all analyses
    analyze_continuous_labels(df)
    analyze_binary_labels(df)
    analyze_multiclass_labels(df)
    
    print("\n" + "="*60)
    print("CLASS BALANCE ANALYSIS COMPLETED!")
    print("="*60)
    
    # Generate summary report
    generate_summary_report(df)

def generate_summary_report(df):
    """Generate a summary report of class balance"""
    print("\nGenerating Summary Report...")
    
    summary = {
        "Dataset Shape": str(df.shape),
        "Total Records": len(df),
        "Total Features": len(df.columns),
        "Binary Labels": ['is_fraud'] if 'is_fraud' in df.columns else [],
        "Multi-class Labels": [col for col in ['country', 'merchant_category', 'channel'] 
                              if col in df.columns],
        "Continuous Labels": [col for col in ['amount', 'account_age_days', 
                                            'shipping_distance_km'] if col in df.columns]
    }
    
    # Add fraud statistics
    if 'is_fraud' in df.columns:
        fraud_count = df['is_fraud'].sum()
        non_fraud_count = len(df) - fraud_count
        fraud_percentage = (fraud_count / len(df)) * 100
        
        summary["Fraud Cases"] = int(fraud_count)
        summary["Non-Fraud Cases"] = int(non_fraud_count)
        summary["Fraud Percentage"] = f"{fraud_percentage:.2f}%"
        summary["Imbalance Ratio"] = f"{non_fraud_count/fraud_count:.2f}:1" if fraud_count > 0 else "N/A"
    
    # Save summary to file
    import json
    with open('results/class_balance_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("Summary report saved to 'results/class_balance_summary.json'")
    print("\nSUMMARY:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()