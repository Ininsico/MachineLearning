import os
import pandas as pd
from src.data_processing import get_initial_data, expand_dataset, scale_features
from src.modeling import train_model, analyze_thresholds, manual_sigmoid
from src.visualization import setup_style, plot_decision_boundary, plot_metrics_v_threshold, plot_confusion_matrix, plot_probability_contours, plot_3d_relationship

def main():
    if not os.path.exists('plots'): os.makedirs('plots')
    setup_style()
    
    # Part A & B
    df_initial = get_initial_data()
    df_expanded = expand_dataset(df_initial, 50)
    df_expanded.to_csv('data/machine_data_50.csv', index=False)
    
    # Modeling
    X = df_expanded[['Temperature', 'Vibration']]
    y = df_expanded['Failure']
    X_scaled, scaler = scale_features(X)
    
    model = train_model(X_scaled, y)
    
    # Coefficients
    b0 = model.intercept_[0]
    coeffs = model.coef_[0]
    print(f"Intercept: {b0:.4f}")
    print(f"Coefficients: {coeffs}")
    
    # Threshold Analysis
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = analyze_thresholds(model, X_scaled, y, thresholds)
    print("\nThreshold Analysis Results:")
    print(results[['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1']])
    
    # Visualizations
    plot_decision_boundary(X, y, model, scaler, 0.4, 'plots/decision_boundary_04.png')
    plot_probability_contours(X, y, model, scaler, 'plots/probability_contours.png')
    plot_metrics_v_threshold(results, 'plots/metrics_plot.png')
    
    row_04 = results[results['Threshold'] == 0.4].iloc[0]
    plot_confusion_matrix(int(row_04['TN']), int(row_04['FP']), int(row_04['FN']), int(row_04['TP']), 0.4, 'plots/cm_04.png')
    
    all_probs = model.predict_proba(X_scaled)[:, 1]
    plot_3d_relationship(X, y, all_probs, 'plots/3d_scatter.png')
    
    # Manual Calculation demo
    test_point = [[82, 4.0]]
    test_point_scaled = scaler.transform(test_point)[0]
    prob_manual = manual_sigmoid(b0, coeffs, test_point_scaled)
    print(f"\nManual Prob for [82, 4.0]: {prob_manual:.4f}")

if __name__ == "__main__":
    main()
