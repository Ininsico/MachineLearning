import os
import sys
import pandas as pd
from src.data_processing import load_data, preprocess_data
from src.modeling import train_logistic_regression, calculate_unscaled_coefficients, analyze_thresholds
from src.visualization import setup_style, plot_confusion_matrix, plot_performance_curves, plot_roc_curve, plot_feature_scatters

def main():
    data_path = "data/CrackDetectiondataset.csv"
    results_dir = "results"
    plots_dir = os.path.join(results_dir, "plots")
    report_path = os.path.join(results_dir, "reports", "analysis_report.txt")

    df = load_data(data_path)
    X_scaled, y, scaler, X_orig = preprocess_data(df)
    
    model = train_logistic_regression(X_scaled, y)
    
    beta0, weights = calculate_unscaled_coefficients(model, scaler)
    results_df, y_probs = analyze_thresholds(model, X_scaled, y)
    
    setup_style()
    plot_confusion_matrix(y, y_probs, 0.5, os.path.join(plots_dir, "confusion_matrix.png"))
    plot_performance_curves(results_df, os.path.join(plots_dir, "threshold_performance.png"))
    plot_roc_curve(y, y_probs, os.path.join(plots_dir, "roc_curve.png"))
    plot_feature_scatters(df, y_probs, ['Stress', 'K', 'Cycles'], plots_dir)
    
    with open(report_path, 'w') as f:
        f.write("COEFFICIENTS\n")
        f.write(f"Intercept: {beta0:.4f}\n")
        f.write(f"Stress: {weights[0]:.4f}\n")
        f.write(f"K: {weights[1]:.4f}\n")
        f.write(f"Cycles: {weights[2]:.4f}\n\n")
        f.write("THRESHOLD ANALYSIS\n")
        f.write(results_df.to_string(index=False))

    print(f"Analysis complete. Results in {results_dir}")

if __name__ == "__main__":
    main()
