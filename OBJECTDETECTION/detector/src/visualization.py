import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def setup_style():
    sns.set_theme(style="whitegrid", palette="muted")

def plot_decision_boundary(X, y, model, scaler, threshold, filename):
    h = .05
    x_min, x_max = X.iloc[:, 0].min() - 5, X.iloc[:, 0].max() + 5
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = (model.predict_proba(grid_scaled)[:, 1] >= threshold).astype(int)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn_r')
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='RdYlGn_r', s=60, edgecolor='k')
    plt.title(f'Decision Boundary (Threshold={threshold})')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Vibration (mm/s)')
    plt.savefig(filename)
    plt.close()

def plot_metrics_v_threshold(results_df, filename):
    plt.figure(figsize=(10, 6))
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        plt.plot(results_df['Threshold'], results_df[metric], marker='o', label=metric)
    plt.title('Performance Metrics vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(tn, fp, fn, tp, threshold, filename):
    cm = [[tn, fp], [fn, tp]]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe', 'Fail'], yticklabels=['Safe', 'Fail'])
    plt.title(f'Confusion Matrix (Threshold={threshold})')
    plt.savefig(filename)
    plt.close()

def plot_probability_contours(X, y, model, scaler, filename):
    h = .1
    x_min, x_max = X.iloc[:, 0].min() - 5, X.iloc[:, 0].max() + 5
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(xx, yy, Z, levels=10, cmap='RdYlGn_r', alpha=0.6)
    plt.colorbar(cp, label='Probability of Failure')
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='dark:black', s=40, alpha=0.5)
    plt.title('Failure Probability Contours')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Vibration (mm/s)')
    plt.savefig(filename)
    plt.close()

def plot_3d_relationship(X, y, probs, filename):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['green' if val == 0 else 'red' for val in y]
    ax.scatter(X['Temperature'], X['Vibration'], probs, c=colors, s=50)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Vibration (mm/s)')
    ax.set_zlabel('Probability of Failure')
    ax.set_title('3D Relationship: Features vs Failure Probability')
    plt.savefig(filename)
    plt.close()
