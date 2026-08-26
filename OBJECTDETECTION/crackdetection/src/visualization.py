import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def setup_style():
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

def plot_confusion_matrix(y_true, y_probs, threshold, save_path):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Threshold = {threshold})')
    plt.savefig(save_path)
    plt.close()

def plot_performance_curves(results_df, save_path):
    plt.figure()
    plt.plot(results_df['Threshold'], results_df['Accuracy'], marker='o', label='Accuracy')
    plt.plot(results_df['Threshold'], results_df['Precision'], marker='s', label='Precision')
    plt.plot(results_df['Threshold'], results_df['Recall'], marker='^', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_feature_scatters(df, y_probs, features, save_dir):
    for feat in features:
        plt.figure()
        sns.scatterplot(x=df[feat], y=y_probs, hue=df['Crack'], palette='coolwarm')
        plt.title(f'Probability vs {feat}')
        plt.savefig(f"{save_dir}/crack_vs_{feat}.png")
        plt.close()
