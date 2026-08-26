import matplotlib.pyplot as plt
import numpy as np

def create_bias_variance_plot(results, model_name, x_title="Model Complexity"):
    complexities = results['complexities']
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(complexities, results['loss'], 'o-', color='black', linewidth=2.5,
            markersize=8, label='Expected Loss (Total Error)')
    ax.plot(complexities, results['bias'], 's--', color='blue', linewidth=2,
            markersize=8, label='Bias')
    ax.plot(complexities, results['variance'], '^-.', color='red', linewidth=2,
            markersize=8, label='Variance')

    min_loss_idx = int(np.argmin(results['loss']))
    optimal_complexity = complexities[min_loss_idx]
    ax.axvline(x=optimal_complexity, color='green', linestyle='--', linewidth=2,
               label=f'Optimal Complexity = {optimal_complexity}')

    ax.set_xlabel(x_title, fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title(f'Bias-Variance Tradeoff: {model_name} on Breast Cancer Wisconsin',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    if isinstance(complexities[0], float):
        ax.set_xscale('log')

    plt.tight_layout()
    return fig
