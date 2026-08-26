import os, json, time, sys
sys.path.insert(0, '.')
os.environ['STREAMLIT_RUN'] = 'false'

from src.models.dataset import BreastCancerDataset
from src.models.decision_tree import DecisionTreeModel
from src.models.knn import KNNModel
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.view.plots import create_bias_variance_plot

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading Breast Cancer Wisconsin dataset...")
X_train, X_test, y_train, y_test = BreastCancerDataset.get_data()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: 30, Classes: 2 (malignant/benign)\n")

models = [
    ("Decision Tree",       DecisionTreeModel(),       [1, 2, 3, 5, 8, 12, 16]),
    ("Random Forest",       RandomForestModel(),       [1, 2, 3, 5, 8, 12, 16]),
    ("KNN",                 KNNModel(),                [20, 15, 10, 5, 3, 2, 1]),
    ("Logistic Regression", LogisticRegressionModel(), [0.001, 0.01, 0.1, 1, 10, 100]),
]

all_results = {}

for name, model, complexities in models:
    print(f"[{name}] Running {len(complexities)} complexities...")
    start = time.time()
    try:
        results = model.compute_bias_variance(X_train, y_train, X_test, y_test, complexities)
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Loss:     {[round(v, 4) for v in results['loss']]}")
        print(f"  Bias:     {[round(v, 4) for v in results['bias']]}")
        print(f"  Variance: {[round(v, 4) for v in results['variance']]}")

        entry = {
            'model': name,
            'time_sec': round(elapsed, 1),
            'complexities': complexities,
            'loss': [round(v, 6) for v in results['loss']],
            'bias': [round(v, 6) for v in results['bias']],
            'variance': [round(v, 6) for v in results['variance']],
        }

        fname = name.lower().replace(' ', '_')
        with open(os.path.join(RESULTS_DIR, f'{fname}.json'), 'w') as f:
            json.dump(entry, f, indent=2)

        fig = create_bias_variance_plot(results, model.name,
            "Max Depth (Higher = More Complex)" if "Forest" in name or "Tree" in name
            else "K (Lower = More Complex)" if "KNN" in name
            else "C (Higher = Less Regularization)")
        fig.savefig(os.path.join(RESULTS_DIR, f'{fname}.png'), dpi=150)
        import matplotlib.pyplot as plt; plt.close(fig)

        all_results[name] = entry

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR after {elapsed:.1f}s: {e}")
        all_results[name] = {'model': name, 'error': str(e)}

with open(os.path.join(RESULTS_DIR, '_all_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nAll results saved to:", RESULTS_DIR)
