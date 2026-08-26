# Bias-Variance Prediction Analyzer

Interactive tool that decomposes model error into **bias** and **variance** across complexity levels, using the **Breast Cancer Wisconsin** dataset and 4 ML models.

---

## The Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Error

  High Bias (underfitting)     |     High Variance (overfitting)
  - Model too simple           |     - Model too complex
  - Misses patterns in data    |     - Chases noise in data
  - Train & test error both    |     - Low train error, high test
    high                       |       error gap
```

The **optimal complexity** is where total error is minimized — the sweet spot between underfitting and overfitting.

---

## Dataset: Breast Cancer Wisconsin

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (real-valued) |
| Classes | 2 — malignant / benign |
| Source | UCI ML Repository (`sklearn.datasets.load_breast_cancer`) |

No subsampling required — the full dataset is used for every run.

---

## Models

Each model is trained across a range of its complexity parameter. The table below shows the parameter that controls model capacity and the values used in the analysis.

| Model | Complexity Parameter | Values Tested | Complexity Direction |
|---|---|---|---|
| **Decision Tree** | `max_depth` | `[1, 2, 3, 5, 8, 12, 16]` | Higher depth = more complex |
| **Random Forest** | `max_depth` | `[1, 2, 3, 5, 8, 12, 16]` | Higher depth = more complex |
| **K-Nearest Neighbors** | `n_neighbors` (K) | `[20, 15, 10, 5, 3, 2, 1]` | Lower K = more complex |
| **Logistic Regression** | `C` (inverse reg. strength) | `[0.001, 0.01, 0.1, 1, 10, 100]` | Higher C = more complex |

---

## Methodology

1. **Load** the Breast Cancer Wisconsin dataset (569 samples, 30 features)
2. **Split** into 70% train / 30% test (stratified)
3. For each complexity value in the model's range:
   - Instantiate the model with that complexity
   - Run `mlxtend.evaluate.bias_variance_decomp` with **10 bootstrap rounds**
   - Record average **bias**, **variance**, and **expected 0-1 loss**
4. **Plot** the bias-variance-loss curves and mark the optimal complexity

---

## Results

Pre-computed results for all 4 models live in `results/` as JSON data files and PNG diagrams.

| Model | Time (seconds) | Loss Curve Behavior |
|---|---|---|
| Decision Tree | 1.1 | U‑shape — bias falls, variance rises, optimum at depth ~3 |
| Random Forest | 35.7 | Bias drops sharply, variance flattens, converges at depth ~5 |
| K-Nearest Neighbors | 1.0 | Shallow U‑shape — best around K=10 |
| Logistic Regression | 0.3 | Steady decrease — higher C (less regularization) improves fit |

### Sample Diagram

Each run produces a Matplotlib figure like this (stored in `results/<model>.png`):

```
Error
  ^
  |  ← Variance (red, dash-dot)
  |    ← Total Loss (black, solid)
  |  ← Bias (blue, dashed)
  |        |
  +--------|--------------------> Complexity
        Optimal
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Opens a browser at `http://localhost:8501`. Select a model from the sidebar and click **Run Analysis**.

### Run All Models (Batch)

```bash
python run_analysis.py
```

Saves JSON results and PNG plots to the `results/` directory.

### Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
Bias_Variance_Predictor/
├── app.py                        # Streamlit web UI
├── requirements.txt              # Python dependencies
├── run_analysis.py               # Batch analysis script (saves results/)
├── README.md
│
├── results/                      # Pre-computed results
│   ├── _all_results.json         # Aggregated JSON
│   ├── decision_tree.json/.png
│   ├── knn.json/.png
│   ├── random_forest.json/.png
│   └── logistic_regression.json/.png
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py         # Abstract base — wraps bias_variance_decomp
│   │   ├── dataset.py            # Breast Cancer Wisconsin loader
│   │   ├── decision_tree.py      # DecisionTreeClassifier
│   │   ├── random_forest.py      # RandomForestClassifier
│   │   ├── knn.py                # KNeighborsClassifier
│   │   └── logistic_regression.py # LogisticRegression
│   └── view/
│       ├── __init__.py
│       └── plots.py              # Matplotlib plotting function
│
├── tests/
│   ├── __init__.py
│   └── test_models.py            # Unit tests for all 4 models
│
└── utils/
    ├── __init__.py
    └── logger.py                 # Logging configuration
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | ML models + dataset + train/test split |
| `mlxtend` | `bias_variance_decomp` implementation |
| `matplotlib` | Plotting |
| `streamlit` | Interactive web UI |
| `numpy` | Numerical operations |
| `pandas` | (utility) |
| `pytest` | Testing |

---

## License

Educational project — free to use and modify.
