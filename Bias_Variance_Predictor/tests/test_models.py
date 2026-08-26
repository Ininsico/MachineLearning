import pytest
import numpy as np
from src.models.decision_tree import DecisionTreeModel
from src.models.knn import KNNModel
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X_train = np.random.rand(40, 10)
    X_test = np.random.rand(10, 10)
    y_train = np.random.randint(0, 2, 40)
    y_test = np.random.randint(0, 2, 10)
    return X_train, y_train, X_test, y_test

def _check_results(results, complexities):
    assert 'loss' in results
    assert 'bias' in results
    assert 'variance' in results
    assert len(results['loss']) == len(complexities)
    assert results['complexities'] == complexities

def test_decision_tree(dummy_data):
    X_train, y_train, X_test, y_test = dummy_data
    model = DecisionTreeModel()
    results = model.compute_bias_variance(X_train, y_train, X_test, y_test, [1, 3])
    _check_results(results, [1, 3])

def test_knn(dummy_data):
    X_train, y_train, X_test, y_test = dummy_data
    model = KNNModel()
    results = model.compute_bias_variance(X_train, y_train, X_test, y_test, [3, 1])
    _check_results(results, [3, 1])

def test_random_forest(dummy_data):
    X_train, y_train, X_test, y_test = dummy_data
    model = RandomForestModel()
    results = model.compute_bias_variance(X_train, y_train, X_test, y_test, [1, 3])
    _check_results(results, [1, 3])

def test_logistic_regression(dummy_data):
    X_train, y_train, X_test, y_test = dummy_data
    model = LogisticRegressionModel()
    results = model.compute_bias_variance(X_train, y_train, X_test, y_test, [0.1, 1])
    _check_results(results, [0.1, 1])
