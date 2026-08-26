import streamlit as st
from src.models.dataset import BreastCancerDataset
from src.models.decision_tree import DecisionTreeModel
from src.models.knn import KNNModel
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.view.plots import create_bias_variance_plot
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Bias-Variance Analyzer",
    page_icon="📈",
    layout="wide"
)

MODELS = {
    "Decision Tree":       (DecisionTreeModel(),       [1, 2, 3, 5, 8, 12, 16],        "Max Depth (Higher = More Complex)"),
    "Random Forest":       (RandomForestModel(),       [1, 2, 3, 5, 8, 12, 16],        "Max Depth (Higher = More Complex)"),
    "K-Nearest Neighbors": (KNNModel(),                [20, 15, 10, 5, 3, 2, 1],        "K (Lower = More Complex)"),
    "Logistic Regression": (LogisticRegressionModel(), [0.001, 0.01, 0.1, 1, 10, 100], "C (Higher = Less Regularization, More Complex)"),
}

if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("Select Model", list(MODELS.keys()))
    st.markdown("---")

    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Computing bias-variance for {model_name}..."):
            try:
                X_train, X_test, y_train, y_test = BreastCancerDataset.get_data()
                model_cls, complexities, x_title = MODELS[model_name]
                results = model_cls.compute_bias_variance(X_train, y_train, X_test, y_test, complexities)
                st.session_state.fig = create_bias_variance_plot(results, model_cls.name, x_title)
                st.session_state.model_name = model_cls.name
                st.success("Analysis Complete!")
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"Computation error: {e}", exc_info=True)

st.title("Bias-Variance Prediction Analyzer")
st.markdown("Visualizing the bias-variance tradeoff on the **Breast Cancer Wisconsin** dataset.")

if st.session_state.fig is not None:
    st.pyplot(st.session_state.fig)
else:
    st.info("Select a model in the sidebar and click **Run Analysis** to see the bias-variance tradeoff.")
