import os
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime

import config
from src.data_processing.preprocessor import load_and_preprocess
from src.models.gan import build_gan
from src.models.classifier import build_classifier
from src.training.train_gan import train_gan
from src.training.train_classifier import train_classifier, predict, predict_proba
from src.explainability.bert_explainer import BertAnomalyExplainer
from src.evaluation.metrics import compute_metrics, build_comparison_table
from src.evaluation.visualizer import (
    plot_gan_losses, plot_training_history, plot_confusion_matrix,
    plot_roc_pr_curves, plot_metric_comparison, plot_per_class_heatmap,
    save_explanation_report, save_metrics_json
)

os.makedirs(config.LOGS_DIR, exist_ok=True)
log_file = os.path.join(config.LOGS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MAIN")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(" Starting GAN + BERT Hybrid Anomaly Detection")
    logger.info(" Device: %s", device)

    data = load_and_preprocess(config)

    logger.info(" Training Baseline Classifier...")
    baseline_model = build_classifier(config, data["n_features"], device)

    _ = train_classifier(
        config, baseline_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        synth_X=np.empty((0, data["n_features"])),
        device=device
    )

    y_pred_base = predict(baseline_model, data["X_test"], device)
    y_prob_base = predict_proba(baseline_model, data["X_test"], device)
    baseline_metrics = compute_metrics(data["y_test"], y_pred_base, y_prob_base, prefix="base_")

    logger.info(" Training GAN for Data Augmentation...")
    G, D = build_gan(config, data["n_features"], device)
    gan_results = train_gan(config, G, D, data["X_train"], data["y_train"], device)

    plot_gan_losses(gan_results["g_losses"], gan_results["d_losses"], config.PLOTS_DIR)

    logger.info(" Training GAN-Augmented Classifier...")
    final_model = build_classifier(config, data["n_features"], device)
    history = train_classifier(
        config, final_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        synth_X=gan_results["synth_X"],
        device=device
    )

    plot_training_history(history, config.PLOTS_DIR)

    logger.info(" Evaluating Final Model...")
    y_pred = predict(final_model, data["X_test"], device)
    y_prob = predict_proba(final_model, data["X_test"], device)

    final_metrics = compute_metrics(data["y_test"], y_pred, y_prob)

    from src.evaluation.metrics import get_confusion_matrix
    cm = get_confusion_matrix(data["y_test"], y_pred)
    plot_confusion_matrix(cm, config.CLASS_NAMES, config.PLOTS_DIR)

    plot_roc_pr_curves(data["y_test"], y_prob, config.PLOTS_DIR)

    plot_per_class_heatmap(final_metrics, config.PLOTS_DIR)

    comparison_results = {
        "Baseline (MLP)": {k.replace("base_", ""): v for k, v in baseline_metrics.items()},
        "GAN-Augmented Hybrid": final_metrics
    }
    comparison_df = build_comparison_table(comparison_results)
    plot_metric_comparison(comparison_df, config.PLOTS_DIR)

    logger.info(" Initializing BERT Explainer...")
    explainer = BertAnomalyExplainer(config, device)

    fine_tune_size = 1000
    X_ft = data["X_train"][:fine_tune_size]
    y_ft = data["y_train"][:fine_tune_size]
    X_v_ft = data["X_val"][:min(200, len(data["X_val"]))]
    y_v_ft = data["y_val"][:min(200, len(data["y_val"]))]

    explainer.fine_tune(X_ft, y_ft, X_v_ft, y_v_ft, data["feature_names"])

    logger.info(" Generating BERT Explanations for Test Samples...")
    explanations = explainer.generate_explanations(
        data["X_test"], y_pred, y_prob, data["att_test"],
        data["feature_names"], n_samples=config.NUM_EXPLAIN_SAMPLES
    )

    logger.info(" Generating Final Reports...")
    report_path = save_explanation_report(explanations, final_metrics, comparison_df, config.REPORTS_DIR)
    save_metrics_json(final_metrics, comparison_df, explanations, config.REPORTS_DIR)

    logger.info(" Pipeline Execution Complete!")
    logger.info(" Report available at: %s", report_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(" Pipeline failed: %s", e, exc_info=True)