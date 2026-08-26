import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

logger = logging.getLogger(__name__)

BG   = "#0F0F1A"
SURF = "#1A1A2E"
TEXT = "#E0E0FF"
PURP = "#6C63FF"
PINK = "#FF6584"
TEAL = "#43AA8B"
GOLD = "#F9C74F"
RED  = "#F94144"
COLS = [PURP, PINK, TEAL, GOLD, RED, "#A8DADC", "#FFB347"]

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURF,
        "axes.edgecolor":    TEXT,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        "#2A2A4A",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  SURF,
        "legend.edgecolor":  PURP,
        "font.family":       "DejaVu Sans",
        "figure.dpi":        150,
    })

def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("    Saved: %s", path)

def plot_gan_losses(g_losses: list, d_losses: list, save_dir: str):
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs  = range(1, len(g_losses) + 1)
    ax.plot(epochs, g_losses, color=PURP, lw=2, label="Generator Loss")
    ax.plot(epochs, d_losses, color=PINK, lw=2, label="Discriminator Loss")
    ax.fill_between(epochs, g_losses, alpha=0.15, color=PURP)
    ax.fill_between(epochs, d_losses, alpha=0.15, color=PINK)
    ax.set_title("GAN Training Loss Curves", fontsize=15, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True)
    _save(fig, os.path.join(save_dir, "gan_loss_curves.png"))

def plot_training_history(history: dict, save_dir: str):
    _apply_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Classifier Training History", fontsize=16, fontweight="bold", color=TEXT, y=1.01)
    metrics_pairs = [
        ("train_loss",  "val_loss",  "Loss",     PURP, PINK),
        ("train_acc",   "val_acc",   "Accuracy", TEAL, GOLD),
        ("train_f1",    "val_f1",    "F1-Score", PURP, RED),
    ]
    for ax, (tr_key, vl_key, title, c1, c2) in zip(axes, metrics_pairs):
        epochs = range(1, len(history[tr_key]) + 1)
        ax.plot(epochs, history[tr_key], color=c1, lw=2, label="Train")
        ax.plot(epochs, history[vl_key], color=c2, lw=2, linestyle="--", label="Validation")
        ax.fill_between(epochs, history[tr_key], history[vl_key], alpha=0.1, color=c1)
        ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT)
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "classifier_training_history.png"))

def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_dir: str, title: str = "Confusion Matrix"):
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", [SURF, PURP, PINK])
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="% of True Class")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, pad=12)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt_color = TEXT if cm_pct[i, j] < 60 else BG
            ax.text(j, i, f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)", ha="center", va="center", fontsize=11, fontweight="bold", color=txt_color)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "confusion_matrix.png"))

def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, save_dir: str):
    _apply_dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC & Precision-Recall Curves", fontsize=15, fontweight="bold", color=TEXT, y=1.01)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc      = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=PURP, lw=2.5, label=f"GAN+BERT Hybrid (AUC = {roc_auc:.4f})")
    ax1.plot([0,1],[0,1], color=TEXT, lw=1, linestyle=":", alpha=0.5, label="Random")
    ax1.fill_between(fpr, tpr, alpha=0.15, color=PURP)
    ax1.set_title("ROC Curve", fontsize=13, fontweight="bold", color=TEXT)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right"); ax1.grid(True)
    ax1.set_xlim([-0.01, 1.01]); ax1.set_ylim([-0.01, 1.05])
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc        = auc(rec, prec)
    ax2.plot(rec, prec, color=TEAL, lw=2.5, label=f"GAN+BERT Hybrid (AUC = {pr_auc:.4f})")
    ax2.fill_between(rec, prec, alpha=0.15, color=TEAL)
    baseline = y_true.mean()
    ax2.axhline(y=baseline, color=GOLD, linestyle=":", lw=1.5, label=f"Baseline (positive rate={baseline:.2f})")
    ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", color=TEXT)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(loc="upper right"); ax2.grid(True)
    ax2.set_xlim([-0.01, 1.01]); ax2.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "roc_pr_curves.png"))

def plot_metric_comparison(comparison_df: pd.DataFrame, save_dir: str):
    _apply_dark_style()
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
    avail = [m for m in metrics_to_plot if m in comparison_df.columns]
    df_plot = comparison_df[avail].astype(float)
    models  = df_plot.index.tolist()
    x       = np.arange(len(avail))
    width   = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, model in enumerate(models):
        vals   = df_plot.loc[model].values
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=model, color=COLS[i % len(COLS)], alpha=0.85, edgecolor=BG, linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(avail, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold", color=TEXT, pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y")
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "metric_comparison.png"))

def plot_per_class_heatmap(metrics: dict, save_dir: str):
    _apply_dark_style()
    data = {
        "Precision": [metrics.get("Benign_precision", 0), metrics.get("Anomaly_precision", 0)],
        "Recall":    [metrics.get("Benign_recall", 0),    metrics.get("Anomaly_recall", 0)],
        "F1-Score":  [metrics.get("Benign_f1", 0),        metrics.get("Anomaly_f1", 0)],
    }
    df = pd.DataFrame(data, index=["Benign", "Anomaly"])
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cm", [SURF, TEAL, GOLD])
    sns.heatmap(df, annot=True, fmt=".4f", cmap=cmap, linewidths=1, linecolor=BG, ax=ax, annot_kws={"fontsize": 13, "fontweight": "bold"}, vmin=0, vmax=1)
    ax.set_title("Per-Class Metrics Heatmap", fontsize=14, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Metric", fontsize=11); ax.set_ylabel("Class", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "per_class_heatmap.png"))

def save_explanation_report(explanations: list, metrics: dict, comparison_df: pd.DataFrame, reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "explanation_report.html")
    acc   = metrics.get("accuracy",  0)
    prec  = metrics.get("precision", 0)
    rec   = metrics.get("recall",    0)
    f1    = metrics.get("f1_score",  0)
    roc   = metrics.get("roc_auc",   0)
    exp_html = ""
    for e in explanations:
        badge_color = "#F94144" if e["prediction"] == "Anomaly" else "#43AA8B"
        exp_lines = e["explanation"].replace("\n", "<br>")
        exp_html += f"""
        <div class="exp-card">
          <div class="exp-header">
            <span class="badge" style="background:{badge_color}">{e["prediction"]}</span>
            <span class="conf">Confidence: {e["confidence"]}</span>
            <span class="attack">Type: {e["attack_type"]}</span>
          </div>
          <div class="exp-body">{exp_lines}</div>
        </div>"""
    comp_rows = ""
    for model, row in comparison_df.iterrows():
        comp_rows += f"<tr><td>{model}</td>"
        for v in row.values:
            comp_rows += f"<td>{v}</td>"
        comp_rows += "</tr>"
    comp_headers = "".join(f"<th>{c}</th>" for c in comparison_df.columns)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GAN+BERT IoT Anomaly Detection  Results Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:    #0F0F1A; --surf: #1A1A2E; --purp: #6C63FF;
    --pink:  #FF6584; --teal: #43AA8B; --gold: #F9C74F;
    --text:  #E0E0FF; --red:  #F94144;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: "Inter", sans-serif; padding: 2rem; }}
  h1 {{ font-size: 2.2rem; font-weight: 900; background: linear-gradient(135deg, var(--purp), var(--pink)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: .4rem; }}
  .subtitle {{ color: #888; margin-bottom: 2.5rem; font-size: .95rem; }}
  h2 {{ font-size: 1.3rem; font-weight: 700; margin: 2rem 0 1rem; color: var(--purp); border-left: 4px solid var(--purp); padding-left: .75rem; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1.2rem; margin-bottom: 2rem; }}
  .metric-card {{ background: var(--surf); border-radius: 14px; padding: 1.4rem 1rem; text-align: center; border: 1px solid #2A2A4A; box-shadow: 0 4px 24px rgba(108,99,255,.15); }}
  .metric-card .val {{ font-size: 2rem; font-weight: 900; }}
  .metric-card .lbl {{ font-size: .8rem; color: #888; margin-top: .3rem; text-transform: uppercase; letter-spacing: 1px; }}
  .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
  .plot-card {{ background: var(--surf); border-radius: 14px; padding: 1rem; border: 1px solid #2A2A4A; text-align: center; }}
  .plot-card img {{ width: 100%; border-radius: 8px; }}
  .plot-card p {{ margin-top: .6rem; font-size: .85rem; color: #888; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surf); border-radius: 12px; overflow: hidden; margin-bottom: 2rem; }}
  th {{ background: var(--purp); color: #fff; padding: .85rem 1rem; font-size: .9rem; text-align: left; }}
  td {{ padding: .75rem 1rem; font-size: .9rem; border-bottom: 1px solid #2A2A4A; }}
  tr:hover {{ background: #22224A; }}
  .exp-card {{ background: var(--surf); border-radius: 14px; padding: 1.4rem; margin-bottom: 1.2rem; border: 1px solid #2A2A4A; border-left: 4px solid var(--purp); }}
  .exp-header {{ display: flex; align-items: center; gap: 1rem; margin-bottom: .8rem; flex-wrap: wrap; }}
  .badge {{ padding: .3rem .9rem; border-radius: 20px; font-size: .8rem; font-weight: 700; color: #fff; }}
  .conf {{ font-size: .85rem; color: var(--gold); font-weight: 600; }}
  .attack {{ font-size: .85rem; color: var(--teal); font-weight: 600; }}
  .exp-body {{ font-size: .88rem; line-height: 1.8; color: #BCC0D6; font-family: monospace; background: #0A0A18; border-radius: 8px; padding: 1rem; white-space: pre-wrap; }}
  footer {{ text-align: center; margin-top: 3rem; color: #444; font-size: .8rem; }}
</style>
</head>
<body>
<h1> GAN + BERT Hybrid  IoT Anomaly Detection</h1>
<p class="subtitle">CIC-IDS-2018 Dataset &nbsp;|&nbsp; Botnet &amp; Malware Traffic Analysis &nbsp;|&nbsp; BERT-Powered Explainability</p>
<h2> Overall Performance Metrics</h2>
<div class="metrics-grid">
  <div class="metric-card"><div class="val" style="color:var(--purp)">{acc:.4f}</div><div class="lbl">Accuracy</div></div>
  <div class="metric-card"><div class="val" style="color:var(--teal)">{prec:.4f}</div><div class="lbl">Precision</div></div>
  <div class="metric-card"><div class="val" style="color:var(--gold)">{rec:.4f}</div><div class="lbl">Recall</div></div>
  <div class="metric-card"><div class="val" style="color:var(--pink)">{f1:.4f}</div><div class="lbl">F1-Score</div></div>
  <div class="metric-card"><div class="val" style="color:var(--red)">{roc:.4f}</div><div class="lbl">ROC-AUC</div></div>
</div>
<h2> Training & Performance Plots</h2>
<div class="plot-grid">
  <div class="plot-card"><img src="../plots/gan_loss_curves.png" alt="GAN Losses"><p>GAN Generator vs Discriminator Loss</p></div>
  <div class="plot-card"><img src="../plots/classifier_training_history.png" alt="Training History"><p>Classifier Training History (Loss / Accuracy / F1)</p></div>
  <div class="plot-card"><img src="../plots/confusion_matrix.png" alt="Confusion Matrix"><p>Confusion Matrix</p></div>
  <div class="plot-card"><img src="../plots/roc_pr_curves.png" alt="ROC PR"><p>ROC &amp; Precision-Recall Curves</p></div>
  <div class="plot-card"><img src="../plots/metric_comparison.png" alt="Comparison"><p>Model Comparison (Baseline vs GAN-Augmented vs GAN+BERT)</p></div>
  <div class="plot-card"><img src="../plots/per_class_heatmap.png" alt="Per-Class"><p>Per-Class Metrics Heatmap</p></div>
</div>
<h2> Model Comparison Table</h2>
<table>
  <thead><tr><th>Model</th>{comp_headers}</tr></thead>
  <tbody>{comp_rows}</tbody>
</table>
<h2> BERT-Based Explanations (Sample Predictions)</h2>
{exp_html}
<footer>Generated by GAN+BERT IoT Anomaly Detection Pipeline &nbsp;&bull;&nbsp; CIC-IDS-2018</footer>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(" HTML Report saved: %s", path)
    return path

def save_metrics_json(metrics: dict, comparison_df: pd.DataFrame, explanations: list, reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    payload = {
        "final_metrics": {k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in metrics.items()},
        "model_comparison": comparison_df.reset_index().to_dict(orient="records"),
        "bert_explanations": explanations,
    }
    path = os.path.join(reports_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(" JSON Results saved: %s", path)