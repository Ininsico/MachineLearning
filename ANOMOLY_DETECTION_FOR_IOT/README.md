# 🛡️ GAN + BERT Hybrid IoT Anomaly Detection

This project implements a state-of-the-art anomaly detection system for IoT traffic (Botnets and Malware) using a hybrid approach of **Generative Adversarial Networks (GANs)** for data augmentation and **BERT (Large Language Model)** for natural language explanations.

## 🚀 Key Features
- **GAN-Powered Augmentation**: Uses a Conditional GAN to generate synthetic anomaly samples, overcoming class imbalance in IoT datasets.
- **Deep MLP Classifier**: A high-performance neural network with residual connections for high-accuracy traffic classification.
- **BERT Explainability**: Converts raw network flow features into human-readable text and uses a fine-tuned BERT model to explain *why* a specific flow was flagged as an anomaly.
- **Comprehensive Evaluation**: Generates Accuracy, Precision, Recall, F1-Score, Confusion Matrices, ROC-AUC curves, and Precision-Recall curves.
- **Comparison Table**: Automatically compares the performance of a baseline model vs. the GAN-Augmented hybrid model.
- **HTML Report**: A premium, interactive dashboard showing all results, plots, and LLM-based explanations.

## 📁 Project Structure
```text
ANOMOLY_DETECTION_FOR_IOT/
├── data/               # Processed datasets and scalers
├── logs/               # Execution logs
├── results/            
│   ├── models/         # Saved PyTorch & BERT model weights
│   ├── plots/          # Generated visualizations (PNG)
│   └── reports/        # HTML and JSON reports
├── src/                
│   ├── data_processing/ # Cleaning, scaling, and splitting
│   ├── evaluation/      # Metrics and visualization suite
│   ├── explainability/ # BERT-based explanation engine
│   ├── models/         # GAN and Classifier architectures
│   └── training/       # Training loops for GAN and Classifier
├── config.py           # Centralized configuration and hyperparameters
├── main.py             # Main orchestrator script
└── requirements.txt    # Project dependencies
```

## 🛠️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
The `main.py` script handles the entire flow from data loading to report generation.
```bash
python main.py
```

## 📊 Evaluation Metrics
After running, check the `results/reports/explanation_report.html` for a full breakdown including:
- **Accuracy, Precision, Recall, F1**
- **Confusion Matrix**
- **ROC and PR Curves**
- **Model Comparison Table**
- **LLM Explanations** (e.g., *"This flow shows elevated packet rates and unusual flags consistent with SSH Brute-force..."*)

## 🧠 Configuration
You can tune hyperparameters, sample sizes, and model architectures in `config.py`.
- `SAMPLE_SIZE`: Number of rows to use from the dataset.
- `GAN_SYNTHETIC_SAMPLES`: Number of synthetic anomalies to generate.
- `NUM_EXPLAIN_SAMPLES`: Number of test predictions to explain using BERT.
