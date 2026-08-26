# 📉 Log Classification System (LCS)

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-orange)
![Author](https://img.shields.io/badge/Author-Arslan%20Rathore-purple)

## 📖 Overview

Welcome to the **Log Classification System**! This project implements a robust **Hybrid Classification Framework** designed to categorize system logs with high precision. By leveraging a combination of rule-based and machine learning strategies, this system effectively handles varying levels of data complexity—from simple, predictable patterns to unstructured, noisy log messages.

The core philosophy of this project is **adaptability**. It seamlessly switches between simplified regex matching, semantic understanding with Sentence Transformers, and advanced reasoning via Large Language Models (LLMs).

---

## 🚀 Key Features

### 🧠 Hybrid Classification Engine
The system employs a tiered approach to ensure optimal performance and accuracy:

1.  **Regular Expressions (Regex)** ⚡
    *   **Best for:** Known, repetitive, and highly structured log patterns.
    *   **Benefit:** Ultra-fast execution with near-zero latency.

2.  **Sentence Transformer + Logistic Regression** 🤖
    *   **Best for:** Complex patterns with sufficient labeled training data.
    *   **Benefit:** Captures semantic meaning using embeddings (BERT-based) and classifies using a lightweight statistical model.

3.  **Large Language Models (LLM)** 🧠
    *   **Best for:** Ambiguous, novel, or "few-shot" scenarios where training data is scarce.
    *   **Benefit:** Utilizes deep contextual understanding to categorize difficult logs that other methods miss.

---

## 🏗️ Architecture

The system pipeline is designed for scalability and fault tolerance. Logs are processed sequentially through the tiers, ensuring that the most efficient method handles the request first.

![System Architecture](resources/arch.png)

---

## 📂 Project Structure

A clean and organized codebase for easy navigation and scalability.

```
📁 project-nlp-log-classification
├── 📁 models/                  # 🧠 Saved models (Embeddings & Classifiers)
├── 📁 resources/               # 📊 Test data, images, and output artifacts
├── 📁 training/                # 🚂 Scripts for model training & regex definition
│   └── train.py                #    (Example training script)
├── 📄 classify.py              # 🏷️ Core classification logic
├── 📄 processor_bert.py        # 🤖 BERT-based embedding processor
├── 📄 processor_llm.py         # 🧠 LLM interaction handler
├── 📄 processor_regex.py       # ⚡ Regex pattern matcher
├── 📄 server.py                # 🌐 FastAPI application entry point
├── 📄 requirements.txt         # 📦 Project dependencies
└── 📄 README.md                # 📖 Documentation
```

---

## 🛠️ Installation & Setup

Follow these steps to get the system running locally.

### Prerequisites
*   Python 3.8 or higher
*   pip (Python Package Installer)

### 1. Clone the Repository
```bash
git clone <repository_url>
cd project-nlp-log-classification
```

### 2. Install Dependencies
Install all required packages using the provided requirements file:
```bash
pip install -r requirements.txt
```

### 3. Launch the Server
Start the FastAPI backend with hot-reloading enabled:
```bash
uvicorn server:app --reload
```

The server will be live at:
*   **API Root:** `http://127.0.0.1:8000/`
*   **Docs (Swagger UI):** `http://127.0.0.1:8000/docs`
*   **Redoc:** `http://127.0.0.1:8000/redoc`

---

## 💻 Usage

### Classifying Logs
The primary endpoint accepts a CSV file upload. The input CSV must contain at least the following columns:

| Column Name   | Description                                      |
| :------------ | :----------------------------------------------- |
| `source`      | The source of the log (e.g., specific service).  |
| `log_message` | The actual text content of the log entry.        |

**Example Response:**
The system returns a downloadable CSV file with an appended `target_label` column containing the predicted class for each log.

---

## 👨‍💻 Author

**Created with ❤️ by [Arslan Rathore](https://github.com/Ininsico)**

A passionate developer dedicated to building intelligent systems and solving complex problems with code.

---

## 📄 License and Disclaimer

This project is for **educational purposes**.

*   **Copyrights Reserved**: @ininsico
*   **Usage**: Not for commercial use without explicit authorization.
