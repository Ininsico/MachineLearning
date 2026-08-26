# MachineLearning

A collection of standalone machine-learning projects. Each subfolder is an
independent project with its own code, README, and `requirements.txt` where
applicable.

## Projects

| Project | Description | Status |
| --- | --- | --- |
| `AirPollutionDataTraining` | Linear-regression AQI predictor (scikit-learn) | Keep / minor fixes |
| `ANOMOLY_DETECTION_FOR_IOT` | IoT anomaly detection (GAN + MLP) with evaluation | Keep |
| `Bias_Variance_Predictor` | Streamlit bias/variance decomposition demo | Keep |
| `ECommereceFraudDetection` | E-commerce fraud detection (LR/RF/SVM) + EDA | Keep (reference project) |
| `Language` | Experimental custom LLM / transformer research | Keep / document |
| `Libraries` | Library learning snippets (numpy/pandas/sklearn/...) | Keep / study folder |
| `LogClassificationSystem` | Hybrid log classifier (regex/BERT/LLM) + API | Keep |

## Removed during cleanup

`AntiMatter`, `BrainTumorDetection`, `GPT-OSS80B`, `ResumeAnalyzerAi`,
`Velocity`, `XLR8`, `VoiceCloner`, `NanoGpt`, `ImageProcessor`,
`OBJECTDETECTION`, `ProductConusmptionAI`, `eman`, `Extension`,
`clothdetection`, `facedetection`, `SimpleImageRecognitionNeuralNetwork`,
`MedicalAI`, `Edge`. History was reset to a single clean commit — all garbage
projects, committed secrets, and PHI were removed in the process.

## Notes

- Secrets (MongoDB URI, HuggingFace / Kaggle tokens) are loaded from
  environment variables — never hardcode credentials. If you see a leaked
  token anywhere, rotate it immediately.
- Large artifacts, models, datasets, caches, and `.env` files are gitignored.
