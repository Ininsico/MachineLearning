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
| `clothdetection` | YOLOv8 clothing detection + outfit recommender | Keep |
| `ECommereceFraudDetection` | E-commerce fraud detection (LR/RF/SVM) + EDA | Keep (reference project) |
| `Edge` | Edge TTS / grapheme-to-phoneme pipeline (LJSpeech) | Keep / fix paths |
| `eman` | From-scratch GPT trained on a WhatsApp chat export | Keep |
| `Extension` | VS Code AI assistant extension (Ollama) | Keep / fix build |
| `facedetection` | Face recognition (InsightFace + HOG/SVM) | Keep / fix |
| `ImageProcessor` | Packaged face-recognition pipeline (InsightFace) | Keep |
| `Language` | Experimental custom LLM / transformer research | Keep / document |
| `Libraries` | Library learning snippets (numpy/pandas/sklearn/...) | Keep / study folder |
| `LogClassificationSystem` | Hybrid log classifier (regex/BERT/LLM) + API | Keep |
| `MedicalAI` | Medical-diagnostics agents + FLUX text2img backend | Keep (partial) |
| `NanoGpt` | Character-level Shakespeare GPT (nanoGPT-style) | Keep |
| `OBJECTDETECTION` | Crack detection + machine-failure detector | Keep (partial) |
| `ProductConusmptionAI` | Product-consumption predictor + ONNX/FastAPI | Keep |
| `ResumeAnalyzerAi` | _Removed_ — non-functional stub | Deleted |
| `SimpleImageRecognitionNeuralNetwork` | From-scratch face generator (Java) | Keep |
| `Velocity` | _Removed_ — non-functional pseudo-ML | Deleted |
| `VoiceCloner` | Vendored Tortoise-TTS voice cloning | Keep |
| `XLR8` | Vendored Qwen3-TTS text-to-speech | Keep / attribute |

## Removed during cleanup

`AntiMatter` (fabricated artifacts), `BrainTumorDetection` (non-functional,
leaked credentials), `GPT-OSS80B` (empty), `ResumeAnalyzerAi` (stub),
`Velocity` (non-functional). History was rewritten with `git filter-repo`
to purge these and any committed secrets / PHI.

## Notes

- Secrets (MongoDB URI, HuggingFace / Kaggle tokens) are loaded from
  environment variables — never hardcode credentials.
- Large artifacts, models, datasets, caches, and `.env` files are gitignored.
