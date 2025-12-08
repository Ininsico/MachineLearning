from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging

app = Flask(__name__)
model = load_model("brain_Tumor_model_v2.h5")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Read and process image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return jsonify({"error": "Invalid image file"}), 400

        # Preprocess exactly like during training
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img)  # This is ResNet50 specific preprocessing
        
        # Make prediction
        pred = model.predict(np.array([img]))
        result = "Tumor" if np.argmax(pred) == 1 else "Normal"
        confidence = float(np.max(pred))
        
        logger.info(f"Prediction: {result} with confidence {confidence:.2f}")
        return jsonify({
            "result": result,
            "confidence": confidence,
            "details": {
                "class_0_prob": float(pred[0][0]),
                "class_1_prob": float(pred[0][1])
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)