from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Product Consumption AI API")

# Load preprocessor and models
try:
    preprocessor = joblib.load('preprocessor.joblib')
    
    # Try to load single merged model
    try:
        session = ort.InferenceSession("model.onnx")
        clf_session = session
        reg_session = session
        is_merged = True
        logger.info("Merged model and preprocessor loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load merged model.onnx: {e}. Trying separate models...")
        clf_session = ort.InferenceSession("classification_model.onnx")
        reg_session = ort.InferenceSession("regression_model.onnx")
        is_merged = False
        logger.info("Separate models and preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or preprocessor: {e}")
    # We allow the app to start but it will fail on requests until models are generated
    preprocessor = None
    clf_session = None
    reg_session = None

class ProductInput(BaseModel):
    category: str
    unit_price: float
    cost_per_unit: float
    consumption_volume: float
    customer_segment: str
    is_holiday: float = 0.0
    is_promotion: float = 0.0

# Pattern labels mapping
PATTERN_LABELS = {
    0: "Low Velocity / Stable",
    1: "Low Velocity / Seasonal",
    2: "Low Velocity / Erratic",
    3: "Medium Velocity / Stable",
    4: "Medium Velocity / Seasonal",
    5: "Medium Velocity / Erratic",
    6: "High Velocity / Stable",
    7: "High Velocity / Seasonal",
    8: "High Velocity / Erratic"
}

@app.post("/predict")
async def predict(data: ProductInput):
    start_time = time.time()
    
    if preprocessor is None or clf_session is None or reg_session is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Run train.py first.")
    
    try:
        # Convert input to DataFrame for preprocessor
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Preprocess
        X_transformed = preprocessor.transform(df).astype(np.float32)
        
        if is_merged:
            inputs = {session.get_inputs()[0].name: X_transformed}
            outs = session.run(None, inputs)
            # Find outputs by name (we named them 'label', 'probabilities', 'prediction' in train.py)
            out_names = [o.name for o in session.get_outputs()]
            
            # Label
            label_idx = out_names.index('label') if 'label' in out_names else 0
            pattern_id = int(outs[label_idx][0])
            
            # Prediction
            pred_idx = out_names.index('prediction') if 'prediction' in out_names else 2
            # If not found by name, default to the 3rd output if length is 3
            if len(outs) == 3 and 'prediction' not in out_names:
                pred_idx = 2
            profit_slope = float(outs[pred_idx][0][0]) if len(np.shape(outs[pred_idx])) > 1 else float(outs[pred_idx][0])
        else:
            # Classification Inference
            clf_inputs = {clf_session.get_inputs()[0].name: X_transformed}
            clf_out = clf_session.run(None, clf_inputs)
            pattern_id = int(clf_out[0][0])
            
            # Regression Inference
            reg_inputs = {reg_session.get_inputs()[0].name: X_transformed}
            reg_out = reg_session.run(None, reg_inputs)
            profit_slope = float(reg_out[0][0][0]) if len(np.shape(reg_out[0])) > 1 else float(reg_out[0][0])
        
        pattern_type = PATTERN_LABELS.get(pattern_id, "Unknown")
        
        # Mocking Confidence Interval (e.g., 5% margin)
        conf_interval = [profit_slope * 0.95, profit_slope * 1.05]
        
        duration_ms = (time.time() - start_time) * 1000
        
        response = {
            "classification": {
                "pattern_id": pattern_id,
                "pattern_type": pattern_type
            },
            "prediction": {
                "next_30_day_profit_slope": round(profit_slope, 4),
                "confidence_interval": [round(c, 4) for c in conf_interval]
            },
            "metadata": {
                "latency_ms": round(duration_ms, 2)
            }
        }
        
        logger.info(f"Prediction made for category {data.category} in {duration_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
