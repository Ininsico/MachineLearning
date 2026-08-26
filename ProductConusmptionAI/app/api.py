import time
import pandas as pd
import numpy as np
import onnxruntime as ort
import joblib
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import contextlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

preprocessor = None
clf_sess = None
reg_sess = None
pattern_map = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, clf_sess, reg_sess, pattern_map
    try:
        preprocessor = joblib.load('models/preprocessor.joblib')
        pattern_map = joblib.load('models/pattern_map.joblib')
        clf_sess = ort.InferenceSession('models/classifier.onnx')
        reg_sess = ort.InferenceSession('models/regressor.onnx')
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(str(e))
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ProductInput(BaseModel):
    product_id: str = "P1000"
    category: str = "Electronics"
    unit_price: float = 10.0
    cost_per_unit: float = 5.0
    daily_volume: int = 100
    is_promotion: str = "no"
    customer_type: str = "retail"

@app.get("/")
def root():
    return {"status": "ok", "models_loaded": reg_sess is not None}

@app.post("/predict")
def predict(data: ProductInput):
    start_t = time.time()
    if not reg_sess or not clf_sess or not preprocessor:
        raise HTTPException(status_code=500, detail="Models missing")
        
    try:
        df = pd.DataFrame([data.dict()])
        X = preprocessor.transform(df).astype(np.float32)
        
        c_in = {clf_sess.get_inputs()[0].name: X}
        c_out = clf_sess.run(None, c_in)
        p_idx = int(c_out[0][0])
        prob = float(c_out[1][0][p_idx]) if len(c_out) > 1 else 0.0
        
        r_in = {reg_sess.get_inputs()[0].name: X}
        r_out = reg_sess.run(None, r_in)
        profit = float(r_out[0][0][0]) if len(np.shape(r_out[0])) > 1 else float(r_out[0][0])
        
        dur = (time.time() - start_t) * 1000
        return {
            "consumption_pattern": pattern_map.get(p_idx, "Unknown"),
            "predicted_profit": round(profit, 2),
            "confidence": round(prob, 2),
            "latency_ms": round(dur, 2)
        }
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
