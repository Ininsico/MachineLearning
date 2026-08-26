# Product Consumption AI

This demonstrates a complete machine learning pipeline exported to ONNX and served via a FastAPI application, specifically designed for deployment on Hugging Face Spaces.

## Quickstart (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate data and train models:**
   ```bash
   python train.py
   ```
   This will generate a `models` folder containing the preprocessor, model binaries (`classifier.onnx`, `regressor.onnx`), and feature importance plots.

3. **Run the server:**
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 7860
   ```

4. **Test the API:**
   ```bash
   curl -X POST "http://localhost:7860/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "product_id": "P3812",
       "category": "Groceries",
       "unit_price": 24.50,
       "cost_per_unit": 12.00,
       "daily_volume": 450,
       "is_promotion": "yes",
       "customer_type": "wholesale"
     }'
   ```

## Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Select **Docker** as the Space SDK.
3. Upload all the files in this directory including the generated `models` folder.
4. The Space will automatically build the image using the provided `Dockerfile` and expose it on port 7860.
