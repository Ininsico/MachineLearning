FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY app/ ./app/

EXPOSE 7860

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
