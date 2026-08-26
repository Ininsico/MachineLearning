from fastapi import FastAPI
from .routes import generate_router

app = FastAPI(
    title="FLUX Text-to-Image API",
    description="Production-ready text-to-image generation API",
    version="1.0.0"
)

app.include_router(generate_router)

@app.get("/")
async def root():
    return {
        "name": "FLUX Text-to-Image API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
