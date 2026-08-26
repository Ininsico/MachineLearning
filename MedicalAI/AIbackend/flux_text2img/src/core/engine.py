import requests
from typing import Dict, Any, Optional
from ..config.settings import HF_TOKEN, MODEL_ID

class FluxEngine:
    def __init__(self, token: str = HF_TOKEN, model_id: str = MODEL_ID):
        self.token = token
        self.model_id = model_id
        self.endpoint = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    
    def generate(self, prompt: str, **kwargs) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "height": kwargs.get("height", 1024),
                "width": kwargs.get("width", 1024),
                "num_inference_steps": kwargs.get("steps", 4),
                "guidance_scale": kwargs.get("guidance", 3.5)
            }
        }
        
        response = requests.post(self.endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Generation failed: {response.text}")
