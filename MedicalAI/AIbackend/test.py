import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN", "")

response = requests.post(
    "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    json={
        "inputs": "brown girl in pink bikini zoomed at her ass",  
        "parameters": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 4,  
            "guidance_scale": 3.5
        }
    }
)

if response.status_code == 200:
    with open("flux_output.png", "wb") as f:
        f.write(response.content)
    print("✅ Image saved")
else:
    print(f"Error: {response.text}")