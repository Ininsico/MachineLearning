import requests
import json

def test_api():
    url = "http://localhost:8000/predict"
    
    payload = {
        "category": "Electronics",
        "unit_price": 450.0,
        "cost_per_unit": 320.0,
        "consumption_volume": 85.0,
        "customer_segment": "retail",
        "is_holiday": 0.0,
        "is_promotion": 1.0
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Response received:")
        print(json.dumps(response.json(), indent=4))
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    test_api()
