import requests
import json

data = {"age": [42],
        "workclass": ["Private"],
        "fnlgt": [12345],
        "education": "HS-grad", 
        "education-num": 12,
        "marital-status": "Married-civ-spouse",
        "occupation": "Transport-moving",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 42000,
        "capital-loss": 42,
        "hours-per-week": 42,
        "native-country":"United-States"
        }

# response = requests.post("http://127.0.0.1:8000/inference/", data=json.dumps(data)) #Local-running version
response = requests.post("https://census-inference-api.onrender.com/inference/", data=json.dumps(data)) #Web-running version, on Render
prediction = response.json()
print(f"Status Code: {response.status_code}, Infered Prediction: {prediction}, Note: 1 = Income > $50K, 0 = Income < $50K")
