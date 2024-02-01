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

prediction = requests.post("http://127.0.0.1:8000/inference/", data=json.dumps(data)).json()
print(f"Infered Prediction: {prediction}, Note: 1 = Income > $50K, 0 = Income < $50K")
