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

app_local = 0
if app_local == 1:
        # Local running app versions:
        response_get = requests.get("http://127.0.0.1:8000/")  #Local-running version, get request
        response_post = requests.post("http://127.0.0.1:8000/inference/", data=json.dumps(data)) #Local-running version, post request
else:
        # Web-running app versions:
        response_get = requests.get("https://census-inference-api.onrender.com/") #Web-running version on Render, get request
        response_post = requests.post("https://census-inference-api.onrender.com/inference/", data=json.dumps(data)) #Web-running version on Render, post request

prediction = response_post.json()
print(f"GET Request, Status Code: {response_get.status_code}, Message: {response_get.json()}")
print(f"POST Request, Status Code: {response_post.status_code}, Inferred Prediction: {prediction}, Note: 1 = Income > $50K, 0 = Income < $50K")
