'''
Test script used to test FastAPI Endpoints
'''
from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

''' Test "/" endpoint, GET method'''
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome": "Welcome to the API for the Census dataset!"}

''' Test "/inference/" endpoint with a single entry, POST method'''
def test_api_locally_post_inference_single_entry():
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
        "capital-gain": 420000,
        "capital-loss": 42,
        "hours-per-week": 42,
        "native-country":"United-States"
        }
    r = client.post("/inference/", content=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == [1] # we know it should yield a 1 (>$50K income) due to our massive capital gain

''' Test "/inference/" endpoint with multiple entries, POST method'''
def test_api_locally_post_inference_multiple_entries():
    data = {"age": [42,50],
            "workclass": ["Private","State-gov"],
            "fnlgt": [12345, 4567],
            "education": ["HS-grad", "Bachelors"], 
            "education-num": [12, 9],
            "marital-status": ["Married-civ-spouse", "Divorced"],
            "occupation": ["Transport-moving","Adm-clerical"],
            "relationship": ["Husband","Not-in-family"],
            "race": ["White", "Black"],
            "sex": ["Male", "Female"],
            "capital-gain": [420000, 3000],
            "capital-loss": [42,0],
            "hours-per-week": [42,40],
            "native-country": ["United-States", "United-States"]
            }
    r = client.post("/inference/", content=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == [1,0] # we know it should yield a [1,0] due to our capital gain discrepancy

