'''
Creating an API that can be used to perform inference on the trained model in production.
'''
import pandas as pd
import uvicorn
from typing import Union, List
from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from utils.model import inference, load_fitted_data, get_feature_names

# cat_features, _ = get_feature_names() #used for one-hot-encoding
#Load in fitted model and one-hot-encoder.
# model, encoder, _ = load_fitted_data()

class InputDataset(BaseModel):
    """
    Input Inference Dataset, consisting of all census dataset features
    """
    age: Union[int, List[int]]
    workclass: Union[str, List[str]]
    fnlgt: Union[int, List[int]]
    education: Union[str, List[str]]
    education_num: Union[int, List[int]] = Field(alias='education-num')
    marital_status: Union[str, List[str]] = Field(alias='marital-status')
    occupation: Union[str, List[str]]
    relationship: Union[str, List[str]]
    race: Union[str, List[str]]
    sex: Union[str, List[str]]
    capital_gain: Union[int, List[int]] = Field(alias='capital-gain')
    capital_loss: Union[int, List[int]] = Field(alias='capital-loss')
    hours_per_week: Union[int, List[int]] = Field(alias='hours-per-week')
    native_country: Union[str, List[str]] = Field(alias='native-country')

    class Config:
        '''used to set an example for all the feature values in the docs.'''
        schema_extra = {
            "examples": [
                {
                "age": 42,
                "workclass": "Private",
                "fnlgt": 12345,
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
            ]
        }

@asynccontextmanager
async def lifespan(app: FastAPI): 
    #Define app start-up and shut-down sequenceL allows for ensuring this doesnt get rerun/loaded for each request made. 
    #Only executed once before app startup, and used for all requests.
    #------
    # Start-up Sequence: Load the ML model and other data
    model, encoder, _ = load_fitted_data()
    cat_features, _ = get_feature_names() #used for one-hot-encoding
    yield {"model": model, "encoder": encoder, "cat_features": cat_features}
    #Shut-down Sequence (none here right now)

app = FastAPI(lifespan=lifespan, title="Census Inference API",
    description="An API that demonstrates inference using an input census dataset.",
    version="1.0.0")

def verify_proper_size(item_dict):
    """
    Verified that the item dictionary has the same number of entries provided for each feature.
    Ensures that a proper inference can be performed.
    Input
    ---
    Input Dataset dictionary, with keys and values corresponding to those in the census dataset
    
    Returns
    ---
    Number of entries for each feature in dataset if input has the proper size.
    Otherwise, an HTTP Exception is raised if the input is of an improper size.

    """
    #Number of entries per feature
    num_entries_features = [len(value) if isinstance(value, list) else 1 for value in item_dict.values()]
    #Ensure number of entries provided per feature is the same for all features
    if len(set(num_entries_features)) != 1:
        raise HTTPException(
            status_code=400,
            detail=f"All features must have the same number of entries.",
        )
    return set(num_entries_features).pop() 

def process_item_dict(item_dict, num_entries, model, encoder, cat_features):
    """
    Perform inference on input dataset
    Input
    ---
    Input Dataset dictionary, with keys and values corresponding to those in the census dataset
    Model, Encoder: Trained model and fitted one-hot-encoder
    Cat_features: Categorical Features, used for one-hot-encoding transform on provided data.
    
    Returns
    ---
    Income range inference predictions, for each sample/entry in the dataset.

    """
    #Convert data model input to dataframe
    data = pd.DataFrame(item_dict, index=list(range(num_entries)))
    #Perform inference
    pred = inference(model, data, encoder, cat_features)
    return pred

@app.post("/inference/")
async def perform_inference(item: InputDataset, request:Request):
    """
    Perform inference using an input raw dataset block.
    Input: census dataset, in model block format.
    Returns: inferred income range classifications for each sample entry in provided dataset
    """
    item_dict = item.dict(by_alias=True)
    num_entries = verify_proper_size(item_dict)
    pred = process_item_dict(item_dict, num_entries, request.state.model, request.state.encoder, request.state.cat_features)
    return JSONResponse(content = pred.tolist())


#Root welcome message
@app.get("/")
async def say_hello():
    """
    Output a welcome message at the root url
    """
    return {"welcome": "Welcome to the API for the Census dataset!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) #can be used when run script directly w/ 'python main.py', as opposed to CLI commands below.

#Running locally with CLI:
#Command: uvicorn main:app --reload
#Running with Cloud Application Platform:
# Set start command to:  uvicorn main:app --host 0.0.0.0 --port 10000