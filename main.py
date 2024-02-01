'''
Creating an API that can be used to perform inference on the trained model in production.
'''
import pandas as pd
from typing import Union, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from utils.train_model import cat_features, encoder, model
from utils.model import inference

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

app = FastAPI(title="Census Inference API",
    description="An API that demonstrates inference using an input census dataset.",
    version="1.0.0",)

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

def process_item_dict(item_dict, num_entries):
    """
    Perform inference on input dataset
    Input
    ---
    Input Dataset dictionary, with keys and values corresponding to those in the census dataset
    
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
async def perform_inference(item: InputDataset):
    """
    Perform inference using an input raw dataset block.
    Input: census dataset, in model block format.
    Returns: inferred income range classifications for each sample entry in provided dataset
    """
    item_dict = item.dict(by_alias=True)
    num_entries = verify_proper_size(item_dict)
    pred = process_item_dict(item_dict, num_entries)
    return JSONResponse(content = pred.tolist())


#Root welcome message
@app.get("/")
async def say_hello():
  """
  Output a welcome message at the root url
  """
  return {"welcome": "Welcome to the API for the Census dataset!"}



#Running locally:
#Command: uvicorn main:app --reload
#Running with Cloud Application Platform:
# Set start command to:  uvicorn main:app --host 0.0.0.0 --port 10000