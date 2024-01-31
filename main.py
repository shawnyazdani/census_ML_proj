'''
Creating an API that can be used to perform inference on the trained model in production.
'''
import pandas as pd
from typing import Union, List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from utils.train_model import cat_features, encoder, model
from utils.model import inference

class InputDataset(BaseModel):
    """
    Input Inference Dataset, consisting of all census dataset features
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str 
    relationship: str 
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

app = FastAPI()
#perform checks on inputs before hand... ensure all values have the same lengths!

def process_item_dict(item_dict):
    """
    Perform inference on input dataset
    """
    #Determine number of entries
    if isinstance(item_dict['age'], list):
        num_entries = len(item_dict['age'])
    else:
        num_entries = 1
    #Convert data model input to dataframe
    data = pd.DataFrame(item_dict, index=list(range(num_entries)))
    #Perform inference
    pred = inference(model, data, encoder, cat_features)
    return pred

@app.post("/inference/")
async def perform_inference(item: InputDataset):
    item_dict = item.dict(by_alias=True)
    pred = (process_item_dict(item_dict))
    return JSONResponse(content = pred.tolist())


#Root welcome message
@app.get("/")
async def say_hello():
  return {"welcome": "Welcome to the API for the Census dataset!"}

