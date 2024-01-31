'''
Creating an API that can be used to perform inference on the trained model in production.
'''
from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel, Field

class InputDataset(BaseModel):
    name: str
    tag_size: str = Field(alias='tag-size')
    item_id: int

app = FastAPI()

@app.post("/inference/")
async def perform_inference(item: InputDataset):
    return item

#Root welcome message
@app.get("/")
async def say_hello():
  return {"welcome": "Welcome to the API for the Census dataset!"}

