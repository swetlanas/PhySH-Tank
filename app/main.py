from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

MODEL_NAME = "swetlanas/physbert-tag-recommender"

@app.get("/")
def read_root():
    return {"status" : "ok"}


class PredictRequest(BaseModel):
    title : str
    abstract : str

class PredictResponse(BaseModel):
    tags: list[str]


@app.post("/predict",response_model=PredictResponse)
def predict(request : PredictRequest):
        return PredictResponse(tags=["a", "b", "c"])