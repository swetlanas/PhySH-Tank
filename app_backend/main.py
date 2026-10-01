from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import torch
from pylatexenc.latex2text import LatexNodes2Text
import re


from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel


BASE_MODEL = "thellert/physbert_uncased"
MODEL_ID = "swetlanas/physbert-tag-recommender"

@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
                    BASE_MODEL,
                    config=config)
    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model.eval()

    # Objects for the lifespan of the server
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.id2label = model.config.id2label

    yield

    #Memory cleanup on shutdown
    del app.state.model
    del app.state.tokenizer
    del app.state.id2label

#Instantiate FastAPI app
app = FastAPI(title="APS PRB Tag Recommender",lifespan=lifespan)



class PredictRequest(BaseModel):
    title : str
    abstract : str
    top_k : int = 5



class PredictResponse(BaseModel):
    tags: list[str]
    warning: str | None = None

@app.post("/predict",response_model=PredictResponse)
async def predict(payload: PredictRequest,request: Request):
    
    #print(f"received: title={payload.title!r} abstract={payload.abstract!r}")
    latex_converter = LatexNodes2Text()
    cleaned_title = re.sub(r'[_^](\d+)', r'\1', latex_converter.latex_to_text(payload.title))
    cleaned_abstract =  re.sub(r'[_^](\d+)', r'\1', latex_converter.latex_to_text(payload.abstract))
    
    tokenizer = request.app.state.tokenizer
    model = request.app.state.model
    id2label = request.app.state.id2label
    
    if not cleaned_title.strip() and not cleaned_abstract.strip():
        raise HTTPException(status_code=400, detail="Title and Abstract fields are empty.")
           
    inputs = tokenizer(
        cleaned_title,
        cleaned_abstract,
        truncation='only_second',
        return_tensors='pt',
    )

    with torch.no_grad():
        logits = model(**inputs).logits

        topk = torch.topk(logits, k=payload.top_k, dim=-1)
        predicted_tags = [id2label[i.item()] for i in topk.indices[0]]

    if not cleaned_title.strip() or not cleaned_abstract.strip():
        return {"tags" : predicted_tags, "warning" : "Results will be more accurate with both a title and abstract."}
    
    return {"tags": predicted_tags}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model") and app.state.model is not None
    }
