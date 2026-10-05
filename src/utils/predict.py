import re
import logging
import torch
from peft import PeftModel
from pylatexenc.latex2text import LatexNodes2Text
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


logger = logging.getLogger(__name__)

BASE_MODEL = "thellert/physbert_uncased"
MODEL_ID = "swetlanas/physbert-tag-recommender"


def clean_text(text: str) -> str:
    """Convert LaTeX to unicode and strip digit-only sub/superscript markers."""
    converted = LatexNodes2Text().latex_to_text(text)
    return re.sub(r"[_^](\d+)", r"\1", converted)


def load_model():
    """Load tokenizer and LoRA model from the Hub. Returns (tokenizer, model, id2label)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)
    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model.eval()
    return tokenizer, model, model.config.id2label


def predict_tags(title: str, abstract: str, top_k: int, tokenizer, model, id2label) -> list[str]:
    """Return the top_k predicted tag names for already-cleaned title and abstract."""
    inputs = tokenizer(title, abstract, truncation="only_second", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        topk = torch.topk(logits, k=top_k, dim=-1)
    return [id2label[i.item()] for i in topk.indices[0]]