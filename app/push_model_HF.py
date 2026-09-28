from huggingface_hub import login
from dotenv import load_dotenv
from peft import PeftModel
import joblib
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()  #Reads HF_TOKEN from .env into environment

ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data"
#PhysBERT 
MODEL_NAME ="thellert/physbert_uncased"

# Reload the binarizer
mlb = joblib.load(DATA_PATH / "mlb_binarizer.pkl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(mlb.classes_), problem_type="multi_label_classification"
)
#Add the index <-> label dictionary to avoid pushing mlb_binarizer.pkl to HF
base_model.config.id2label = {i: label for i, label in enumerate(mlb.classes_)}
base_model.config.label2id = {label: i for i, label in enumerate(mlb.classes_)}

best_model = PeftModel.from_pretrained(base_model, DATA_PATH / "models" / "physbert_lora_finetuned" / "checkpoint-20160")

tokenizer.push_to_hub("swetlanas/physbert-tag-recommender")
best_model.push_to_hub("swetlanas/physbert-tag-recommender")
