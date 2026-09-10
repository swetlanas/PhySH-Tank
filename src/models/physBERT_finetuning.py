#src/data/phyBERT_finetuning.py

# Fine-tuning PhysBERT model parameters using LoRA for PhySH classification limited to PRB (Condensed Matter Physics Topics) using the Huggingface Trainer API.

import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#From HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer,TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data"

#CLEANED_DATA is output of prep_data() from preprocessing.py
CLEANED_DATA  = DATA_PATH / "cleaned_data.json"
CLEANED_DATA_SAMPLE  = DATA_PATH / "cleaned_data_sample.json"

BATCH_SIZE = 32
MODEL_NAME = "thellert/physbert_uncased"

EPOCHS=10

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",handlers=[
        logging.FileHandler(ROOT / "logs" / "physbert_classifier.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)




if __name__=='__main__':

    #Load preprocessed dataset
    data_to_load = CLEANED_DATA if CLEANED_DATA.exists() else CLEANED_DATA_SAMPLE
    df = pd.read_json(data_to_load)

    target = 'physh_names'
    y_raw = df[target]

    #Binarizing multi-label tags
    # Reload the binarizer
    mlb = joblib.load(DATA_PATH / "mlb_binarizer.pkl")
    #Transform int64 -> float32 for HuggingFace Trainer API
    y_binarized = mlb.transform(y_raw).astype(np.float32)
    num_classes = len(mlb.classes_)

    #For HuggingFace to look for ground truth under "labels"
    df["labels"] = y_binarized.toarray().tolist()

    #Full HuggingFace dataset
    raw_dataset = Dataset.from_pandas(df[["title", "abstract", "labels"]])

    #Split into train/test splits
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    

    # Load PhysBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_physbert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                                        num_labels=num_classes,
                                                                        problem_type ="multi_label_classification")

    # Setting up LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        modules_to_save = ["classifier"],
    )

    # Applying LoRA configuration to the distilbert model
    model_physbert = get_peft_model(model_physbert, lora_config)
    model_physbert.to(device)

    # Splitting the dataset into train and test sets for single label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create PyTorch Dataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #Prevent test_set shuffling to maintain reproducibility and comprehension

    
    # Data collator for dynamic padding
    collator = DataCollatorWithPadding(tokenizer)

    inputs = tokenizer(df["title"],
                       df["abstract"],
                       use_fast=True,
                       truncation="only_second",
                       max_length=512,
                       return_tensors='pt')
    
    train_dataset = Dataset(inputs)

    model_physbert.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model_physbert.parameters(), lr=1e-3)

    
    logger.info("Starting training loop...")
    for epoch in range(EPOCHS):

        #Training
        model_physbert.train()
        train_loss = 0.0
        logger.info(f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        #Validation
        model_physbert.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(test_loader.dataset)
        train_loss /= len(train_loader.dataset)

        logger.info(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")



    torch.save(model_physbert.state_dict(), DATA_PATH / "models" / "physbert_lora_finetuned.pt")
    