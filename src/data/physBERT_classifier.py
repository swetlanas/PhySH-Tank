#src/data/phyBERT_classifier.py

#Train a classification layer on top of physBERT using RTX 3060 12GB. 

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
#from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data"

#CLEANED_DATA is output of prep_data() from preprocessing.py
CLEANED_DATA  = DATA_PATH / "cleaned_data.json"
CLEANED_DATA_SAMPLE  = DATA_PATH / "cleaned_data_sample.json"

BATCH_SIZE = 64
MODEL_NAME = "thellert/physbert_uncased"

EPOCHS=10

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",handlers=[
        logging.FileHandler(ROOT / "logs" / "physbert_classifier.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)


#Class inheritance of PhysBERTClassifierHead from Module in torch.nn
class PhysBERTClassifierHead(nn.Module): 

    def __init__(self, num_classes):
        super().__init__()
        self.fullyconnected = nn.Linear(768,num_classes)

    def forward(self, x_embeddings):
        return self.fullyconnected(x_embeddings)



if __name__=='__main__':

    #Load preprocessed dataset
    data_to_load = CLEANED_DATA if CLEANED_DATA.exists() else CLEANED_DATA_SAMPLE
    df = pd.read_json(data_to_load)

    target = 'physh_names'
    y_raw = df[target]

    #Load pre-generated physBERT embeddings and create a PyTorch tensor
    X = torch.tensor(np.load(DATA_PATH / "physbert_embeddings.npy"),dtype=torch.float32)


    #Binarizing multi-label tags
    mlb = MultiLabelBinarizer(sparse_output=True)
    y_binarized = mlb.fit_transform(y_raw)
    num_classes = len(mlb.classes_)

    y = torch.tensor(y_binarized.toarray(), dtype=torch.float32)

    # Splitting the dataset into train and test sets for single label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create PyTorch Dataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #Prevent test_set shuffling to maintain reproducibility and comprehension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    model = PhysBERTClassifierHead(num_classes=num_classes)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    
    logger.info("Starting training loop...")
    for epoch in range(EPOCHS):

        #Training
        model.train()
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
        model.eval()
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



    torch.save(model.state_dict(), DATA_PATH / "models" / "physbert_classifier_head.pt")
    joblib.dump(mlb, DATA_PATH / "mlb_binarizer.pkl")