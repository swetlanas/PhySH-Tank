#src/data/generate_physBERT_embeddings.py

#For generating abstract+title embeddings using physBERT on RTX 3060 12GB. 


from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging




ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data"
DATA = DATA_PATH / "prb_articles_labeled_Jan2016-Jun2026_duplicate-free.json"
DATA_SAMPLE = DATA_PATH / "prb_articles_labeled_Jan2016-Jun2026_duplicate-free_sample.json"

#CLEANED_DATA is output of prep_data() from preprocessing.py
CLEANED_DATA  = DATA_PATH / "cleaned_data.json"

BATCH_SIZE = 64
MODEL_NAME = "thellert/physbert_uncased"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",handlers=[
        logging.FileHandler("physbert_embeddings.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)


def assemble_chunks(chunks_dir):
    """
    Consolidates individual chunks of embeddings to make a single embedding matrix.
    """

    chunk_files = sorted(chunks_dir.glob("chunk_*.npy"))
    logger.info(f"Found {len(chunk_files)} chunk files. Combining...")

    # Load and stack all chunks into a single matrix
    X_transformed_physbert = np.vstack([np.load(f) for f in chunk_files])

    # Save final master array
    np.save(str(DATA_PATH / "physbert_embeddings.npy"), X_transformed_physbert)
    logger.info(f"Saved final embeddings. Shape: {X_transformed_physbert.shape}" )


def generate_embeddings():
    """
    Generates batches of embeddings for a given title+abstract in chunks and finally, combines to output a single matrix of size (n_samples, 768)
    """

    #Load cleaned up dataset
    df = pd.read_json(CLEANED_DATA)


    # Load PhysBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_physbert = AutoModel.from_pretrained(MODEL_NAME)

    chunks_dir = DATA_PATH / "physbert_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_physbert.to(device)
    model_physbert.eval()

    texts = (df['title'].fillna('') + ". " + df['abstract'].fillna('')).tolist()
    temp_batch_embeddings = []
    save_every_n_batches=50

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch_texts = texts[i : i+BATCH_SIZE]

            inputs = tokenizer(batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt').to(device)

            # Model forward pass
            outputs = model_physbert(**inputs)

            # Mean pooling using attention mask
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1)

            # Multiply by attention mask to zero out [PAD] token vectors
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            batch_embedding = (sum_embeddings / sum_mask).cpu().numpy()

            temp_batch_embeddings.append(batch_embedding)

            # Save chunk to disk every save_every_n_batches
            current_batch_num = i // BATCH_SIZE
            if (current_batch_num + 1) % save_every_n_batches == 0 or i + BATCH_SIZE >= len(texts):
                logger.info(f"Currently working on batch number: {current_batch_num }")
                # Stack all the batch arrays in 2D numpy array
                chunk_matrix = np.vstack(temp_batch_embeddings)
                chunk_path = chunks_dir / f"chunk_{current_batch_num:06d}.npy"
                np.save(chunk_path, chunk_matrix)

                # Clear memory list for the next chunk
                temp_batch_embeddings = []

    assemble_chunks(chunks_dir)


if __name__ == "__main__":
    generate_embeddings()