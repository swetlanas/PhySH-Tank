#src/data/phyBERT_finetuning.py

# Fine-tuning PhysBERT model parameters using LoRA for PhySH classification limited to PRB (Condensed Matter Physics Topics) using the Huggingface Trainer API.

import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch


#From HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer,TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import Dataset

ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data"

#CLEANED_DATA is output of prep_data() from preprocessing.py
CLEANED_DATA  = DATA_PATH / "cleaned_data.json"
CLEANED_DATA_SAMPLE  = DATA_PATH / "cleaned_data_sample.json"

BATCH_SIZE = 16
MODEL_NAME = "thellert/physbert_uncased"

EPOCHS=20

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",handlers=[
        logging.FileHandler(ROOT / "logs" / "physbert_finetuning.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred,k = 5):

    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    total_true_tags_per_paper = labels.sum(dim=1).clamp(min=1) #in case it encounters papers with 0 tags (which it shouldn't, since it was dropped in preprocessing)

    _, topk_indices = torch.topk(logits, k=k, dim=1)
    topk_targets = torch.gather(labels, dim=1, index=topk_indices)
    precision_per_sample = topk_targets.sum(dim=1) / float(k)
    recall_per_sample = topk_targets.sum(dim=1) / total_true_tags_per_paper
    position_discount = torch.log2(torch.tensor(range(2,k+2)))
    dcg_per_sample = (topk_targets / position_discount).sum(dim=1)
    ideal_dcg_per_sample =  (torch.tensor([[1 if idx < tags else 0 for idx in range(k)] for tags in total_true_tags_per_paper]) / position_discount).sum(dim=1)
    ndcg_per_sample = dcg_per_sample / ideal_dcg_per_sample
    hits_per_sample = (topk_targets.sum(dim=1) >= 1).float()

    return {"precision_at_5": precision_per_sample.mean().item(),
            "recall_at_5": recall_per_sample.mean().item(), 
            "ndcg_at_5": ndcg_per_sample.mean().item(),
            "hit_at_5": hits_per_sample.mean().item()}


def tokenize_batch(batch):
    """
    For providing batched tokens to huggingface trainer

    Args:
        batch : batch_size of the dataframe to be processed for training
        
    Returns:
        tokenized batch
    """
    return tokenizer(batch["title"],
                       batch["abstract"],
                       truncation="only_second",
                       max_length=512)

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

    #Split into train/test splits for huggingface TrainingArguments
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=0)

    #Prepare PhysBERT for tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_train = split_dataset["train"].map(tokenize_batch, batched=True)
    tokenized_test = split_dataset["test"].map(tokenize_batch, batched=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    
    # Data collator for dynamic padding
    collator = DataCollatorWithPadding(tokenizer)

    # Load PhysBERT for classification task training
    model_physbert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                                        num_labels=num_classes,
                                                                        problem_type ="multi_label_classification")

    # Setting up LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save = ["classifier"],
    )

    # Applying LoRA configuration to the distilbert model
    model_physbert = get_peft_model(model_physbert, lora_config)
    logger.info(f"{model_physbert.print_trainable_parameters()}")
    model_physbert.to(device)


    training_args = TrainingArguments(
        output_dir=str(DATA_PATH / "models" / "physbert_lora_finetuned"),
        logging_dir = str(ROOT / "logs"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        metric_for_best_model="precision_at_5",
        greater_is_better=True,   # tells the trainer precision@5 needs to be maximized
        load_best_model_at_end=True,
    )


    trainer = Trainer(
        model=model_physbert,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    trainer.train()
    
    #torch.save(model_physbert.state_dict(), DATA_PATH / "models" / "physbert_lora_finetuned.pt")
    