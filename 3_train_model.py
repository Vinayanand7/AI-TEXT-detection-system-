import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset


# -------------------------
# Device
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# Load Data
# -------------------------

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")


# -------------------------
# Model + Tokenizer
# -------------------------

MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(device)


# -------------------------
# Dataset
# -------------------------

class EssayDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["generated"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_dataset = EssayDataset(train_df)
val_dataset = EssayDataset(val_df)


# -------------------------
# Weighted Loss (CRITICAL PART)
# -------------------------

# Class 0 = Human
# Class 1 = AI
# Penalize false accusation of humans
class_weights = torch.tensor([3.0, 1.0]).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


# -------------------------
# Metrics
# -------------------------

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# -------------------------
# Custom Trainer (Uses Weighted Loss)
# -------------------------

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(device)
        outputs = model(
            input_ids=inputs.get("input_ids").to(device),
            attention_mask=inputs.get("attention_mask").to(device)
        )
        logits = outputs.get("logits")
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# -------------------------
# Training Arguments
# -------------------------

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    save_total_limit=2
)


# -------------------------
# Trainer
# -------------------------

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# -------------------------
# Train
# -------------------------

trainer.train()


# -------------------------
# Save Model
# -------------------------

trainer.save_model("saved_model1")
tokenizer.save_pretrained("saved_model1")

print("Training complete. Model saved.")