import torch
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# Load test data
test_df = pd.read_csv("AI Generated Essays Dataset.csv")

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")

# Dataset class
class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["generated"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load trained model
model = RobertaForSequenceClassification.from_pretrained("saved_model1")

test_dataset = EssayDataset(test_df)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics
)

print("\nEvaluating on TEST set...")
results = trainer.evaluate(test_dataset)

print("\nTest Results:")
for k, v in results.items():
    print(f"{k}: {v}")


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids

probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
ai_probs = probs[:, 1]
for threshold in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    pred_labels = (ai_probs >= threshold).astype(int)
    
    acc = accuracy_score(labels, pred_labels)
    prec = precision_score(labels, pred_labels)
    rec = recall_score(labels, pred_labels)
    
    cm = confusion_matrix(labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n========== Threshold: {threshold} ==========")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("Confusion Matrix:")
    print(cm)
    print("Humans protected:", tn, "/", tn + fp)
    print("False accusations:", fp)