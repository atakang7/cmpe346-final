"""
Evaluation metrics for binary classification.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = output.logits if hasattr(output, "logits") else output[1]
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    metrics = {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "f1": round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
    }
    return metrics, (all_labels, all_preds)
