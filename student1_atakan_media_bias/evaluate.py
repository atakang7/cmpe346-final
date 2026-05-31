"""
Evaluation: Accuracy and F1 score.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}
