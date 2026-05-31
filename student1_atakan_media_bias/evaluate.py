"""
Evaluation: Accuracy, F1 score, and raw predictions for confusion matrix.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, loader, device):
    """Returns metrics dict and (y_true, y_pred) tuple."""
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
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    metrics = {"accuracy": round(acc, 4), "f1": round(f1, 4)}
    return metrics, (all_labels, all_preds)
