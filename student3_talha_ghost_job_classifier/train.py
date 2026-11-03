"""
Training loop with validation F1 and early stopping.
"""
import copy

import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


torch.backends.cudnn.enabled = False

MIN_IMPROVEMENT = 0.005
PATIENCE = 2


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss if hasattr(output, "loss") else output[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_f1(model, loader, device):
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
    return f1_score(all_labels, all_preds, average="weighted", zero_division=0)


def train(model, train_loader, val_loader, epochs, lr, device, use_scheduler=True):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = None
    if use_scheduler:
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )
    model.to(device)

    best_f1 = 0.0
    best_model_state = None
    no_improve = 0
    loss_history = []

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        f1 = val_f1(model, val_loader, device)
        loss_history.append(round(loss, 4))
        improvement = f1 - best_f1
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f} | Val F1: {f1:.4f} | delta {improvement:+.4f}")

        if improvement >= MIN_IMPROVEMENT:
            best_f1 = f1
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping — no improvement for {PATIENCE} epochs")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, loss_history
