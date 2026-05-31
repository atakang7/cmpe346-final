"""
Training loop for all models.
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train(model, train_loader, val_loader, epochs, lr, device, use_scheduler=True):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = None
    if use_scheduler:
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                                     num_training_steps=total_steps)
    model.to(device)
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f}")
    return model
