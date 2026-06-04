"""
Main entry point for Talha Demir's CMPE346 final project component.

Task: Ghost-job risk classification from job posting text.
Models: LSTM, BERT, RoBERTa.

Run:
    python3 main.py

Local datasets and saved model weights are generated under data/ and
saved_models/; both are ignored by git.
"""
import json
import os
import pickle
import re

import torch
from torch.utils.data import DataLoader

from dataset import GhostJobDataset, get_vocab
from evaluate import evaluate
from models import LSTMClassifier, get_transformer_model
from plot import plot_comparison_bar, plot_confusion_matrices, plot_loss_curves
from train import train


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 256
SAVED_MODELS_DIR = "saved_models"

MODELS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
}


def collate_lstm(batch, vocab, max_length):
    input_ids, labels = [], []
    for item in batch:
        tokens = re.findall(r"\w+", item["text"].lower())[:max_length]
        ids = [vocab.get(token, 1) for token in tokens]
        ids += [0] * (max_length - len(ids))
        input_ids.append(ids)
        labels.append(item["label"].item())
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), max_length, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long),
    }


def run_transformer(name, model_name, train_loader, val_loader, test_loader):
    print(f"\n=== {name} ===")
    save_path = os.path.join(SAVED_MODELS_DIR, f"{name}.pt")
    loss_history = []
    model = get_transformer_model(model_name)
    if os.path.exists(save_path):
        print(f"Loading saved model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
    else:
        model, loss_history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=2e-5, device=DEVICE)
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    metrics, preds = evaluate(model, test_loader, DEVICE)
    print(f"{name} -> Accuracy: {metrics['accuracy']} | Weighted F1: {metrics['f1']}")
    return metrics, preds, loss_history


def run_lstm(train_loader, val_loader, test_loader, vocab):
    print("\n=== LSTM ===")
    save_path = os.path.join(SAVED_MODELS_DIR, "LSTM.pt")
    vocab_path = os.path.join(SAVED_MODELS_DIR, "LSTM_vocab.pkl")
    loss_history = []
    model = LSTMClassifier(vocab_size=len(vocab))
    if os.path.exists(save_path):
        print(f"Loading saved model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
    else:
        model, loss_history = train(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=1e-3,
            device=DEVICE,
            use_scheduler=False,
        )
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        with open(vocab_path, "wb") as file:
            pickle.dump(vocab, file)
    metrics, preds = evaluate(model, test_loader, DEVICE)
    print(f"LSTM -> Accuracy: {metrics['accuracy']} | Weighted F1: {metrics['f1']}")
    return metrics, preds, loss_history


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("results", exist_ok=True)

    train_raw = GhostJobDataset("train")
    val_raw = GhostJobDataset("validation")
    test_raw = GhostJobDataset("test")
    vocab = get_vocab(train_raw)
    collate_fn = lambda batch: collate_lstm(batch, vocab, MAX_LENGTH)

    lstm_train_loader = DataLoader(train_raw, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    lstm_val_loader = DataLoader(val_raw, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    lstm_test_loader = DataLoader(test_raw, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    results, all_preds, loss_histories = {}, {}, {}

    metrics, preds, losses = run_lstm(lstm_train_loader, lstm_val_loader, lstm_test_loader, vocab)
    results["LSTM"] = metrics
    all_preds["LSTM"] = preds
    if losses:
        loss_histories["LSTM"] = losses

    for name, model_name in MODELS.items():
        train_ds = GhostJobDataset("train", tokenizer_name=model_name, max_length=MAX_LENGTH)
        val_ds = GhostJobDataset("validation", tokenizer_name=model_name, max_length=MAX_LENGTH)
        test_ds = GhostJobDataset("test", tokenizer_name=model_name, max_length=MAX_LENGTH)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        metrics, preds, losses = run_transformer(name, model_name, train_loader, val_loader, test_loader)
        results[name] = metrics
        all_preds[name] = preds
        if losses:
            loss_histories[name] = losses

    output = {
        "student": "Talha Demir",
        "student_id": "122200106",
        "topic": "Ghost-Job Risk Classification",
        "dataset": "Zenodo record 20321172 unified_core.parquet",
        "results": results,
        "loss_histories": loss_histories,
    }
    with open("results/results.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    plot_comparison_bar(results)
    plot_confusion_matrices(all_preds)
    if loss_histories:
        plot_loss_curves(loss_histories)
    print("\nResults saved to results/results.json and figures saved to results/")
