"""
Main entry point. Trains and evaluates LSTM, BERT, and RoBERTa on BABE dataset.
Run: python3 main.py — results and figures saved to results/
Trained models are saved to saved_models/ and reused on subsequent runs.
"""
import torch
import json
import os
import pickle
from torch.utils.data import DataLoader, random_split
from dataset import BABEDataset, get_vocab
from models import LSTMClassifier, get_transformer_model
from train import train
from evaluate import evaluate
from plot import plot_loss_curves, plot_comparison_bar, plot_confusion_matrices
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
VAL_SPLIT = 0.1
SAVED_MODELS_DIR = "saved_models"

MODELS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
}


def collate_lstm(batch, vocab, max_length):
    """Tokenize raw text to token IDs for LSTM."""
    input_ids, labels = [], []
    for item in batch:
        tokens = re.findall(r'\w+', item["text"].lower())[:max_length]
        ids = [vocab.get(t, 1) for t in tokens]
        ids += [0] * (max_length - len(ids))
        input_ids.append(ids)
        labels.append(item["label"].item())
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), max_length, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long),
    }


def split_train_val(dataset, val_ratio=VAL_SPLIT):
    """Split dataset into train and validation sets."""
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size],
                        generator=torch.Generator().manual_seed(42))


def run_transformer(name, model_name, test_loader, train_loader, val_loader):
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
        print(f"Model saved to {save_path}")
    metrics, preds = evaluate(model, test_loader, DEVICE)
    print(f"{name} → Accuracy: {metrics['accuracy']} | F1: {metrics['f1']}")
    return metrics, preds, loss_history


def run_lstm(test_loader, train_loader, val_loader, vocab):
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
        model, loss_history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, device=DEVICE, use_scheduler=False)
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Model saved to {save_path}")
    metrics, preds = evaluate(model, test_loader, DEVICE)
    print(f"LSTM → Accuracy: {metrics['accuracy']} | F1: {metrics['f1']}")
    return metrics, preds, loss_history


if __name__ == "__main__":
    # fix all random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # run from any directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("results", exist_ok=True)

    # split first, then build vocab only from train subset to avoid leakage
    train_ds_raw = BABEDataset("train")
    test_ds_raw = BABEDataset("test")
    lstm_train_sub, lstm_val_sub = split_train_val(train_ds_raw)
    vocab = get_vocab(lstm_train_sub.dataset, indices=lstm_train_sub.indices)
    collate_fn = lambda b: collate_lstm(b, vocab, MAX_LENGTH)
    lstm_train_loader = DataLoader(lstm_train_sub, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    lstm_val_loader = DataLoader(lstm_val_sub, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    lstm_test_loader = DataLoader(test_ds_raw, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    results, all_preds, loss_histories = {}, {}, {}

    metrics, preds, losses = run_lstm(lstm_test_loader, lstm_train_loader, lstm_val_loader, vocab)
    results["LSTM"] = metrics
    all_preds["LSTM"] = preds
    if losses:
        loss_histories["LSTM"] = losses

    for name, model_name in MODELS.items():
        train_ds = BABEDataset("train", tokenizer_name=model_name, max_length=MAX_LENGTH)
        test_ds = BABEDataset("test", tokenizer_name=model_name, max_length=MAX_LENGTH)
        train_sub, val_sub = split_train_val(train_ds)
        tr_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE)
        te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        metrics, preds, losses = run_transformer(name, model_name, te_loader, tr_loader, val_loader)
        results[name] = metrics
        all_preds[name] = preds
        if losses:
            loss_histories[name] = losses

    print("\n=== FINAL RESULTS ===")
    print(f"{'Model':<12} {'Accuracy':>10} {'F1':>10}")
    print("-" * 34)
    for model, metrics in results.items():
        print(f"{model:<12} {metrics['accuracy']:>10} {metrics['f1']:>10}")

    output = {
        "student": "Atakan Gül",
        "student_id": "121200152",
        "topic": "Media Bias Detection in News",
        "dataset": "mediabiasgroup/BABE",
        "results": results,
        "loss_histories": loss_histories,
    }
    with open("results/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results/results.json")

    plot_comparison_bar(results)
    plot_confusion_matrices(all_preds)
    if loss_histories:
        plot_loss_curves(loss_histories)

    print("\nAll figures saved to results/")
