"""
Main entry point. Trains and evaluates LSTM, BERT, and RoBERTa on BABE dataset.
Run: python main.py — results saved to results/results.json
"""
import torch
import json
import os
from torch.utils.data import DataLoader
from dataset import BABEDataset, get_vocab
from models import LSTMClassifier, get_transformer_model
from train import train
from evaluate import evaluate
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128

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


def run_transformer(name, model_name):
    print(f"\n=== {name} ===")
    train_ds = BABEDataset("train", tokenizer_name=model_name, max_length=MAX_LENGTH)
    test_ds = BABEDataset("test", tokenizer_name=model_name, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    model = get_transformer_model(model_name)
    model = train(model, train_loader, test_loader, epochs=EPOCHS, lr=2e-5, device=DEVICE)
    results = evaluate(model, test_loader, DEVICE)
    print(f"{name} → Accuracy: {results['accuracy']} | F1: {results['f1']}")
    return results


def run_lstm():
    print("\n=== LSTM ===")
    train_ds = BABEDataset("train")
    test_ds = BABEDataset("test")
    vocab = get_vocab(train_ds)
    collate_fn = lambda b: collate_lstm(b, vocab, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    model = LSTMClassifier(vocab_size=len(vocab))
    model = train(model, train_loader, test_loader, epochs=EPOCHS, lr=1e-3, device=DEVICE, use_scheduler=False)
    results = evaluate(model, test_loader, DEVICE)
    print(f"LSTM → Accuracy: {results['accuracy']} | F1: {results['f1']}")
    return results


if __name__ == "__main__":
    results = {}
    results["LSTM"] = run_lstm()
    for name, model_name in MODELS.items():
        results[name] = run_transformer(name, model_name)

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
    }
    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results/results.json")
