import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score

from dataset import EDOSDataset, load_splits
from models.bert_model import BERTClassifier
from models.roberta_model import RoBERTaClassifier
from models.lstm_model import LSTMClassifier


# Config
DATA_PATH = "../data/edos_labelled_aggregated.csv"
RESULTS_PATH = "../results/results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
MAX_LENGTH = 128
EPOCHS = 3
LEARNING_RATE = 2e-5


# Evaluation
def evaluate(model, dataloader, model_type):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(DEVICE)

            if model_type in ("bert", "roberta"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits

            else:  # lstm
                input_ids = batch["input_ids"].to(DEVICE)
                logits = model(input_ids=input_ids)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


# BERT Training
def train_bert():
    print("\n========== Training BERT ==========")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_df, dev_df, test_df = load_splits(DATA_PATH)

    train_dataset = EDOSDataset(train_df, tokenizer, MAX_LENGTH)
    dev_dataset = EDOSDataset(dev_df, tokenizer, MAX_LENGTH)
    test_dataset = EDOSDataset(test_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = BERTClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dev_acc, dev_f1 = evaluate(model, dev_loader, "bert")
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

    test_acc, test_f1 = evaluate(model, test_loader, "bert")
    print(f"BERT Test → Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    return {"model": "BERT", "accuracy": test_acc, "f1": test_f1}


# RoBERTa Training
def train_roberta():
    print("\n========== Training RoBERTa ==========")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_df, dev_df, test_df = load_splits(DATA_PATH)

    train_dataset = EDOSDataset(train_df, tokenizer, MAX_LENGTH)
    dev_dataset = EDOSDataset(dev_df, tokenizer, MAX_LENGTH)
    test_dataset = EDOSDataset(test_df, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = RoBERTaClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dev_acc, dev_f1 = evaluate(model, dev_loader, "roberta")
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

    test_acc, test_f1 = evaluate(model, test_loader, "roberta")
    print(f"RoBERTa Test → Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    return {"model": "RoBERTa", "accuracy": test_acc, "f1": test_f1}


# LSTM Training
def train_lstm():
    print("\n========== Training LSTM ==========")
    train_df, dev_df, test_df = load_splits(DATA_PATH)

    # Vocabulary oluştur
    from collections import Counter
    import re

    def basic_tokenizer(text):
        return re.findall(r'\b\w+\b', text.lower())

    # Vocab oluştur
    counter = Counter()
    for text in train_df["text"].tolist():
        counter.update(basic_tokenizer(str(text)))

    vocab_list = ["<pad>", "<unk>"] + [word for word, _ in counter.most_common()]
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}
    vocab_size = len(word2idx)

    def text_to_tensor(texts, max_len=MAX_LENGTH):
        tensors = []
        for text in texts:
            tokens = basic_tokenizer(str(text))
            ids = [word2idx.get(token, 1) for token in tokens][:max_len]
            ids += [0] * (max_len - len(ids))  # padding
            tensors.append(ids)
        return torch.tensor(tensors, dtype=torch.long)

    # Dataset ve DataLoader
    class LSTMDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.input_ids = text_to_tensor(df["text"].tolist())
            self.labels = torch.tensor(
                df["label_sexist"].map({"not sexist": 0, "sexist": 1}).tolist(),
                dtype=torch.long
            )

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "label": self.labels[idx]}

    train_dataset = LSTMDataset(train_df)
    dev_dataset = LSTMDataset(dev_df)
    test_dataset = LSTMDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = LSTMClassifier(vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dev_acc, dev_f1 = evaluate(model, dev_loader, "lstm")
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

    test_acc, test_f1 = evaluate(model, test_loader, "lstm")
    print(f"LSTM Test → Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    return {"model": "LSTM", "accuracy": test_acc, "f1": test_f1}


# Main
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    os.makedirs("../results", exist_ok=True)

    results = []
    results.append(train_bert())
    results.append(train_roberta())
    results.append(train_lstm())

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print("\n========== Final Results ==========")
    print(results_df)