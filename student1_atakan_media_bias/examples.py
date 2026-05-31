"""
Pulls interesting prediction examples from the test set:
- Correct by all 3 models
- Wrong by LSTM but correct by transformers
- Disagreement between BERT and RoBERTa
"""
import torch
torch.backends.cudnn.enabled = False
import pickle
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from models import LSTMClassifier, get_transformer_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
LABEL_NAMES = {0: "Not Biased", 1: "Biased"}


def load_lstm(vocab):
    model = LSTMClassifier(vocab_size=len(vocab))
    model.load_state_dict(torch.load("saved_models/LSTM.pt", map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


def load_transformer(name, model_name):
    model = get_transformer_model(model_name)
    model.load_state_dict(torch.load(f"saved_models/{name}.pt", map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


def predict_lstm(model, text, vocab):
    tokens = re.findall(r'\w+', text.lower())[:MAX_LENGTH]
    ids = [vocab.get(t, 1) for t in tokens]
    ids += [0] * (MAX_LENGTH - len(ids))
    input_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    attention_mask = torch.ones(1, MAX_LENGTH, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output if not hasattr(output, "logits") else output.logits
        if isinstance(logits, tuple): logits = logits[1]
    return torch.argmax(logits, dim=1).item()


def predict_transformer(model, tokenizer, text):
    enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output[0]
    return torch.argmax(logits, dim=1).item()


if __name__ == "__main__":
    # load test set
    ds = load_dataset("mediabiasgroup/BABE", split="test")
    texts = ds["text"]
    labels = ds["label"]

    # load models
    with open("saved_models/LSTM_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    lstm = load_lstm(vocab)
    bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    roberta_tok = AutoTokenizer.from_pretrained("roberta-base")
    bert = load_transformer("BERT", "bert-base-uncased")
    roberta = load_transformer("RoBERTa", "roberta-base")

    all_correct, lstm_wrong, bert_roberta_disagree = [], [], []

    for i, (text, label) in enumerate(zip(texts, labels)):
        p_lstm = predict_lstm(lstm, text, vocab)
        p_bert = predict_transformer(bert, bert_tok, text)
        p_rob = predict_transformer(roberta, roberta_tok, text)

        row = {
            "text": text,
            "true": label,
            "lstm": p_lstm,
            "bert": p_bert,
            "roberta": p_rob,
        }

        if p_lstm == label and p_bert == label and p_rob == label:
            all_correct.append(row)
        if p_lstm != label and p_bert == label and p_rob == label:
            lstm_wrong.append(row)
        if p_bert != p_rob:
            bert_roberta_disagree.append(row)

    def show(rows, title, n=3):
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        for r in rows[:n]:
            print(f"\n  Text   : {r['text'][:120]}")
            print(f"  True   : {LABEL_NAMES[r['true']]}")
            print(f"  LSTM   : {LABEL_NAMES[r['lstm']]} {'✓' if r['lstm']==r['true'] else '✗'}")
            print(f"  BERT   : {LABEL_NAMES[r['bert']]} {'✓' if r['bert']==r['true'] else '✗'}")
            print(f"  RoBERTa: {LABEL_NAMES[r['roberta']]} {'✓' if r['roberta']==r['true'] else '✗'}")

    show(all_correct, "ALL CORRECT (confident examples)", n=3)
    show(lstm_wrong, "LSTM WRONG — BERT & RoBERTa CORRECT (transformer advantage)", n=3)
    show(bert_roberta_disagree, "BERT vs RoBERTa DISAGREEMENT", n=3)
