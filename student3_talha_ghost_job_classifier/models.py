"""
Model definitions: LSTM, BERT, and RoBERTa for binary text classification.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(out)
        logits = self.classifier(out)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, logits) if loss is not None else logits


def get_transformer_model(model_name, num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
