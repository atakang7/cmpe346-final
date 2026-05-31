"""
Dataset loading and preprocessing for Media Bias Detection (BABE dataset).
"""
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class BABEDataset(Dataset):
    def __init__(self, split, tokenizer_name=None, max_length=128):
        raw = load_dataset("mediabiasgroup/BABE", split=split)
        self.texts = raw["text"]
        self.labels = raw["label"]
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "label": torch.tensor(label, dtype=torch.long),
            }
        return {"text": text, "label": torch.tensor(label, dtype=torch.long)}


def get_vocab(dataset):
    """Build vocabulary from a list of BABEDataset items (for LSTM)."""
    from collections import Counter
    import re

    counter = Counter()
    for text in dataset.texts:
        tokens = re.findall(r'\w+', text.lower())
        counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(20000):
        vocab[word] = len(vocab)
    return vocab
