import pandas as pd
from torch.utils.data import Dataset


class EDOSDataset(Dataset):
    """
    PyTorch Dataset class for the EDOS Sexism Detection dataset.
    Handles loading and preprocessing of the dataset for model training.
    """

    LABEL_MAP = {"not sexist": 0, "sexist": 1}

    def __init__(self, dataframe, tokenizer, max_length=128):
        """
        Args:
            dataframe   : pandas DataFrame containing 'text' and 'label_sexist' columns
            tokenizer   : HuggingFace tokenizer for BERT/RoBERTa, or None for LSTM
            max_length  : maximum token length for tokenization
        """
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label_sexist"].map(self.LABEL_MAP).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": label,
            }
        else:
            # LSTM için ham metin döndür
            return {"text": text, "label": label}


def load_splits(data_path):
    """
    Loads and splits the EDOS dataset into train, dev, and test sets.

    Args:
        data_path : path to the edos_labelled_aggregated.csv file

    Returns:
        train_df, dev_df, test_df : pandas DataFrames for each split
    """
    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, dev_df, test_df