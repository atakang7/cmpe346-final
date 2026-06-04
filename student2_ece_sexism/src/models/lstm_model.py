import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM-based binary text classifier for sexism detection.
    Uses an embedding layer followed by a bidirectional LSTM and a classification head.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_labels=2, dropout=0.3):
        """
        Args:
            vocab_size  : size of the vocabulary
            embed_dim   : dimension of word embeddings
            hidden_dim  : number of hidden units in LSTM
            num_layers  : number of stacked LSTM layers
            num_labels  : number of output classes (2 for binary)
            dropout     : dropout rate
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        # bidirectional olduğu için bu şekilde
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, labels=None):
        """
        Args:
            input_ids   : tokenized input tensor (batch_size, seq_len)
            labels      : ground truth labels (optional, for loss computation)

        Returns:
            loss (if labels provided), logits
        """
        embedded = self.dropout(self.embedding(input_ids))

        lstm_out, (hidden, _) = self.lstm(embedded)

        # Son timestep yerine tüm sekansın ortalamasını al
        pooled = lstm_out.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits