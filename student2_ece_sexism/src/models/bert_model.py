import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class BERTClassifier(nn.Module):
    """
    BERT-based binary text classifier for sexism detection.
    Fine-tunes a pre-trained BERT model with a classification head.
    """

    def __init__(self, model_name="bert-base-uncased", num_labels=2, dropout=0.3):
        """
        Args:
            model_name  : HuggingFace model identifier
            num_labels  : number of output classes (2 for binary)
            dropout     : dropout rate applied before classification head
        """
        super(BERTClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids       : tokenized input tensor
            attention_mask  : attention mask tensor
            labels          : ground truth labels (optional, for loss computation)

        Returns:
            loss (if labels provided), logits
        """
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output