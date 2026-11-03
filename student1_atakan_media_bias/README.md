Part of [cmpe346-final](../README.md)

# Media bias detection

Classifies BABE news sentences as biased (1) or not biased (0) with a BiLSTM, BERT, and RoBERTa.

```text
+------------+  batches  +---------+  scores  +---------+
| dataset.py | --------> | main.py | -------> | plot.py |
+------------+           +---------+          +---------+
```

## Run

From this directory with Python 3.11:

```bash
python -m pip install -r requirements.txt
python main.py
```

`python examples.py` prints sample predictions after the models have been saved.

## Test

```bash
python -m compileall -q .
```

## Config

| Var | Default | What |
| --- | --- | --- |
| `EPOCHS` | 3 | Training epochs |
| `BATCH_SIZE` | 16 | Samples per batch |
| `MAX_LENGTH` | 128 | Tokens per sample |
| `VAL_SPLIT` | 0.1 | Fraction of train data held for validation |

These constants are in `main.py`.

## Interfaces

| Path | Reads or writes |
| --- | --- |
| `dataset.py` | Reads `mediabiasgroup/BABE` from Hugging Face |
| `saved_models/*.pt` | Writes model weights; loads them on subsequent runs |
| `saved_models/LSTM_vocab.pkl` | Writes LSTM vocabulary for `examples.py` |
| `results/results.json` | Writes accuracy, weighted F1, and loss histories |
| `results/*.png` | Writes model comparison, confusion matrices, and loss plots |

See [dataset and result formats](../docs/reference/data-and-outputs.md).
