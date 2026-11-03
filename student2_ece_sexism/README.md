Part of [cmpe346-final](../README.md)

# Sexism detection

Classifies EDOS posts as sexist (1) or not sexist (0) with an LSTM, BERT, and RoBERTa.

```text
+----------------+  batches  +--------------+  CSV  +-------------+
| src/dataset.py | --------> | src/train.py | ----> | src/plot.py |
+----------------+           +--------------+       +-------------+
```

## Run

From this directory with Python 3.11:

```bash
python -m pip install -r requirements.txt
python src/train.py
python src/plot.py
```

Put the aggregated EDOS CSV at `data/edos_labelled_aggregated.csv` before training. See [prepare EDOS data](../docs/how-to/prepare-edos.md).

## Test

```bash
python -m compileall -q src
```

## Config

| Var | Default | What |
| --- | --- | --- |
| `BATCH_SIZE` | 32 | Samples per batch |
| `MAX_LENGTH` | 128 | Tokens per sample |
| `EPOCHS` | 3 | Training epochs |
| `LEARNING_RATE` | 2e-5 | BERT and RoBERTa learning rate |

These constants are in `src/train.py`.

## Interfaces

| Path | Reads or writes |
| --- | --- |
| `data/edos_labelled_aggregated.csv` | Reads labeled text and train/dev/test split |
| `results/results.csv` | Writes model, accuracy, and macro F1 |
| `results/comparison.png` | Plot output from `src/plot.py` |

See [dataset and result formats](../docs/reference/data-and-outputs.md).
