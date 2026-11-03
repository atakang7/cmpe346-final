Part of [cmpe346-final](../README.md)

# Ghost-job risk classification

Classifies job postings as realistic (0) or likely ghost-job risk (1) from visible text signals.

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

The first run downloads the Zenodo parquet file and creates local data splits.

## Test

```bash
python -m compileall -q .
```

## Config

| Var | Default | What |
| --- | --- | --- |
| `EPOCHS` | 3 | Training epochs |
| `BATCH_SIZE` | 8 | Samples per batch |
| `MAX_LENGTH` | 256 | Tokens per sample |
| `NEGATIVE_RATIO` | 3 | Maximum negatives per positive, per split |

The training constants are in `main.py`; `NEGATIVE_RATIO` is in `dataset.py`.

## Interfaces

| Path | Reads or writes |
| --- | --- |
| `data/unified_core.parquet` | Downloads Zenodo record `20321172` |
| `data/processed_ghost_adjudicated_ratio3/*.csv` | Writes train, validation, and test splits |
| `saved_models/*.pt` | Writes and reloads model weights |
| `results/results.json` | Writes accuracy, precision, recall, weighted F1 |
| `results/*.png` | Writes comparison, confusion matrices, and loss plots |

Positive labels come from rule matches against the posting text. See [architecture](../docs/explanation/architecture.md).
