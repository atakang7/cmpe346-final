# Sexism detection (EDOS)

Binary classification of EDOS posts as sexist or not sexist, using BERT, RoBERTa, and a bidirectional LSTM. The training script reports test accuracy and macro F1 after three epochs.

## Dataset

Obtain `edos_labelled_aggregated.csv` from the SemEval-2023 Task 10 EDOS release. Place it at `student2_ece_sexism/data/edos_labelled_aggregated.csv`, or set `EDOS_DATA_PATH` to its location. The CSV must contain `text`, `label_sexist`, and `split` columns with train, dev, and test rows. The dataset is not committed.

## Run

From the repository root with Python 3.11.11:

```bash
python -m pip install -r student2_ece_sexism/requirements.txt
python student2_ece_sexism/src/train.py
python student2_ece_sexism/src/plot.py
```

Training writes `student2_ece_sexism/results/results.csv`. Plotting reads that file and writes `results/comparison.png` in the same directory. Both commands work regardless of the current working directory when invoked with paths to the scripts.

## Recorded test results

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| BERT | 0.8565 | 0.8115 |
| RoBERTa | 0.8545 | 0.7981 |
| LSTM | 0.8365 | 0.7530 |

These numbers are from the course run. No pretrained fine-tuned checkpoints are included.
