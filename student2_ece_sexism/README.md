# EDOS sexism detection

`src/train.py` trains BERT, RoBERTa, and a BiLSTM on the SemEval-2023 Task 10 EDOS split.

```text
+-----------------------------+  reads  +--------------+  writes  +---------------------+
| edos_labelled_aggregated.csv | ------> | src/train.py | -------> | results/results.csv |
+-----------------------------+         +--------------+          +---------------------+
```

## Run

Place the EDOS CSV in `student2_ece_sexism/data/` and run from the repository root:

```bash
python -m pip install -r student2_ece_sexism/requirements.txt
python student2_ece_sexism/src/train.py
python student2_ece_sexism/src/plot.py
```

## Config

| var | default | what |
| --- | --- | --- |
| `EDOS_DATA_PATH` | `student2_ece_sexism/data/edos_labelled_aggregated.csv` | Input CSV |

## Results

| model | accuracy | macro F1 |
| --- | ---: | ---: |
| BERT | 0.8565 | 0.8115 |
| RoBERTa | 0.8545 | 0.7981 |
| LSTM | 0.8365 | 0.7530 |

## Limits

- Input CSV must have `text`, `label_sexist`, and `split` columns.
- Reported scores are from the course run; weights are not included.
