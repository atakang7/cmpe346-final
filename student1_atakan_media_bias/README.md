# Media bias detection

`main.py` trains a BiLSTM, BERT, and RoBERTa on the BABE news-sentence dataset (3,121 train and 1,000 test rows).

```text
+-------------+  loads  +---------+  writes  +----------------------+
| BABEDataset | ------> | main.py | -------> | results/results.json |
+-------------+         +---------+          +----------------------+
```

## Run

From the repository root with Python 3.11.11:

```bash
python -m pip install -r student1_atakan_media_bias/requirements.txt
python student1_atakan_media_bias/main.py
```

## Results

| model | accuracy | weighted F1 |
| --- | ---: | ---: |
| LSTM | 0.6560 | 0.6511 |
| BERT | 0.8260 | 0.8266 |
| RoBERTa | 0.8460 | 0.8464 |

## Limits

- Scores are from the original course run, not a CI test.
- Existing `saved_models/` checkpoints are reused.
