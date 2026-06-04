# Media Bias Detection — Atakan Gül (121200152)

Binary text classification of news sentences as **biased** or **not biased**.

- **Task:** Media Bias Detection
- **Dataset:** [mediabiasgroup/BABE](https://huggingface.co/datasets/mediabiasgroup/BABE) (3,121 train / 1,000 test), loaded via `datasets`
- **Models:** BiLSTM (from scratch), BERT (`bert-base-uncased`), RoBERTa (`roberta-base`)
- **Metrics:** Accuracy, weighted F1

## Files
| File | Purpose |
|------|---------|
| `dataset.py` | Loads BABE, tokenizes, builds LSTM vocab (train split only) |
| `models.py` | `LSTMClassifier` and transformer loader |
| `train.py` | Training loop with validation and early stopping |
| `evaluate.py` | Accuracy / F1 / predictions |
| `plot.py` | Generates comparison, confusion-matrix, and loss figures |
| `main.py` | Entry point — trains all three models and writes results |
| `examples.py` | Prints qualitative prediction examples (requires trained models) |

## Run
```bash
pip install -r requirements.txt
python3 main.py
```
Trains LSTM, BERT, RoBERTa; writes metrics to `results/results.json` and figures to `results/`. Weights are cached in `saved_models/` and reused on later runs. Random seed is fixed to 42.

## Results (test set)
| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| LSTM | 0.6560 | 0.6511 |
| BERT | 0.8260 | 0.8266 |
| RoBERTa | 0.8460 | 0.8464 |
