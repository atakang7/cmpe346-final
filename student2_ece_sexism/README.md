# CMPE 346 - NLP Final Project
## Sexism Detection using LSTM, BERT, and RoBERTa

### Project Structure
FinalProjectCMPE346/
├── data/
│   └── edos_labelled_aggregated.csv
├── results/
│   ├── results.csv
│   └── evaluation_report.txt
├── src/
│   ├── models/
│   │   ├── init.py
│   │   ├── bert_model.py
│   │   ├── roberta_model.py
│   │   └── lstm_model.py
│   ├── init.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt

### Dataset
- **Name:** EDOS (Explainable Detection of Online Sexism)
- **Source:** SemEval-2023 Task 10
- **Task:** Binary Sexism Detection (sexist / not sexist)
- **Size:** 20,000 samples (Train: 14,000 | Dev: 2,000 | Test: 4,000)

### Requirements
```bash
pip install -r requirements.txt
```

### Training
```bash
cd src
python train.py
```

### Evaluation
```bash
cd src
python evaluate.py
```

### Results
| Model   | Accuracy | F1     |
|---------|----------|--------|
| BERT    | 0.8565   | 0.8115 |
| RoBERTa | 0.8545   | 0.7981 |
| LSTM    | 0.8365   | 0.7530 |