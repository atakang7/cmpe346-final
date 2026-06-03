# Talha Demir — Ghost-Job Risk Classification

This folder contains the Student 2 component for the CMPE346 Group 2 final
project.

## Task

Binary text classification of job postings:

- `0`: realistic / no visible ghost-job evidence
- `1`: likely ghost-job risk based on visible textual evidence

The model does not prove whether a company truly hired. It detects visible
signals such as contingent award, pending funding, future-only opportunities,
talent-pool wording, and no-current-opening language.

## Dataset

Source: Zenodo record `20321172`, `unified_core.parquet`.

The script downloads the parquet file on first run and creates local splits
under `data/`. These files are ignored by git.

## Models

- LSTM trained from scratch
- BERT (`bert-base-uncased`)
- RoBERTa (`roberta-base`)

## Run

```bash
python3 main.py
```

Outputs:

```text
results/results.json
results/model_comparison.png
results/confusion_matrices.png
results/training_curves.png
```

Local generated files that should not be committed:

```text
data/
saved_models/
```
