# Ghost-job risk classification

Binary text classification of job postings using an LSTM, BERT, and RoBERTa. A positive label means the text includes wording associated with uncertain or future hiring; it does not establish whether the employer intended to hire.

## Dataset and labels

The input is `unified_core.parquet` from Zenodo record `20321172`. On first use, `dataset.py` downloads it into `data/`, filters records, derives labels from visible wording, and creates train/validation/test CSV files. Set `GHOST_JOB_DATASET_URL` to override the source URL. The dataset and model weights are not committed.

## Run

From the repository root with Python 3.11.11:

```bash
python -m pip install -r student3_talha_ghost_job_classifier/requirements.txt
python student3_talha_ghost_job_classifier/main.py
```

Metrics are written to `results/results.json`; figures go to the same directory. The scripts reuse checkpoints in `saved_models/` when present. Delete that directory to retrain.

## Limits

The labels come from a set of text patterns and an existing assessment field, not verified hiring outcomes. The download requires network access, and the models may take time to train without a GPU.
