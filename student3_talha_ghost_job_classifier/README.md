# Ghost-job risk classification

`main.py` trains LSTM, BERT, and RoBERTa classifiers on job-posting text labeled by visible hiring language.

```text
+-----------------+  downloads  +------------+  loads   +---------+  writes  +----------------------+
| Zenodo 20321172 | ----------->| dataset.py | -------->| main.py | -------->| results/results.json |
+-----------------+             +------------+          +---------+          +----------------------+
```

## Run

From the repository root with Python 3.11.11:

```bash
python -m pip install -r student3_talha_ghost_job_classifier/requirements.txt
python student3_talha_ghost_job_classifier/main.py
```

## Config

| var | default | what |
| --- | --- | --- |
| `GHOST_JOB_DATASET_URL` | `https://zenodo.org/api/records/20321172/files/unified_core.parquet/content` | Parquet source |

## Limits

- Labels reflect posting text, not verified hiring outcomes.
- First run requires network access; checkpoints in `saved_models/` are reused.
