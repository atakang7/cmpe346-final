# CMPE 346 final project

Three course experiments compare LSTM, BERT, and RoBERTa classifiers on media bias, sexism, and ghost-job risk.

```text
+-------------+  loads   +---------+  writes  +----------------------+
| BABEDataset | -------->| main.py | -------->| results/results.json |
+-------------+          +---------+          +----------------------+
```

## Run

Use Python 3.11.11 and a separate virtual environment per experiment; run these commands from the repository root.

Media bias:

```bash
python -m pip install -r student1_atakan_media_bias/requirements.txt
python student1_atakan_media_bias/main.py
```

Sexism (place `edos_labelled_aggregated.csv` in `student2_ece_sexism/data/`):

```bash
python -m pip install -r student2_ece_sexism/requirements.txt
python student2_ece_sexism/src/train.py
python student2_ece_sexism/src/plot.py
```

Ghost-job risk:

```bash
python -m pip install -r student3_talha_ghost_job_classifier/requirements.txt
python student3_talha_ghost_job_classifier/main.py
```

With `pdflatex` installed, build the report and slides using `bash report/render.sh` and `bash presentation/render.sh`.

## Config

| var | default | what |
| --- | --- | --- |
| `EDOS_DATA_PATH` | `student2_ece_sexism/data/edos_labelled_aggregated.csv` | EDOS input |
| `GHOST_JOB_DATASET_URL` | `https://zenodo.org/api/records/20321172/files/unified_core.parquet/content` | Ghost-job parquet source |

Export overrides in the shell; `.env` is not loaded automatically.

## Limits

- EDOS data and fine-tuned checkpoints are not included.
- Training needs model downloads and may be slow on CPU.
- CI does not retrain models or verify reported scores.

## CI

```text
+------------+  runs    +---------------------+  uses    +-----------------+  runs    +------------+  runs    +---------+
| checks.yml | -------->| actions/checkout@v4 | -------->| setup-python@v5 | -------->| compileall | -------->| bash -n |
+------------+          +---------------------+          +-----------------+          +------------+          +---------+
```
