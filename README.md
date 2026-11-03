# CMPE 346 final project

Three course experiments compare LSTM, BERT, and RoBERTa classifiers on media bias, sexism, and ghost-job risk.

```text
+-------------+  loads  +---------+  writes  +----------------------+
| BABEDataset | ------> | main.py | -------> | results/results.json |
+-------------+         +---------+          +----------------------+
```

## Run

Python 3.11.11; run from the repository root with a separate virtual environment for each experiment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate`.

Media bias (downloads BABE):

```bash
python -m pip install -r student1_atakan_media_bias/requirements.txt
python student1_atakan_media_bias/main.py
```

Sexism (requires `student2_ece_sexism/data/edos_labelled_aggregated.csv`):

```bash
python -m pip install -r student2_ece_sexism/requirements.txt
python student2_ece_sexism/src/train.py
python student2_ece_sexism/src/plot.py
```

Ghost-job risk (downloads Zenodo record 20321172):

```bash
python -m pip install -r student3_talha_ghost_job_classifier/requirements.txt
python student3_talha_ghost_job_classifier/main.py
```

Build the LaTeX files with `bash report/render.sh` and `bash presentation/render.sh` after installing `pdflatex`.

## Config

| var | default | what |
| --- | --- | --- |
| `EDOS_DATA_PATH` | `student2_ece_sexism/data/edos_labelled_aggregated.csv` | EDOS input CSV |
| `GHOST_JOB_DATASET_URL` | `https://zenodo.org/api/records/20321172/files/unified_core.parquet/content` | Ghost-job dataset download |

Export overrides in the shell; the scripts do not load `.env` automatically.

## Limits

- EDOS data and fine-tuned model weights are not included.
- Runs need pretrained model downloads and can be slow without a GPU.
- Recorded metrics are from course runs, not CI.

## CI

```text
+------------+  runs  +---------------------+  uses  +-----------------+  runs  +------------+  runs  +--------+
| checks.yml | -----> | actions/checkout@v4 | -----> | setup-python@v5 | -----> | compileall | -----> | bash -n |
+------------+        +---------------------+        +-----------------+        +------------+        +--------+
```

CI checks syntax only; it does not train the models.
