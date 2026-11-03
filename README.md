# CMPE 346: text classification

Three independent course experiments compare an LSTM, BERT, and RoBERTa for news bias, online sexism, and ghost-job risk. Each experiment trains locally and writes evaluation results to its own folder.

## Quick start

Use Python 3.11.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

Run the media-bias experiment (the BABE dataset downloads on first use):

```bash
python -m pip install -r student1_atakan_media_bias/requirements.txt
python student1_atakan_media_bias/main.py
```

Run the sexism experiment after placing `edos_labelled_aggregated.csv` in `student2_ece_sexism/data/`:

```bash
python -m pip install -r student2_ece_sexism/requirements.txt
python student2_ece_sexism/src/train.py
python student2_ece_sexism/src/plot.py
```

Run the ghost-job experiment (downloads `unified_core.parquet` from Zenodo record 20321172):

```bash
python -m pip install -r student3_talha_ghost_job_classifier/requirements.txt
python student3_talha_ghost_job_classifier/main.py
```

Install each experiment's requirements in a separate virtual environment if running more than one; their dependency versions are not locked together.

## Configuration

- `EDOS_DATA_PATH`: optional path to the EDOS CSV when it is not under `student2_ece_sexism/data/`.
- `GHOST_JOB_DATASET_URL`: optional override for the Zenodo parquet download URL.
- `.env.example` lists these settings. Python does not load `.env` files automatically; export any overrides in your shell before running.

The datasets, cached weights, and generated results live under the experiment folders. Datasets and checkpoints are excluded from Git.

## How it works

Each experiment loads its own labeled text, trains three binary classifiers, then reports metrics on a held-out split. Student 1 uses the BABE dataset with accuracy and weighted F1; Student 2 uses EDOS with accuracy and macro F1; Student 3 labels job postings using visible wording and reports accuracy, precision, recall, and weighted F1.

See the README in each experiment folder for the dataset details and recorded results.

## Usage and report builds

These are local training scripts, not a deployed service. The media-bias and ghost-job scripts write JSON and PNG figures under their `results/` directories. The sexism script writes `results/results.csv`; `src/plot.py` generates its chart.

To build the written report or slides, install `pdflatex` and run `bash report/render.sh` or `bash presentation/render.sh` from the repository root.

## Limitations

Training downloads pretrained model weights and may be slow on CPU. The EDOS CSV is not included; you must obtain it from the SemEval-2023 Task 10 dataset release. Recorded metrics reflect earlier local runs, not a CI benchmark. GitHub Actions checks Python and shell syntax only; it does not retrain models or validate the datasets.
