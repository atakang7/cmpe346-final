# Prepare EDOS data

1. Obtain the aggregated CSV for SemEval-2023 Task 10 (EDOS).
2. From the repository root, place it at the path read by the classifier:

```bash
mkdir -p student2_ece_sexism/data
cp /path/to/edos_labelled_aggregated.csv student2_ece_sexism/data/
```

3. Check that the CSV has `text`, `label_sexist`, and `split` columns. `label_sexist` must contain `sexist` or `not sexist`; `split` must contain `train`, `dev`, and `test`.
4. Train from the repository root:

```bash
python student2_ece_sexism/src/train.py
```

The dataset file is excluded from git.
