# Contributing

Keep changes to each classifier in its student directory. The [data formats](docs/reference/data-and-outputs.md) differ between experiments.

From the repository root, check Python and render script syntax before opening a pull request:

```bash
python -m compileall -q student1_atakan_media_bias student2_ece_sexism student3_talha_ghost_job_classifier
bash -n report/render.sh presentation/render.sh
```

For model changes, run the relevant experiment using its README commands and inspect the saved metrics. Do not commit datasets, checkpoints, notebooks, or generated PDFs.
