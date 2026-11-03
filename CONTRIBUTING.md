# Contributing

Changes to a classifier should stay in its student folder unless shared project behavior needs an update. Keep the dataset requirements and result paths in the corresponding README accurate.

Before opening a pull request, run:

```bash
python -m compileall -q student1_atakan_media_bias student2_ece_sexism student3_talha_ghost_job_classifier
bash -n report/render.sh presentation/render.sh
```

Do not commit datasets, model weights, local configuration, personal identifiers, or generated training outputs.
