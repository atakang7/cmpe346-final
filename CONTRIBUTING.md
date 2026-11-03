# Contributing

Keep classifier edits in their student folder and update the folder README when commands or outputs change.

## Verify

```bash
python -m compileall -q student1_atakan_media_bias student2_ece_sexism student3_talha_ghost_job_classifier
bash -n report/render.sh presentation/render.sh
```

Do not commit datasets, weights, local configuration, or personal identifiers.
