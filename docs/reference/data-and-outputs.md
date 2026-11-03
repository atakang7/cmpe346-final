# Data and output formats

## Input datasets

| Component | Source | Labels | Split |
| --- | --- | --- | --- |
| Media bias | Hugging Face `mediabiasgroup/BABE` | `0` not biased, `1` biased | train/test; 10% of train used for validation |
| Sexism | EDOS, SemEval-2023 Task 10 | `not sexist`, `sexist` | `split` column: train/dev/test |
| Ghost-job risk | Zenodo record `20321172`, `unified_core.parquet` | `0` realistic, `1` likely ghost | Stratified group folds; downsample negatives to at most 3 per positive |

## Output fields

| Component | File | Fields |
| --- | --- | --- |
| Media bias | `student1_atakan_media_bias/results/results.json` | `student`, `topic`, `dataset`, `results`, `loss_histories` |
| Sexism | `student2_ece_sexism/results/results.csv` | `model`, `accuracy`, `f1` |
| Ghost-job risk | `student3_talha_ghost_job_classifier/results/results.json` | `student`, `topic`, `dataset`, `results`, `loss_histories` |

Media bias and ghost-job experiments report weighted F1; the EDOS experiment reports macro F1. Ghost-job `results` also contains precision and recall. All three classifiers report accuracy.
