# CMPE346 Final Project — Group 2

## Group Task: Text Classification
## Models: LSTM, BERT (bert-base-uncased), RoBERTa (roberta-base)

## Students
| Folder | Student | Topic | Dataset |
|--------|---------|-------|---------|
| student1_atakan_media_bias | Atakan Gül (121200152) | Media Bias Detection | mediabiasgroup/BABE |
| student2_ece_sexism | Ece Doğa Gül (121200123) | Sexism Detection | EDOS / SemEval-2023 Task 10 |
| student3_talha_ghost_job_classifier | Talha Demir (122200106) | Ghost-Job Risk Classification | Zenodo 20321172 unified_core |

## Requirements
```
pip install -r requirements.txt
```

## How to Run
Each student's code is self-contained in their folder.

```bash
# Atakan - Media Bias Detection
cd student1_atakan_media_bias
python main.py

# Ece - Sexism Detection
cd student2_ece_sexism/src
python train.py

# Talha - Ghost-Job Risk Classification
cd student3_talha_ghost_job_classifier
python main.py
```
