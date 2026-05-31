# Contributing Guide — CMPE346 Group 2

## Overview

This is a group project for CMPE346 Natural Language Processing.
Each student fine-tunes 3 shared models (LSTM, BERT, RoBERTa) on their own dataset and reports Accuracy + F1.

**Reference implementation:** `student1_atakan_media_bias/` — fully working. Read it before writing anything.

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/atakang7/cmpe346-final.git
cd cmpe346-final
pip install -r requirements.txt
```

---

## Step 2 — Create your folder

Copy the structure from the reference:

```bash
cp -r student1_atakan_media_bias studentN_yourname_yourtopic
```

Rename it to match your name and topic, e.g. `student2_ali_hate_speech`.

---

## Step 3 — Implement your dataset

Edit `dataset.py`. Your job is to load your HuggingFace dataset and return items with:
- `input_ids` — tokenized text (for transformers) or raw text (for LSTM)
- `attention_mask`
- `label` — integer class label

Look at `student1_atakan_media_bias/dataset.py` as the exact template.

---

## Step 4 — Do NOT touch these files

Copy these directly from `student1_atakan_media_bias/` — they work for any classification task:

- `models.py` — LSTM, BERT, RoBERTa definitions (same for everyone)
- `train.py` — training loop (same for everyone)
- `evaluate.py` — accuracy + F1 (same for everyone)

---

## Step 5 — Adapt main.py

In `main.py`, change only these 4 lines at the top:

```python
# Change these to match your info
STUDENT_NAME = "Your Name"
STUDENT_ID = "XXXXXXXXX"
TOPIC = "Your NLP Topic"
DATASET = "huggingface/dataset-name"
```

And update the dataset loading calls to use your dataset class instead of `BABEDataset`.

---

## Step 6 — Run and verify

```bash
cd studentN_yourname_yourtopic
python main.py
```

When it finishes, check that `results/results.json` was created and contains real numbers:

```json
{
  "student": "Your Name",
  "student_id": "XXXXXXXXX",
  "topic": "Your NLP Topic",
  "dataset": "huggingface/dataset-name",
  "results": {
    "LSTM":    {"accuracy": 0.7812, "f1": 0.7634},
    "BERT":    {"accuracy": 0.8901, "f1": 0.8876},
    "RoBERTa": {"accuracy": 0.9012, "f1": 0.8994}
  }
}
```

---

## Step 7 — Push to the repo

```bash
git add studentN_yourname_yourtopic/
git commit -m "Add student2 - Ali - Hate Speech Detection"
git push
```

---

## Hard rules

| Rule | Why |
|------|-----|
| No `.ipynb` files | Blocked by `.gitignore` and violates submission rules |
| No model weights (`.pt`, `.pth`, `.bin`) | Blocked by `.gitignore` |
| No datasets (`.csv`, `.json` except results) | Blocked by `.gitignore` |
| `results/results.json` must be committed | This is how we verify your experiments are done |
| Use classes and separate files | Grading criteria requires it |

---

## Using AI (Claude, ChatGPT, etc.)

You can use AI to help write your `dataset.py` and `main.py`.
Give it this prompt as context:

> "I am doing a text classification project. I need to implement dataset.py that loads [YOUR DATASET] from HuggingFace using the datasets library. It should work with the same interface as this reference: [paste student1's dataset.py]. My dataset name is [YOUR DATASET NAME] and the label column is [LABEL COLUMN]."

Do NOT ask AI to rewrite `models.py`, `train.py`, or `evaluate.py` — they already work.
