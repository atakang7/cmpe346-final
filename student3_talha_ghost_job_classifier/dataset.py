"""
Dataset loading and preprocessing for ghost-job risk classification.

The code downloads Zenodo record 20321172's unified_core.parquet on first run,
creates an evidence-based binary target, and writes local train/validation/test
CSV files. Datasets and generated CSVs are ignored by git.
"""
from collections import Counter
from pathlib import Path
import re

import pandas as pd
import requests
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from transformers import AutoTokenizer


RANDOM_SEED = 42
NEGATIVE_RATIO = 3
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "unified_core.parquet"
PROCESSED_DIR = DATA_DIR / "processed_ghost_adjudicated_ratio3"
ZENODO_URL = "https://zenodo.org/api/records/20321172/files/unified_core.parquet/content"

READ_COLUMNS = [
    "uid",
    "title",
    "description_core_llm",
    "source",
    "period",
    "company_name_canonical",
    "company_industry",
    "location",
    "skill_themes",
    "role_families",
    "is_aggregator",
    "ghost_assessment_llm",
]

POSITIVE_EVIDENCE_PATTERNS = {
    "not_funded": [
        r"\bnot funded\b",
        r"\bnot yet funded\b",
        r"\bfunding (?:is )?not\b",
    ],
    "being_bid": [
        r"\bbeing bid\b",
        r"\bbid on\b",
        r"\bbidding on\b",
        r"\bpending bid\b",
    ],
    "contingent_award": [
        r"\bcontingent\b.{0,120}\b(?:award|contract|selected|selection|funding|win|winner)\b",
        r"\b(?:award|contract|selected|selection|funding|win|winner)\b.{0,120}\bcontingent\b",
        r"\bpending (?:award|contract|funding)\b",
        r"\baward of (?:a )?contract\b",
    ],
    "no_current_opening": [
        r"\bdo not currently have an opening\b",
        r"\bno current opening\b",
        r"\bno immediate opening\b",
        r"\bno active opening\b",
    ],
    "talent_pool": [
        r"\btalent (?:community|pool|pipeline|network)\b",
        r"\balways building (?:our )?(?:talent|candidate)\b",
        r"\bjoin our talent community\b",
    ],
    "future_pipeline": [
        r"\bfuture opportunit(?:y|ies)\b",
        r"\bpipeline opportunit(?:y|ies)\b",
        r"\bfuture openings?\b",
    ],
}

UNCLEAR_PATTERNS = {
    "unpaid_or_volunteer": [
        r"\bunpaid\b",
        r"\bvoluntary\b",
        r"\bvolunteer\b",
    ],
    "expert_network_ai_training": [
        r"\bexpert network\b",
        r"\bai training\b",
        r"\btrain ai\b",
        r"\bevaluate ai\b",
        r"\bai responses\b",
    ],
}


def stringify_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if str(item).strip())
    if hasattr(value, "tolist") and not isinstance(value, str):
        items = value.tolist()
        if isinstance(items, list):
            return ", ".join(str(item) for item in items if str(item).strip())
        value = items
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_key(value) -> str:
    value = stringify_value(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def is_healthcare_posting(row: pd.Series) -> bool:
    text = " ".join(
        stringify_value(row.get(column))
        for column in ["title", "company_industry", "skill_themes", "role_families"]
    ).lower()
    return bool(
        re.search(r"\b(?:rn|registered nurse|travel nurse|nursing|therapist|hospital|healthcare)\b", text)
    )


def content_text(row: pd.Series) -> str:
    parts = [
        stringify_value(row.get("title")),
        stringify_value(row.get("company_name_canonical")),
        stringify_value(row.get("company_industry")),
        stringify_value(row.get("location")),
        stringify_value(row.get("skill_themes")),
        stringify_value(row.get("role_families")),
        stringify_value(row.get("description_core_llm")),
    ]
    return "\n\n".join(part for part in parts if part)


def evidence_text(row: pd.Series) -> str:
    return " ".join(
        stringify_value(row.get(column))
        for column in ["title", "company_name_canonical", "company_industry", "description_core_llm"]
    ).lower()


def matching_evidence(text: str, pattern_groups: dict[str, list[str]]) -> list[str]:
    matches = []
    for name, patterns in pattern_groups.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            matches.append(name)
    return matches


def adjudicate(row: pd.Series) -> tuple[str, str]:
    text = evidence_text(row)
    positive_evidence = matching_evidence(text, POSITIVE_EVIDENCE_PATTERNS)
    unclear_evidence = matching_evidence(text, UNCLEAR_PATTERNS)
    old_label = stringify_value(row.get("ghost_assessment_llm"))

    if positive_evidence:
        return "likely_ghost", ";".join(positive_evidence)
    if unclear_evidence:
        return "unclear", ";".join(unclear_evidence)
    if old_label == "ghost_likely":
        return "unclear", "old_ghost_likely_without_visible_evidence"
    if old_label == "realistic":
        return "realistic", "no_visible_ghost_evidence"
    return "unclear", f"old_label_{old_label or 'missing'}"


def download_raw_data() -> None:
    if RAW_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {RAW_PATH}")
    with requests.get(ZENODO_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with RAW_PATH.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def add_group_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["company_title_key"] = (
        df["company_name_canonical"].apply(normalize_key)
        + " | "
        + df["title"].apply(normalize_key)
    )
    return df


def split_dataframes(df: pd.DataFrame):
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    folds = list(splitter.split(df, df["label"], groups=df["company_title_key"]))
    validation_indices = folds[0][1]
    test_indices = folds[1][1]
    held_out = set(validation_indices) | set(test_indices)
    train_indices = [index for index in range(len(df)) if index not in held_out]
    return (
        df.iloc[train_indices].copy(),
        df.iloc[validation_indices].copy(),
        df.iloc[test_indices].copy(),
    )


def sample_negatives(split_df: pd.DataFrame) -> pd.DataFrame:
    positives = split_df[split_df["label"] == 1]
    negatives = split_df[split_df["label"] == 0]
    requested_negatives = min(len(negatives), len(positives) * NEGATIVE_RATIO)
    sampled_negatives = negatives.sample(n=requested_negatives, random_state=RANDOM_SEED)
    return (
        pd.concat([positives, sampled_negatives])
        .sample(frac=1.0, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )


def prepare_dataset() -> None:
    if all((PROCESSED_DIR / f"{split}.csv").exists() for split in ["train", "validation", "test"]):
        return

    download_raw_data()
    df = pd.read_parquet(RAW_PATH, columns=READ_COLUMNS)
    df = df[df["ghost_assessment_llm"].isin(["realistic", "inflated", "ghost_likely"])].copy()
    df = df[~df.apply(is_healthcare_posting, axis=1)].copy()

    labels_and_evidence = df.apply(adjudicate, axis=1)
    df["adjudicated_label"] = [item[0] for item in labels_and_evidence]
    df["adjudication_evidence"] = [item[1] for item in labels_and_evidence]
    df["text"] = df.apply(content_text, axis=1)

    eligible = df[df["adjudicated_label"].isin(["likely_ghost", "realistic"])].copy()
    eligible["label"] = (eligible["adjudicated_label"] == "likely_ghost").astype(int)
    eligible = add_group_key(eligible.reset_index(drop=True))

    train_df, validation_df, test_df = split_dataframes(eligible)
    train_df = sample_negatives(train_df)
    validation_df = sample_negatives(validation_df)
    test_df = sample_negatives(test_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["uid", "text", "label", "adjudicated_label", "adjudication_evidence", "ghost_assessment_llm"]
    train_df[columns].to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation_df[columns].to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df[columns].to_csv(PROCESSED_DIR / "test.csv", index=False)

    for name, split_df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        print(f"{name}: {len(split_df)} rows, labels={split_df['label'].value_counts().sort_index().to_dict()}")


class GhostJobDataset(Dataset):
    def __init__(self, split, tokenizer_name=None, max_length=256):
        prepare_dataset()
        path = PROCESSED_DIR / f"{split}.csv"
        df = pd.read_csv(path)
        self.texts = df["text"].fillna("").astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "label": torch.tensor(label, dtype=torch.long),
            }
        return {"text": text, "label": torch.tensor(label, dtype=torch.long)}


def get_vocab(dataset, indices=None):
    counter = Counter()
    texts = [dataset.texts[i] for i in indices] if indices is not None else dataset.texts
    for text in texts:
        tokens = re.findall(r"\w+", text.lower())
        counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(20000):
        vocab[word] = len(vocab)
    return vocab
