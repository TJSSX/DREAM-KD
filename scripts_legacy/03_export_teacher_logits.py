# -*- coding: utf-8 -*-
"""
03_export_teacher_logits.py

Export logits from trained teacher models.

Example:
    python scripts/03_export_teacher_logits.py

It will export:
    outputs/teacher_logits/
        bert_train_logits.npy
        bert_dev_logits.npy
        bert_test_in_logits.npy
        bert_test_ood_logits.npy
        roberta_train_logits.npy
        ...
        electra_test_ood_logits.npy
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_DIR = Path(r"D:\Pycharm_Work\DREAM_KD")
DATA_DIR = PROJECT_DIR / "data_processed"
TEACHER_DIR = PROJECT_DIR / "outputs" / "teachers"
LOGIT_DIR = PROJECT_DIR / "outputs" / "teacher_logits"

MAX_LENGTH = 128
BATCH_SIZE = 32

TEACHERS = {
    "bert": TEACHER_DIR / "bert" / "best_model",
    "roberta": TEACHER_DIR / "roberta" / "best_model",
    "electra": TEACHER_DIR / "electra" / "best_model",
}

SPLITS = {
    "train": DATA_DIR / "train.csv",
    "dev": DATA_DIR / "dev.csv",
    "test_in": DATA_DIR / "test_in.csv",
    "test_ood": DATA_DIR / "test_ood.csv",
}


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.domains = df["domain"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )

        item = {k: torch.tensor(v) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@torch.no_grad()
def export_logits_for_split(model, tokenizer, df: pd.DataFrame, split_name: str, teacher_name: str, device: torch.device):
    dataset = TextDataset(df, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_logits = []
    all_labels = []

    model.eval()

    for batch in tqdm(loader, desc=f"{teacher_name} | {split_name}"):
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits.detach().cpu().numpy()

        all_logits.append(logits)
        all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_logits, all_labels


def main():
    LOGIT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Exporting teacher logits")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Data dir:    {DATA_DIR}")
    print(f"Teacher dir: {TEACHER_DIR}")
    print(f"Logit dir:   {LOGIT_DIR}")
    print(f"Device:      {device}")

    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")

    split_dfs = {}
    for split_name, split_path in SPLITS.items():
        df = pd.read_csv(split_path)
        split_dfs[split_name] = df
        print(f"{split_name}: {len(df)} examples")

    metadata = {
        "teachers": list(TEACHERS.keys()),
        "splits": list(SPLITS.keys()),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_labels": 2,
    }

    for teacher_name, model_path in TEACHERS.items():
        print("\n" + "=" * 80)
        print(f"Loading teacher: {teacher_name}")
        print(f"Path: {model_path}")
        print("=" * 80)

        if not model_path.exists():
            raise FileNotFoundError(f"Cannot find teacher model: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.to(device)

        for split_name, df in split_dfs.items():
            logits, labels = export_logits_for_split(
                model=model,
                tokenizer=tokenizer,
                df=df,
                split_name=split_name,
                teacher_name=teacher_name,
                device=device,
            )

            logits_path = LOGIT_DIR / f"{teacher_name}_{split_name}_logits.npy"
            labels_path = LOGIT_DIR / f"{split_name}_labels.npy"

            np.save(logits_path, logits)

            # labels are same for all teachers; save/overwrite safely
            np.save(labels_path, labels)

            print(f"Saved logits: {logits_path} | shape={logits.shape}")
            print(f"Saved labels: {labels_path} | shape={labels.shape}")

        del model
        torch.cuda.empty_cache()

    metadata_path = LOGIT_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()