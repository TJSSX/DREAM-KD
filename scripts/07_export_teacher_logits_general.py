# -*- coding: utf-8 -*-
"""
07_export_teacher_logits_general.py

Export teacher logits for a generic classification dataset.

Example:
    python scripts/07_export_teacher_logits_general.py ^
        --dataset_name 20ng ^
        --data_dir data_processed_20ng ^
        --teachers bert roberta electra xlnet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_DIR = Path(r"D:\Pycharm_Work\DREAM_KD")


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
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
def export_logits_for_split(model, tokenizer, df, split_name, teacher_name, device, max_length, batch_size):
    dataset = TextDataset(df, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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


def load_num_labels(data_dir: Path, train_df: pd.DataFrame) -> int:
    info_path = data_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if "num_labels" in info:
            return int(info["num_labels"])

    return int(train_df["label"].nunique())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--teachers", nargs="+", required=True)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    data_dir = PROJECT_DIR / args.data_dir
    teacher_root = PROJECT_DIR / "outputs" / args.dataset_name / "teachers"
    logit_dir = PROJECT_DIR / "outputs" / args.dataset_name / "teacher_logits"
    logit_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Exporting teacher logits")
    print("=" * 80)
    print(f"Dataset:     {args.dataset_name}")
    print(f"Data dir:    {data_dir}")
    print(f"Teacher dir: {teacher_root}")
    print(f"Logit dir:   {logit_dir}")
    print(f"Teachers:    {args.teachers}")
    print(f"Device:      {device}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")

    split_paths = {
        "train": data_dir / "train.csv",
        "dev": data_dir / "dev.csv",
        "test": data_dir / "test.csv",
    }

    split_dfs = {}
    for split_name, path in split_paths.items():
        df = pd.read_csv(path)
        split_dfs[split_name] = df
        print(f"{split_name}: {len(df)}")

    num_labels = load_num_labels(data_dir, split_dfs["train"])
    print(f"num_labels: {num_labels}")

    metadata = {
        "dataset_name": args.dataset_name,
        "teachers": args.teachers,
        "splits": list(split_paths.keys()),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_labels": num_labels,
    }

    for teacher_name in args.teachers:
        model_path = teacher_root / teacher_name / "best_model"

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
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

            logits_path = logit_dir / f"{teacher_name}_{split_name}_logits.npy"
            labels_path = logit_dir / f"{split_name}_labels.npy"

            np.save(logits_path, logits)
            np.save(labels_path, labels)

            print(f"Saved logits: {logits_path} | shape={logits.shape}")
            print(f"Saved labels: {labels_path} | shape={labels.shape}")

        del model
        torch.cuda.empty_cache()

    metadata_path = logit_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()