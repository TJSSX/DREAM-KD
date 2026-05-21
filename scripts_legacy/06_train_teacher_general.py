# -*- coding: utf-8 -*-
"""
06_train_teacher_general.py

Generic teacher training script for classification datasets.

Supports data directory containing:
    train.csv
    dev.csv
    test.csv

Optional:
    dataset_info.json with "num_labels"

Example for 20 Newsgroups:
    python scripts/06_train_teacher_general.py ^
        --dataset_name 20ng ^
        --data_dir data_processed_20ng ^
        --teacher_name bert ^
        --model_name bert-base-uncased ^
        --fp16
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_DIR = Path(r"D:\Pycharm_Work\DREAM_KD")


class TextClassificationDataset(Dataset):
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

        encoded["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in encoded.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
    }


def load_num_labels(data_dir: Path, train_df: pd.DataFrame) -> int:
    info_path = data_dir / "dataset_info.json"

    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if "num_labels" in info:
            return int(info["num_labels"])

    return int(train_df["label"].nunique())


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_obj = {}
    for k, v in obj.items():
        if isinstance(v, (np.float32, np.float64)):
            clean_obj[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            clean_obj[k] = int(v)
        else:
            clean_obj[k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--teacher_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    data_dir = PROJECT_DIR / args.data_dir
    output_dir = PROJECT_DIR / "outputs" / args.dataset_name / "teachers" / args.teacher_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Training teacher")
    print("=" * 80)
    print(f"Dataset:      {args.dataset_name}")
    print(f"Data dir:     {data_dir}")
    print(f"Teacher name: {args.teacher_name}")
    print(f"Model name:   {args.model_name}")
    print(f"Output dir:   {output_dir}")
    print(f"CUDA:         {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    num_labels = load_num_labels(data_dir, train_df)

    print("\nData sizes:")
    print(f"  train: {len(train_df)}")
    print(f"  dev:   {len(dev_df)}")
    print(f"  test:  {len(test_df)}")
    print(f"  labels:{num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    train_dataset = TextClassificationDataset(train_df, tokenizer, args.max_length)
    dev_dataset = TextClassificationDataset(dev_df, tokenizer, args.max_length)
    test_dataset = TextClassificationDataset(test_df, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,

        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=1,
        report_to="none",

        fp16=args.fp16,

        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nEvaluating on dev...")
    dev_metrics = trainer.evaluate(dev_dataset)
    print(dev_metrics)

    print("\nEvaluating on test...")
    test_metrics = trainer.evaluate(test_dataset)
    print(test_metrics)

    save_json(dev_metrics, output_dir / "dev_metrics.json")
    save_json(test_metrics, output_dir / "test_metrics.json")

    print("\nSaving best model and tokenizer...")
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    run_config = vars(args)
    run_config["project_dir"] = str(PROJECT_DIR)
    run_config["data_dir_abs"] = str(data_dir)
    run_config["output_dir_abs"] = str(output_dir)
    run_config["num_labels"] = num_labels

    save_json(run_config, output_dir / "run_config.json")

    print("\nDone.")
    print(f"Teacher saved to: {output_dir / 'best_model'}")


if __name__ == "__main__":
    main()