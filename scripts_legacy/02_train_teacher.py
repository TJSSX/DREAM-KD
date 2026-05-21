# -*- coding: utf-8 -*-
"""
02_train_teacher.py

Train one teacher model for Amazon Multi-Domain Sentiment.

Example:
    python scripts/02_train_teacher.py --teacher_name bert --model_name bert-base-uncased

After debugging:
    python scripts/02_train_teacher.py --teacher_name roberta --model_name roberta-base
    python scripts/02_train_teacher.py --teacher_name electra --model_name google/electra-base-discriminator
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
DATA_DIR = PROJECT_DIR / "data_processed"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "teachers"

NUM_LABELS = 2


class TextClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
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


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev.csv")
    test_in_df = pd.read_csv(DATA_DIR / "test_in.csv")
    test_ood_df = pd.read_csv(DATA_DIR / "test_ood.csv")

    return train_df, dev_df, test_in_df, test_ood_df


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64)):
            clean_metrics[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            clean_metrics[k] = int(v)
        else:
            clean_metrics[k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # 本地 4060 Laptop 可以先 True。若报错，再改成 False。
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    print("=" * 80)
    print("Training teacher")
    print("=" * 80)
    print(f"Teacher name: {args.teacher_name}")
    print(f"Model name:   {args.model_name}")
    print(f"Project dir:  {PROJECT_DIR}")
    print(f"Data dir:     {DATA_DIR}")
    print(f"CUDA:         {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")

    train_df, dev_df, test_in_df, test_ood_df = load_data()

    print("\nData sizes:")
    print(f"  train:    {len(train_df)}")
    print(f"  dev:      {len(dev_df)}")
    print(f"  test_in:  {len(test_in_df)}")
    print(f"  test_ood: {len(test_ood_df)}")

    teacher_dir = OUTPUT_DIR / args.teacher_name
    teacher_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
    )

    train_dataset = TextClassificationDataset(train_df, tokenizer, args.max_length)
    dev_dataset = TextClassificationDataset(dev_df, tokenizer, args.max_length)
    test_in_dataset = TextClassificationDataset(test_in_df, tokenizer, args.max_length)
    test_ood_dataset = TextClassificationDataset(test_ood_df, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(teacher_dir / "checkpoints"),

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

    print("\nEvaluating on in-domain test...")
    test_in_metrics = trainer.evaluate(test_in_dataset)
    print(test_in_metrics)

    print("\nEvaluating on OOD test...")
    test_ood_metrics = trainer.evaluate(test_ood_dataset)
    print(test_ood_metrics)

    save_metrics(dev_metrics, teacher_dir / "dev_metrics.json")
    save_metrics(test_in_metrics, teacher_dir / "test_in_metrics.json")
    save_metrics(test_ood_metrics, teacher_dir / "test_ood_metrics.json")

    print("\nSaving best model and tokenizer...")
    trainer.save_model(str(teacher_dir / "best_model"))
    tokenizer.save_pretrained(str(teacher_dir / "best_model"))

    run_config = vars(args)
    run_config["project_dir"] = str(PROJECT_DIR)
    run_config["data_dir"] = str(DATA_DIR)
    run_config["teacher_dir"] = str(teacher_dir)

    save_metrics(run_config, teacher_dir / "run_config.json")

    print("\nDone.")
    print(f"Teacher saved to: {teacher_dir / 'best_model'}")


if __name__ == "__main__":
    main()