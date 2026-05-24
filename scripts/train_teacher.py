import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer import train_one_epoch, evaluate_model
from src.utils import ensure_dir, get_safe_model_name, save_json, set_seed


class TextClassificationDataset(Dataset):
    """Simple text classification dataset for CSV files with text,label columns."""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 256):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df["text"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()
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
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Train a teacher model for DREAM-KD.")

    parser.add_argument("--dataset", type=str, required=True, choices=["amazon_marc"])
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="outputs")

    parser.add_argument("--num_labels", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.data_dir is None:
        data_dir = PROJECT_ROOT / "data_processed" / args.dataset
    else:
        data_dir = Path(args.data_dir)

    train_path = data_dir / "train.csv"
    dev_path = data_dir / "dev.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    safe_model_name = get_safe_model_name(args.model_name)
    output_dir = PROJECT_ROOT / args.output_root / args.dataset / "teachers" / safe_model_name
    ensure_dir(str(output_dir))

    print("=" * 80)
    print("Training teacher model")
    print(f"Dataset:      {args.dataset}")
    print(f"Model:        {args.model_name}")
    print(f"Data dir:     {data_dir}")
    print(f"Output dir:   {output_dir}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
    )
    model.to(device)

    train_dataset = TextClassificationDataset(
        csv_path=str(train_path),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    dev_dataset = TextClassificationDataset(
        csv_path=str(dev_path),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_dataset = TextClassificationDataset(
        csv_path=str(test_path),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
    )

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_dev_macro_f1 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        print("=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )

        dev_loss, dev_metrics = evaluate_model(
            model=model,
            dataloader=dev_loader,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }
        history.append(epoch_result)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Dev loss:   {dev_loss:.4f}")
        print(f"Dev metrics: {dev_metrics}")

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch

            print(f"New best model found at epoch {epoch}. Saving...")

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_dev_macro_f1": best_dev_macro_f1,
                    "model_name": args.model_name,
                    "dataset": args.dataset,
                    "args": vars(args),
                },
                str(output_dir / "best_info.json"),
            )

    print("=" * 80)
    print("Loading best model for final test evaluation")
    print("=" * 80)

    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    best_model.to(device)

    test_loss, test_metrics = evaluate_model(
        model=best_model,
        dataloader=test_loader,
        device=device,
    )

    final_results = {
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_dev_macro_f1,
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "history": history,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "num_train_examples": len(train_dataset),
        "num_dev_examples": len(dev_dataset),
        "num_test_examples": len(test_dataset),
    }

    save_json(final_results, str(output_dir / "results.json"))

    print("=" * 80)
    print("Final test results")
    print("=" * 80)
    print(final_results)
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()