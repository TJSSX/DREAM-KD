# -*- coding: utf-8 -*-
"""
10_prepare_yahoo.py

Prepare Yahoo Answers Topics for DREAM-KD experiments.

Output:
    data_processed/yahoo/
        train.csv
        dev.csv
        test.csv
        label_map.json
        dataset_info.json

Default sampled setting:
    train_per_class = 3000
    dev_per_class   = 500
    test_per_class  = 1000

Total:
    train = 30,000
    dev   = 5,000
    test  = 10,000

Usage:
    python scripts/10_prepare_yahoo.py

Debug usage:
    python scripts/10_prepare_yahoo.py --train_per_class 500 --dev_per_class 100 --test_per_class 300
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


PROJECT_DIR = Path(r"D:\Pycharm_Work\DREAM_KD")
OUT_DIR = PROJECT_DIR / "data_processed" / "yahoo"

SEED = 42


YAHOO_LABELS = {
    0: "Society & Culture",
    1: "Science & Mathematics",
    2: "Health",
    3: "Education & Reference",
    4: "Computers & Internet",
    5: "Sports",
    6: "Business & Finance",
    7: "Entertainment & Music",
    8: "Family & Relationships",
    9: "Politics & Government",
}


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label):
    """
    Yahoo Answers labels are usually 0..9.
    Some versions may be 1..10, so normalize if needed.
    """
    label = int(label)

    if 0 <= label <= 9:
        return label

    if 1 <= label <= 10:
        return label - 1

    raise ValueError(f"Unexpected label value: {label}")


def load_yahoo_from_hf():
    """
    Try several common HF dataset names.
    """
    candidates = [
        ("yahoo_answers_topics", None),
        ("fancyzhx/yahoo_answers_topics", None),
        ("community-datasets/yahoo_answers_topics", None),
    ]

    last_error = None

    for dataset_name, config_name in candidates:
        try:
            print(f"Trying to load dataset: {dataset_name}")
            if config_name is None:
                ds = load_dataset(dataset_name)
            else:
                ds = load_dataset(dataset_name, config_name)
            print(f"Loaded dataset: {dataset_name}")
            return ds
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            last_error = e

    raise RuntimeError(f"Failed to load Yahoo Answers dataset. Last error: {last_error}")


def convert_split_to_df(ds_split, split_name: str) -> pd.DataFrame:
    rows = []
    columns = set(ds_split.column_names)

    print(f"\nColumns in {split_name}: {ds_split.column_names}")
    print(f"First example in {split_name}: {ds_split[0]}")

    for item in ds_split:
        # Yahoo Answers Topics uses "topic" as label column
        label = normalize_label(item["topic"])

        parts = []

        for col in [
            "question_title",
            "question_content",
            "best_answer",
        ]:
            if col in columns:
                value = clean_text(item.get(col, ""))
                if value:
                    parts.append(value)

        if len(parts) == 0:
            raise ValueError(
                f"Cannot find text columns in dataset split. Available columns: {columns}"
            )

        text = " ".join(parts)
        text = clean_text(text)

        if len(text.split()) < 5:
            continue

        rows.append(
            {
                "text": text,
                "label": label,
                "label_name": YAHOO_LABELS[label],
                "domain": "yahoo",
                "source_split": split_name,
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return df


def split_train_dev(train_full: pd.DataFrame, train_per_class: int, dev_per_class: int, seed: int):
    train_parts = []
    dev_parts = []

    needed = train_per_class + dev_per_class

    for label, group in train_full.groupby("label"):
        if len(group) < needed:
            raise ValueError(
                f"Class {label} has only {len(group)} examples, but {needed} are required."
            )

        group = group.sample(n=needed, random_state=seed).reset_index(drop=True)

        train_parts.append(group.iloc[:train_per_class])
        dev_parts.append(group.iloc[train_per_class:train_per_class + dev_per_class])

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dev_df = pd.concat(dev_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, dev_df


def sample_test(test_full: pd.DataFrame, test_per_class: int, seed: int) -> pd.DataFrame:
    parts = []

    for label, group in test_full.groupby("label"):
        n = min(test_per_class, len(group))
        parts.append(group.sample(n=n, random_state=seed))

    test_df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return test_df


def print_stats(name: str, df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Total: {len(df)}")
    print("Label counts:")
    print(df["label_name"].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_per_class", type=int, default=3000)
    parser.add_argument("--dev_per_class", type=int, default=500)
    parser.add_argument("--test_per_class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Preparing Yahoo Answers Topics")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Output dir:  {OUT_DIR}")
    print(f"train_per_class: {args.train_per_class}")
    print(f"dev_per_class:   {args.dev_per_class}")
    print(f"test_per_class:  {args.test_per_class}")

    ds = load_yahoo_from_hf()

    print("\nDataset splits:")
    print(ds)

    train_full = convert_split_to_df(ds["train"], "train")
    test_full = convert_split_to_df(ds["test"], "test")

    print("\nRaw converted sizes:")
    print(f"train_full: {len(train_full)}")
    print(f"test_full:  {len(test_full)}")

    train_df, dev_df = split_train_dev(
        train_full=train_full,
        train_per_class=args.train_per_class,
        dev_per_class=args.dev_per_class,
        seed=args.seed,
    )

    test_df = sample_test(
        test_full=test_full,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )

    print_stats("TRAIN", train_df)
    print_stats("DEV", dev_df)
    print_stats("TEST", test_df)

    train_df.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(OUT_DIR / "dev.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    label_map = {str(k): v for k, v in YAHOO_LABELS.items()}

    with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    dataset_info = {
        "dataset": "yahoo",
        "num_labels": 10,
        "label_map": label_map,
        "seed": args.seed,
        "train_per_class": args.train_per_class,
        "dev_per_class": args.dev_per_class,
        "test_per_class": args.test_per_class,
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "test_size": len(test_df),
    }

    with open(OUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print("\nSaved files:")
    print(OUT_DIR / "train.csv")
    print(OUT_DIR / "dev.csv")
    print(OUT_DIR / "test.csv")
    print(OUT_DIR / "label_map.json")
    print(OUT_DIR / "dataset_info.json")

    print("\nDone.")


if __name__ == "__main__":
    main()