# -*- coding: utf-8 -*-
"""
09_prepare_dbpedia.py

Prepare DBpedia-14 for DREAM-KD experiments.

Output:
    data_processed/dbpedia/
        train.csv
        dev.csv
        test.csv
        label_map.json
        dataset_info.json

Default full sampled setting:
    train_per_class = 2000
    dev_per_class   = 300
    test_per_class  = 1000

Total:
    train = 28,000
    dev   = 4,200
    test  = 14,000

Usage:
    python scripts/09_prepare_dbpedia.py

Debug usage:
    python scripts/09_prepare_dbpedia.py --train_per_class 500 --dev_per_class 100 --test_per_class 300
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
OUT_DIR = PROJECT_DIR / "data_processed" / "dbpedia"

SEED = 42


DBPEDIA_LABELS = {
    0: "Company",
    1: "EducationalInstitution",
    2: "Artist",
    3: "Athlete",
    4: "OfficeHolder",
    5: "MeanOfTransportation",
    6: "Building",
    7: "NaturalPlace",
    8: "Village",
    9: "Animal",
    10: "Plant",
    11: "Album",
    12: "Film",
    13: "WrittenWork",
}


def clean_text(text: str) -> str:
    text = str(text)

    # Basic cleanup
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_label(label):
    """
    Hugging Face DBpedia labels may be 0-based or 1-based depending on source.
    Try to normalize to 0..13.
    """
    label = int(label)

    if 0 <= label <= 13:
        return label

    if 1 <= label <= 14:
        return label - 1

    raise ValueError(f"Unexpected label value: {label}")


def load_dbpedia_from_hf():
    """
    Try several common dataset names to avoid version/source issues.
    """
    candidates = [
        ("dbpedia_14", None),
        ("fancyzhx/dbpedia_14", None),
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

    raise RuntimeError(f"Failed to load DBpedia dataset from all candidates. Last error: {last_error}")


def convert_split_to_df(ds_split, split_name: str) -> pd.DataFrame:
    rows = []

    # Common column names:
    # dbpedia_14 usually has: title, content, label
    # Some versions may have: text, label
    columns = set(ds_split.column_names)

    for item in ds_split:
        label = normalize_label(item["label"])

        if "title" in columns and "content" in columns:
            title = clean_text(item["title"])
            content = clean_text(item["content"])
            text = (title + ". " + content).strip()
        elif "text" in columns:
            text = clean_text(item["text"])
        elif "content" in columns:
            text = clean_text(item["content"])
        else:
            raise ValueError(f"Cannot find text columns in dataset split: {columns}")

        if len(text.split()) < 3:
            continue

        rows.append(
            {
                "text": text,
                "label": label,
                "label_name": DBPEDIA_LABELS[label],
                "domain": "dbpedia",
                "source_split": split_name,
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return df


def sample_per_class(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    parts = []

    for label, group in df.groupby("label"):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=seed))

    out = pd.concat(parts, axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


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


def print_stats(name: str, df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Total: {len(df)}")
    print("Label counts:")
    print(df["label_name"].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_per_class", type=int, default=2000)
    parser.add_argument("--dev_per_class", type=int, default=300)
    parser.add_argument("--test_per_class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Preparing DBpedia-14")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Output dir:  {OUT_DIR}")
    print(f"train_per_class: {args.train_per_class}")
    print(f"dev_per_class:   {args.dev_per_class}")
    print(f"test_per_class:  {args.test_per_class}")

    ds = load_dbpedia_from_hf()

    print("\nDataset splits:")
    print(ds)

    # HF split names are usually "train" and "test"
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

    test_df = sample_per_class(
        df=test_full,
        n_per_class=args.test_per_class,
        seed=args.seed,
    )

    print_stats("TRAIN", train_df)
    print_stats("DEV", dev_df)
    print_stats("TEST", test_df)

    train_df.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(OUT_DIR / "dev.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    label_map = {str(k): v for k, v in DBPEDIA_LABELS.items()}

    with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    dataset_info = {
        "dataset": "dbpedia",
        "num_labels": 14,
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