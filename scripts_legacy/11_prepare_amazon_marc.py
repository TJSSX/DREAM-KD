# -*- coding: utf-8 -*-
"""
11_prepare_amazon_marc.py

Prepare Multilingual Amazon Reviews Corpus (MARC) English subset
for DREAM-KD experiments.

Task:
    5-class star rating prediction

Labels:
    0 -> 1 star
    1 -> 2 stars
    2 -> 3 stars
    3 -> 4 stars
    4 -> 5 stars

Default fast setting:
    train_per_star = 5000
    dev_per_star   = 1000
    test_per_star  = 1000

Total:
    train = 25,000
    dev   = 5,000
    test  = 5,000

Output:
    data_processed/amazon_marc/
        train.csv
        dev.csv
        test.csv
        label_map.json
        dataset_info.json

Usage:
    python scripts/11_prepare_amazon_marc.py
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
OUT_DIR = PROJECT_DIR / "data_processed" / "amazon_marc"

SEED = 42


LABEL_MAP = {
    0: "1 star",
    1: "2 stars",
    2: "3 stars",
    3: "4 stars",
    4: "5 stars",
}


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_stars(stars) -> int:
    """
    Convert original star rating 1..5 into label 0..4.
    """
    stars = int(stars)

    if 1 <= stars <= 5:
        return stars - 1

    if 0 <= stars <= 4:
        return stars

    raise ValueError(f"Unexpected stars value: {stars}")


def load_marc_en():
    """
    Load English MARC dataset from Hugging Face.
    Common dataset name:
        amazon_reviews_multi, config='en'
    """
    candidates = [
        ("amazon_reviews_multi", "en"),
        ("SetFit/amazon_reviews_multi_en", None),
    ]

    last_error = None

    for dataset_name, config_name in candidates:
        try:
            print(f"Trying to load dataset: {dataset_name}, config={config_name}")
            if config_name is None:
                ds = load_dataset(dataset_name)
            else:
                ds = load_dataset(dataset_name, config_name)
            print(f"Loaded dataset: {dataset_name}")
            return ds
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            last_error = e

    raise RuntimeError(f"Failed to load MARC English dataset. Last error: {last_error}")


def get_split(ds, preferred_names):
    for name in preferred_names:
        if name in ds:
            return ds[name], name
    raise KeyError(f"None of split names {preferred_names} found. Available splits: {list(ds.keys())}")


def convert_split_to_df(ds_split, split_name: str) -> pd.DataFrame:
    rows = []
    columns = set(ds_split.column_names)

    print(f"\nColumns in {split_name}: {ds_split.column_names}")
    print(f"First example in {split_name}: {ds_split[0]}")

    for item in ds_split:
        # MARC usually has "stars"; SetFit versions may have "label"
        if "stars" in columns:
            label = normalize_stars(item["stars"])
        elif "label" in columns:
            label = normalize_stars(item["label"])
        else:
            raise KeyError(f"Cannot find stars/label column. Available columns: {columns}")

        parts = []

        # Common MARC columns
        for col in [
            "review_title",
            "review_body",
            "text",
            "title",
            "content",
        ]:
            if col in columns:
                value = clean_text(item.get(col, ""))
                if value:
                    parts.append(value)

        if len(parts) == 0:
            raise ValueError(f"Cannot find text columns. Available columns: {columns}")

        text = " ".join(parts)
        text = clean_text(text)

        if len(text.split()) < 5:
            continue

        product_category = item.get("product_category", "unknown") if "product_category" in columns else "unknown"

        rows.append(
            {
                "text": text,
                "label": label,
                "label_name": LABEL_MAP[label],
                "domain": str(product_category),
                "source_split": split_name,
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return df


def sample_per_class(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    parts = []

    for label, group in df.groupby("label"):
        if len(group) < n_per_class:
            print(f"Warning: label={label} has only {len(group)} examples, using all.")
            n = len(group)
        else:
            n = n_per_class

        parts.append(group.sample(n=n, random_state=seed))

    out = pd.concat(parts, axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def print_stats(name: str, df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Total: {len(df)}")
    print("Label counts:")
    print(df["label_name"].value_counts().sort_index())

    if "domain" in df.columns:
        print("\nTop domains:")
        print(df["domain"].value_counts().head(15))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_per_star", type=int, default=5000)
    parser.add_argument("--dev_per_star", type=int, default=1000)
    parser.add_argument("--test_per_star", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Preparing Amazon MARC English")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Output dir:  {OUT_DIR}")
    print(f"train_per_star: {args.train_per_star}")
    print(f"dev_per_star:   {args.dev_per_star}")
    print(f"test_per_star:  {args.test_per_star}")

    ds = load_marc_en()

    print("\nDataset splits:")
    print(ds)

    train_split, train_name = get_split(ds, ["train"])
    dev_split, dev_name = get_split(ds, ["validation", "dev"])
    test_split, test_name = get_split(ds, ["test"])

    train_full = convert_split_to_df(train_split, train_name)
    dev_full = convert_split_to_df(dev_split, dev_name)
    test_full = convert_split_to_df(test_split, test_name)

    print("\nRaw converted sizes:")
    print(f"train_full: {len(train_full)}")
    print(f"dev_full:   {len(dev_full)}")
    print(f"test_full:  {len(test_full)}")

    train_df = sample_per_class(train_full, args.train_per_star, args.seed)
    dev_df = sample_per_class(dev_full, args.dev_per_star, args.seed)
    test_df = sample_per_class(test_full, args.test_per_star, args.seed)

    print_stats("TRAIN", train_df)
    print_stats("DEV", dev_df)
    print_stats("TEST", test_df)

    train_df.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(OUT_DIR / "dev.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    label_map = {str(k): v for k, v in LABEL_MAP.items()}

    with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    dataset_info = {
        "dataset": "amazon_marc",
        "language": "en",
        "task": "5-class star rating prediction",
        "num_labels": 5,
        "label_map": label_map,
        "seed": args.seed,
        "train_per_star": args.train_per_star,
        "dev_per_star": args.dev_per_star,
        "test_per_star": args.test_per_star,
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