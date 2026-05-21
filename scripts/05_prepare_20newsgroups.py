# -*- coding: utf-8 -*-
"""
05_prepare_20newsgroups.py

Prepare 20 Newsgroups for multi-class DREAM-KD experiments.

Output:
    data_processed_20ng/
        train.csv
        dev.csv
        test.csv
        label_map.json
        dataset_info.json
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(r"D:\Pycharm_Work\DREAM_KD")
OUT_DIR = PROJECT_DIR / "data_processed_20ng"

SEED = 42
DEV_SIZE = 0.15

# Debug mode: set smaller numbers if you want faster local tests.
# For first real run, keep None.
MAX_TRAIN_PER_CLASS = None
MAX_TEST_PER_CLASS = None


def clean_text(text: str) -> str:
    """
    Basic cleaning for 20 Newsgroups raw text.
    Keep it simple and reproducible.
    """
    text = str(text)

    # Remove email headers if any remain
    text = re.sub(r"From:.*?\n", " ", text)
    text = re.sub(r"Subject:.*?\n", " ", text)
    text = re.sub(r"Organization:.*?\n", " ", text)
    text = re.sub(r"Lines:.*?\n", " ", text)

    # Remove quoted email lines
    lines = []
    for line in text.splitlines():
        if line.strip().startswith(">"):
            continue
        lines.append(line)

    text = " ".join(lines)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_df(subset: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load 20NG subset and return dataframe.
    """
    data = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
    )

    texts = [clean_text(t) for t in data.data]
    labels = data.target.astype(int)
    target_names = list(data.target_names)

    df = pd.DataFrame(
        {
            "text": texts,
            "label": labels,
            "label_name": [target_names[i] for i in labels],
            "domain": "20newsgroups",
        }
    )

    # Drop extremely short texts
    df["num_words"] = df["text"].str.split().apply(len)
    df = df[df["num_words"] >= 5].copy()
    df = df.drop(columns=["num_words"])
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    return df, target_names


def sample_per_class(df: pd.DataFrame, max_per_class: int | None, seed: int) -> pd.DataFrame:
    if max_per_class is None:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    parts = []
    for label, group in df.groupby("label"):
        n = min(max_per_class, len(group))
        parts.append(group.sample(n=n, random_state=seed))

    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def print_stats(name: str, df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Total: {len(df)}")
    print("Label counts:")
    print(df["label_name"].value_counts().sort_index())


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Preparing 20 Newsgroups")
    print("=" * 80)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Output dir:  {OUT_DIR}")

    train_full, target_names = build_df("train")
    test_df, target_names_test = build_df("test")

    assert target_names == target_names_test

    train_full = sample_per_class(train_full, MAX_TRAIN_PER_CLASS, SEED)
    test_df = sample_per_class(test_df, MAX_TEST_PER_CLASS, SEED)

    train_df, dev_df = train_test_split(
        train_full,
        test_size=DEV_SIZE,
        random_state=SEED,
        stratify=train_full["label"],
    )

    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print_stats("TRAIN", train_df)
    print_stats("DEV", dev_df)
    print_stats("TEST", test_df)

    train_df.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(OUT_DIR / "dev.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    label_map = {str(i): name for i, name in enumerate(target_names)}
    with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    dataset_info = {
        "dataset": "20newsgroups",
        "num_labels": len(target_names),
        "label_map": label_map,
        "seed": SEED,
        "dev_size": DEV_SIZE,
        "train_size": len(train_df),
        "dev_size_abs": len(dev_df),
        "test_size": len(test_df),
        "max_train_per_class": MAX_TRAIN_PER_CLASS,
        "max_test_per_class": MAX_TEST_PER_CLASS,
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