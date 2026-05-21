# -*- coding: utf-8 -*-
"""
01_prepare_data.py

Prepare Amazon Multi-Domain Sentiment Dataset.

Input:
    data_raw/processed_acl/
        books/
        dvd/
        electronics/
        kitchen/

Output:
    data_processed/all.csv
    data_processed/train.csv
    data_processed/dev.csv
    data_processed/test_in.csv
    data_processed/test_ood.csv
"""

import re
import random
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"D:\Pycharm_Work\DREAM_KD")

RAW_DIR = PROJECT_DIR / "data_raw" / "processed_acl"
OUT_DIR = PROJECT_DIR / "data_processed"

DOMAINS = ["books", "dvd", "electronics", "kitchen"]

TRAIN_DOMAINS = ["books", "dvd", "electronics"]
OOD_DOMAIN = "kitchen"

SEED = 42

# Debug size. processed_acl usually has 1000 pos + 1000 neg per domain.
N_TRAIN_PER_LABEL_PER_DOMAIN = 650
N_DEV_PER_LABEL_PER_DOMAIN = 150
N_TEST_IN_PER_LABEL_PER_DOMAIN = 150
N_OOD_PER_LABEL = 950


def clean_review_line(line: str) -> str:
    """
    processed_acl files are usually sparse token-count format:
        word:count word:count ...
    Convert into plain text.
    """
    line = line.strip()
    if not line:
        return ""

    line = re.sub(r"#label#:[^\s]+", " ", line)
    line = re.sub(r"#label#", " ", line)

    tokens = []

    for item in line.split():
        if item.startswith("#"):
            continue

        if ":" in item:
            word, count = item.rsplit(":", 1)
            word = word.strip()
            if not word:
                continue

            try:
                count_int = int(float(count))
                count_int = max(1, min(count_int, 5))
            except ValueError:
                count_int = 1

            tokens.extend([word] * count_int)
        else:
            tokens.append(item)

    text = " ".join(tokens)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_review_file(file_path: Path, label: int, domain: str) -> list[dict]:
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_id, line in enumerate(f):
            text = clean_review_line(line)

            if len(text.split()) < 3:
                continue

            rows.append(
                {
                    "text": text,
                    "label": label,
                    "domain": domain,
                    "source_file": file_path.name,
                    "line_id": line_id,
                }
            )

    return rows


def load_all_data() -> pd.DataFrame:
    all_rows = []

    for domain in DOMAINS:
        domain_dir = RAW_DIR / domain
        pos_file = domain_dir / "positive.review"
        neg_file = domain_dir / "negative.review"

        if not pos_file.exists():
            raise FileNotFoundError(f"Cannot find {pos_file}")
        if not neg_file.exists():
            raise FileNotFoundError(f"Cannot find {neg_file}")

        print(f"Reading domain: {domain}")

        pos_rows = read_review_file(pos_file, label=1, domain=domain)
        neg_rows = read_review_file(neg_file, label=0, domain=domain)

        print(f"  positive: {len(pos_rows)}")
        print(f"  negative: {len(neg_rows)}")

        all_rows.extend(pos_rows)
        all_rows.extend(neg_rows)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["text", "label", "domain"]).reset_index(drop=True)
    return df


def sample_balanced(df_domain: pd.DataFrame, n_per_label: int, seed: int) -> pd.DataFrame:
    parts = []

    for label in [0, 1]:
        part = df_domain[df_domain["label"] == label]
        n = min(n_per_label, len(part))
        parts.append(part.sample(n=n, random_state=seed))

    out = pd.concat(parts, axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def make_splits(df: pd.DataFrame):
    train_parts = []
    dev_parts = []
    test_in_parts = []

    for domain in TRAIN_DOMAINS:
        df_d = df[df["domain"] == domain].copy()

        needed = (
            N_TRAIN_PER_LABEL_PER_DOMAIN
            + N_DEV_PER_LABEL_PER_DOMAIN
            + N_TEST_IN_PER_LABEL_PER_DOMAIN
        )

        df_d = sample_balanced(df_d, needed, SEED)

        domain_train = []
        domain_dev = []
        domain_test = []

        for label in [0, 1]:
            part = df_d[df_d["label"] == label].sample(frac=1.0, random_state=SEED)
            part = part.reset_index(drop=True)

            train_end = N_TRAIN_PER_LABEL_PER_DOMAIN
            dev_end = train_end + N_DEV_PER_LABEL_PER_DOMAIN
            test_end = dev_end + N_TEST_IN_PER_LABEL_PER_DOMAIN

            domain_train.append(part.iloc[:train_end])
            domain_dev.append(part.iloc[train_end:dev_end])
            domain_test.append(part.iloc[dev_end:test_end])

        train_parts.append(pd.concat(domain_train, axis=0))
        dev_parts.append(pd.concat(domain_dev, axis=0))
        test_in_parts.append(pd.concat(domain_test, axis=0))

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    dev_df = pd.concat(dev_parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_in_df = pd.concat(test_in_parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df_ood = df[df["domain"] == OOD_DOMAIN].copy()
    test_ood_df = sample_balanced(df_ood, N_OOD_PER_LABEL, SEED)

    return train_df, dev_df, test_in_df, test_ood_df


def print_stats(name: str, df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Total: {len(df)}")
    print("\nDomain counts:")
    print(df["domain"].value_counts())
    print("\nLabel counts:")
    print(df["label"].value_counts().sort_index())


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project dir: {PROJECT_DIR}")
    print(f"Raw dir: {RAW_DIR}")
    print(f"Output dir: {OUT_DIR}")

    df = load_all_data()

    print("\nAll data loaded.")
    print(df.head())
    print("\nDomain-label table:")
    print(pd.crosstab(df["domain"], df["label"]))

    all_path = OUT_DIR / "all.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved all.csv to: {all_path}")

    train_df, dev_df, test_in_df, test_ood_df = make_splits(df)

    print_stats("TRAIN", train_df)
    print_stats("DEV", dev_df)
    print_stats("TEST_IN", test_in_df)
    print_stats("TEST_OOD", test_ood_df)

    train_df.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(OUT_DIR / "dev.csv", index=False, encoding="utf-8-sig")
    test_in_df.to_csv(OUT_DIR / "test_in.csv", index=False, encoding="utf-8-sig")
    test_ood_df.to_csv(OUT_DIR / "test_ood.csv", index=False, encoding="utf-8-sig")

    print("\nSaved processed data:")
    print(OUT_DIR / "train.csv")
    print(OUT_DIR / "dev.csv")
    print(OUT_DIR / "test_in.csv")
    print(OUT_DIR / "test_ood.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()