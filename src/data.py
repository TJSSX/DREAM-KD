import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from datasets import load_dataset


def load_yaml_config(config_path: str) -> Dict:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> Path:
    """Create a directory if it does not exist."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_text(text: Optional[str]) -> str:
    """Basic text normalization."""
    if text is None:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def load_hf_dataset_with_optional_config(hf_name: str, hf_config: Optional[str]):
    """Load a Hugging Face dataset with or without a config name."""
    if hf_config is None or str(hf_config).lower() == "null":
        return load_dataset(hf_name)
    return load_dataset(hf_name, hf_config)


def get_split(raw_dataset, split_name: Optional[str]):
    """Safely get a dataset split."""
    if split_name is None or str(split_name).lower() == "null":
        return None
    if split_name not in raw_dataset:
        return None
    return raw_dataset[split_name]


def convert_standard_split(
    hf_split,
    text_column: str,
    label_column: str,
    label_offset: int = 0,
    min_text_length: int = 5,
) -> pd.DataFrame:
    """
    Convert a Hugging Face split to a standard DataFrame.

    Output columns:
        text: str
        label: int, 0-based class label
    """
    rows = []

    for example in hf_split:
        if text_column not in example:
            raise KeyError(
                f"Text column '{text_column}' not found. "
                f"Available columns: {list(example.keys())}"
            )

        if label_column not in example:
            raise KeyError(
                f"Label column '{label_column}' not found. "
                f"Available columns: {list(example.keys())}"
            )

        text = normalize_text(example.get(text_column))
        if len(text) < min_text_length:
            continue

        raw_label = example.get(label_column)
        if raw_label is None:
            continue

        label = int(raw_label) - int(label_offset)

        rows.append(
            {
                "text": text,
                "label": label,
            }
        )

    return pd.DataFrame(rows)


def sample_dataframe(
    df: pd.DataFrame,
    split_name: str,
    sampling_cfg: Dict[str, Any],
    num_labels: int,
) -> pd.DataFrame:
    """
    Sample a standardized DataFrame.

    Supported strategies:
        full:
            Use all examples.
        random:
            Randomly sample N examples from the split, preserving natural imbalance.
        balanced:
            Stratified balanced sampling, with the same number of examples per class.
    """
    strategy = sampling_cfg.get("strategy", "full")
    seed = int(sampling_cfg.get("seed", 42))

    if strategy is None:
        strategy = "full"

    strategy = str(strategy).lower()

    if strategy == "full":
        sampled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return sampled

    if strategy == "random":
        size_key = f"{split_name}_size"
        requested_size = sampling_cfg.get(size_key)

        if requested_size is None or str(requested_size).lower() == "null":
            sampled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            return sampled

        requested_size = int(requested_size)
        actual_size = min(requested_size, len(df))

        sampled = df.sample(n=actual_size, random_state=seed).reset_index(drop=True)
        return sampled

    if strategy == "balanced":
        per_class_key = f"{split_name}_per_class"
        requested_per_class = sampling_cfg.get(per_class_key)

        if requested_per_class is None or str(requested_per_class).lower() == "null":
            raise ValueError(
                f"sampling.strategy='balanced' requires '{per_class_key}' in config."
            )

        requested_per_class = int(requested_per_class)

        sampled_parts = []

        for label in range(num_labels):
            class_df = df[df["label"] == label]

            if class_df.empty:
                raise ValueError(
                    f"No examples found for label={label} in {split_name} split."
                )

            actual_n = min(requested_per_class, len(class_df))

            if actual_n < requested_per_class:
                print(
                    f"[Warning] {split_name} label={label}: "
                    f"requested {requested_per_class}, but only {len(class_df)} available. "
                    f"Using {actual_n}."
                )

            sampled_class_df = class_df.sample(
                n=actual_n,
                random_state=seed + label,
            )

            sampled_parts.append(sampled_class_df)

        sampled = pd.concat(sampled_parts, axis=0)
        sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        return sampled

    raise ValueError(
        f"Unsupported sampling strategy: {strategy}. "
        f"Choose from ['full', 'random', 'balanced']."
    )


def validate_label_range(df: pd.DataFrame, split_name: str, num_labels: int) -> None:
    """Validate labels are within [0, num_labels - 1]."""
    if df.empty:
        raise ValueError(f"{split_name} split is empty.")

    min_label = int(df["label"].min())
    max_label = int(df["label"].max())

    if min_label < 0 or max_label >= num_labels:
        raise ValueError(
            f"Invalid label range in {split_name}: "
            f"min={min_label}, max={max_label}, expected [0, {num_labels - 1}]"
        )


def print_label_distribution(df: pd.DataFrame, split_name: str, num_labels: int) -> Dict[str, Any]:
    """Print and return label distribution for one split."""
    counts = df["label"].value_counts().sort_index()

    print("=" * 80)
    print(f"{split_name.upper()} split")
    print(f"Total examples: {len(df):,}")
    print("-" * 80)

    distribution = {}

    for label in range(num_labels):
        count = int(counts.get(label, 0))
        ratio = count / len(df) if len(df) > 0 else 0.0
        distribution[str(label)] = {
            "count": count,
            "ratio": ratio,
        }
        print(f"label {label}: {count:,} ({ratio:.4%})")

    return {
        "num_examples": int(len(df)),
        "label_distribution": distribution,
    }


def build_dataset_stats(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_labels: int,
    selected_source: str,
    sampling_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Build dataset statistics and print label distributions."""
    stats = {
        "selected_source": selected_source,
        "sampling": sampling_cfg,
        "splits": {},
    }

    stats["splits"]["train"] = print_label_distribution(train_df, "train", num_labels)
    stats["splits"]["dev"] = print_label_distribution(dev_df, "dev", num_labels)
    stats["splits"]["test"] = print_label_distribution(test_df, "test", num_labels)

    return stats


def try_load_amazon_candidate(candidate: Dict[str, Any]):
    """Try to load one Amazon MARC candidate dataset."""
    hf_name = candidate["hf_name"]
    hf_config = candidate.get("hf_config")

    print("=" * 80)
    print(f"Trying Hugging Face dataset: {hf_name}, config: {hf_config}")
    print("=" * 80)

    raw_dataset = load_hf_dataset_with_optional_config(hf_name, hf_config)

    print(f"Successfully loaded dataset: {hf_name}")
    print(f"Available splits: {list(raw_dataset.keys())}")

    return raw_dataset


def prepare_amazon_marc(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Prepare Amazon MARC English 5-class review classification data.

    Supports multiple candidate Hugging Face sources. The loader tries candidates
    in order and uses the first one that works.

    Expected output:
        train_df, dev_df, test_df with columns: text, label
        dataset_stats
    """
    dataset_cfg = config["dataset"]
    split_cfg = config["split"]
    processing_cfg = config.get("processing", {})
    sampling_cfg = config.get("sampling", {"strategy": "full"})

    num_labels = int(dataset_cfg["num_labels"])

    candidates: List[Dict[str, Any]] = dataset_cfg.get("candidates", [])

    # Backward compatibility: allow single hf_name / hf_config config.
    if not candidates:
        candidates = [
            {
                "hf_name": dataset_cfg["hf_name"],
                "hf_config": dataset_cfg.get("hf_config"),
                "text_column": dataset_cfg["text_column"],
                "label_column": dataset_cfg["label_column"],
                "label_offset": dataset_cfg.get("label_offset", 0),
            }
        ]

    min_text_length = int(processing_cfg.get("min_text_length", 5))

    train_name = split_cfg.get("train_name", "train")
    dev_name = split_cfg.get("dev_name", None)
    test_name = split_cfg.get("test_name", "test")

    dev_size = float(split_cfg.get("dev_size", 0.1))
    seed = int(split_cfg.get("seed", 42))

    last_error = None

    for candidate in candidates:
        try:
            raw_dataset = try_load_amazon_candidate(candidate)

            text_column = candidate["text_column"]
            label_column = candidate["label_column"]
            label_offset = int(candidate.get("label_offset", 0))

            if train_name not in raw_dataset:
                raise ValueError(
                    f"Train split '{train_name}' not found. "
                    f"Available splits: {list(raw_dataset.keys())}"
                )

            if test_name not in raw_dataset:
                raise ValueError(
                    f"Test split '{test_name}' not found. "
                    f"Available splits: {list(raw_dataset.keys())}"
                )

            raw_train = raw_dataset[train_name]
            raw_test = raw_dataset[test_name]

            raw_dev = get_split(raw_dataset, dev_name)

            if raw_dev is not None:
                print(f"Using existing dev split: {dev_name}")
            else:
                print(
                    f"No valid dev split found. Creating dev from train "
                    f"with dev_size={dev_size}, seed={seed}."
                )
                train_dev = raw_train.train_test_split(test_size=dev_size, seed=seed)
                raw_train = train_dev["train"]
                raw_dev = train_dev["test"]

            print("Converting Hugging Face splits to standard DataFrames...")
            train_df = convert_standard_split(
                raw_train,
                text_column=text_column,
                label_column=label_column,
                label_offset=label_offset,
                min_text_length=min_text_length,
            )

            dev_df = convert_standard_split(
                raw_dev,
                text_column=text_column,
                label_column=label_column,
                label_offset=label_offset,
                min_text_length=min_text_length,
            )

            test_df = convert_standard_split(
                raw_test,
                text_column=text_column,
                label_column=label_column,
                label_offset=label_offset,
                min_text_length=min_text_length,
            )

            validate_label_range(train_df, "train_raw", num_labels=num_labels)
            validate_label_range(dev_df, "dev_raw", num_labels=num_labels)
            validate_label_range(test_df, "test_raw", num_labels=num_labels)

            print("=" * 80)
            print("Raw split label distributions before sampling")
            print("=" * 80)
            print_label_distribution(train_df, "train_raw", num_labels)
            print_label_distribution(dev_df, "dev_raw", num_labels)
            print_label_distribution(test_df, "test_raw", num_labels)

            print("=" * 80)
            print(f"Applying sampling strategy: {sampling_cfg.get('strategy', 'full')}")
            print("=" * 80)

            train_df = sample_dataframe(train_df, "train", sampling_cfg, num_labels)
            dev_df = sample_dataframe(dev_df, "dev", sampling_cfg, num_labels)
            test_df = sample_dataframe(test_df, "test", sampling_cfg, num_labels)

            validate_label_range(train_df, "train", num_labels=num_labels)
            validate_label_range(dev_df, "dev", num_labels=num_labels)
            validate_label_range(test_df, "test", num_labels=num_labels)

            stats = build_dataset_stats(
                train_df=train_df,
                dev_df=dev_df,
                test_df=test_df,
                num_labels=num_labels,
                selected_source=candidate["hf_name"],
                sampling_cfg=sampling_cfg,
            )

            print("=" * 80)
            print("Amazon MARC data preparation succeeded.")
            print(f"Selected source: {candidate['hf_name']}")
            print("=" * 80)

            return train_df, dev_df, test_df, stats

        except Exception as e:
            last_error = e
            print("=" * 80)
            print(f"Failed to load candidate: {candidate.get('hf_name')}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("Trying next candidate if available...")
            print("=" * 80)

    raise RuntimeError(
        "All Amazon MARC candidate datasets failed. "
        f"Last error: {repr(last_error)}"
    )


def save_standard_dataset(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    label_mapping: Dict[str, int],
    dataset_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Save train/dev/test CSV files, label mapping, and dataset statistics."""
    out_dir = ensure_dir(output_dir)

    train_path = out_dir / "train.csv"
    dev_path = out_dir / "dev.csv"
    test_path = out_dir / "test.csv"
    label_path = out_dir / "label_mapping.json"
    stats_path = out_dir / "dataset_stats.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    dev_df.to_csv(dev_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    with label_path.open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    if dataset_stats is not None:
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(dataset_stats, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Saved standardized dataset")
    print("=" * 80)
    print(f"Saved train: {train_path} ({len(train_df):,} rows)")
    print(f"Saved dev:   {dev_path} ({len(dev_df):,} rows)")
    print(f"Saved test:  {test_path} ({len(test_df):,} rows)")
    print(f"Saved label mapping: {label_path}")

    if dataset_stats is not None:
        print(f"Saved dataset stats: {stats_path}")


def get_label_mapping_for_amazon_marc() -> Dict[str, int]:
    """Amazon MARC 5-star label mapping."""
    return {
        "1_star": 0,
        "2_star": 1,
        "3_star": 2,
        "4_star": 3,
        "5_star": 4,
    }