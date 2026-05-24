import argparse
import sys
from pathlib import Path

# Make project root importable when running as:
# python scripts/prepare_data.py --dataset amazon_marc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    get_label_mapping_for_amazon_marc,
    load_yaml_config,
    prepare_amazon_marc,
    save_standard_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets for DREAM-KD.")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["amazon_marc"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. If not provided, use configs/{dataset}.yaml.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config is None:
        config_path = PROJECT_ROOT / "configs" / f"{args.dataset}.yaml"
    else:
        config_path = Path(args.config)

    print(f"Using config: {config_path}")
    config = load_yaml_config(str(config_path))

    if args.dataset == "amazon_marc":
        train_df, dev_df, test_df, dataset_stats = prepare_amazon_marc(config)
        label_mapping = get_label_mapping_for_amazon_marc()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    output_dir = PROJECT_ROOT / config["output"]["dir"]

    save_standard_dataset(
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        output_dir=str(output_dir),
        label_mapping=label_mapping,
        dataset_stats=dataset_stats,
    )


if __name__ == "__main__":
    main()