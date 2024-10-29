"""
Training and saving the gpt2-based tokenizer on preprocessed dataset.
"""

import pathlib
import argparse
from rich import print
from datasets import load_dataset
from transformers import AutoTokenizer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Argument Parser for tokenizer training configurations."
    )
    # Define arguments
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="openai-community/gpt2",
        help="Checkpoint for the base model. (default: 'openai-community/gpt2')",
    )
    parser.add_argument(
        "--data_ckpt",
        type=str,
        default="wikimedia/wikipedia",
        help="Checkpoint for the original dataset. (default: 'wikimedia/wikipedia')",
    )
    parser.add_argument(
        "--processed_data_file_path",
        type=pathlib.Path,
        default=pathlib.Path(f"data/{{data_ckpt}}.csv"),
        help="Path to the processed data CSV file. (Default: 'data/{data_ckpt}.csv')",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=1024,
        help="Model maximum context length. (default: 1024)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8192,
        help="Tokenizer vocab size. (default: 8192)",
    )
    parser.add_argument(
        "--iterator_batch_size",
        type=int,
        default=1024,
        help="Tokenizer training iterator batch size. (default: 1024)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="arabic-nano-gpt",
        help="To be trained model name. (default: 'arabic-nano-gpt')",
    )
    parser.add_argument(
        "--target_model_path",
        type=pathlib.Path,
        default=pathlib.Path(f"models/{{model_name}}"),
        help="Path to the end model. (Default: 'models/{model_name}')",
    )
    # Parse the arguments
    args = parser.parse_args()
    # Handle dynamic default for `processed_data_file_path` based on `data_ckpt`
    if args.processed_data_file_path == pathlib.Path(f"data/{{data_ckpt}}.csv"):
        args.processed_data_file_path = pathlib.Path(f"data/{args.data_ckpt}.csv")
    # Handle dynamic default for `target_model_path` based on `model_name`
    if args.target_model_path == pathlib.Path(f"models/{{model_name}}"):
        args.target_model_path = pathlib.Path(f"models/{args.model_name}")
    return args


def main(args: argparse.Namespace) -> None:
    # loading the dataset
    dataset = load_dataset("csv", data_files=[str(args.processed_data_file_path)])
    print(f"Loaded dataset\n{dataset}")
    print(f"Sample record\n{dataset['train'][0]}")
    # loading the base model tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.model_ckpt, model_max_length=args.model_max_length
    )
    print(f"Base tokenizer\n{base_tokenizer}")
    # training tokenizer

    def get_dataset_iterator(data, batch_size: int = 1024):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]["text"]

    dataset_iterator = get_dataset_iterator(
        dataset["train"], batch_size=args.iterator_batch_size
    )
    tokenizer = base_tokenizer.train_new_from_iterator(
        dataset_iterator, vocab_size=args.vocab_size, length=len(dataset)
    )
    print(f"New tokenizer\n{tokenizer}")
    args.target_model_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.target_model_path)
    print(f"Saved tokenizer to path: {args.target_model_path}")


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
