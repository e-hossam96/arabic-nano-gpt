"""
Loading and preprocessing Arabic Wikipedia Dump from Wikimedia Datasets.
"""

import os
import re
import pathlib
import argparse
import pandas as pd
from rich import print
from datasets import load_dataset
from pyarabic.araby import strip_tashkeel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Argument Parser for processing dataset configurations."
    )
    # Define arguments
    parser.add_argument(
        "--data_ckpt",
        type=str,
        default="wikimedia/wikipedia",
        help="Checkpoint for the dataset. (default: 'wikimedia/wikipedia')",
    )
    parser.add_argument(
        "--sub_data",
        type=str,
        default="20231101.ar",
        help="Sub-dataset name. (default: '20231101.ar')",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Name of the dataset split. (default: 'train')",
    )
    parser.add_argument(
        "--split_size",
        type=str,
        default="",
        help="Size of the dataset split. (default: '')",
    )
    parser.add_argument(
        "--processed_data_file_path",
        type=pathlib.Path,
        default=pathlib.Path(f"data/{{data_ckpt}}.csv"),
        help="Path to the processed data CSV file. (Default: 'data/{data_ckpt}.csv')",
    )
    # Parse the arguments
    args = parser.parse_args()
    # Handle dynamic default for `processed_data_file_path` based on `data_ckpt`
    if args.processed_data_file_path == pathlib.Path(f"data/{{data_ckpt}}.csv"):
        args.processed_data_file_path = pathlib.Path(f"data/{args.data_ckpt}.csv")
    return args


def main(args: argparse.Namespace) -> None:
    # loading dataset
    dataset = load_dataset(
        args.data_ckpt, args.sub_data, split=f"{args.split_name}[:{args.split_size}]"
    )
    print(f"Original Datasets\n{dataset}")
    # splitting into sentences
    dataset = dataset.map(
        lambda x: {"sentences": [s.split("\n\n") for s in x["text"]]},
        batched=True,
        num_proc=os.cpu_count(),
        desc="Sentence Splitting",
    )
    print(dataset)
    # Stripping Tashkeel
    dataset = dataset.map(
        lambda x: {
            "sentences": [
                [strip_tashkeel(s.strip()) for s in art] for art in x["sentences"]
            ]
        },
        batched=True,
        num_proc=os.cpu_count(),
        desc="Stripping Tashkeel",
    )
    print(dataset)
    # adding spaces around punctuations
    dataset = dataset.map(
        lambda x: {
            "sentences": [
                [re.sub(r"([^\w\s])", r" \1 ", s) for s in art]
                for art in x["sentences"]
            ]
        },
        batched=True,
        num_proc=os.cpu_count(),
        desc="Space-Padding Punctuations",
    )
    print(dataset)
    # replacing multiple white spaces with only one
    dataset = dataset.map(
        lambda x: {
            "sentences": [
                [re.sub(r" +", " ", s) for s in art] for art in x["sentences"]
            ]
        },
        batched=True,
        num_proc=os.cpu_count(),
        desc="One-Space Processing",
    )
    print(dataset)
    # calculating sentences lengths
    dataset = dataset.map(
        lambda x: {
            "sentences_length": [[len(s) for s in art] for art in x["sentences"]]
        },
        batched=True,
        num_proc=os.cpu_count(),
        desc="Calc Sents Lengths",
    )
    print(dataset)
    # filtering sentences that are less than 60 characters and more than 1250 characters

    def filter_sentences(batch):
        sentences = []
        sents, lengths = batch["sentences"], batch["sentences_length"]
        for s, l in zip(sents, lengths):
            sentences.append([s_ for s_, l_ in zip(s, l) if 60 <= l_ <= 1250])
        return {
            "sentences": sentences,
            "num_sentences": [len(art) for art in sentences],
        }

    dataset = dataset.map(
        filter_sentences,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["sentences_length"],
        desc="Filtering Sentences",
    )
    print(dataset)
    # filtering articles with no sentences
    dataset = dataset.filter(
        lambda x: [num_sents != 0 for num_sents in x["num_sentences"]],
        batched=True,
        num_proc=os.cpu_count(),
        desc="No-Sents Filtering",
    )
    print(dataset)
    # getting final sentences
    sentences = []
    for art in dataset["sentences"]:
        sentences.extend(art)
    print(f"Total Number of Sentences (Docs): {len(sentences)}")
    args.processed_data_file_path.parent.mkdir(parents=True, exist_ok=True)
    sentences = pd.DataFrame.from_dict({"text": sentences})
    sentences.to_csv(args.processed_data_file_path, index=False)
    print("Finished Preprocessing")


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
