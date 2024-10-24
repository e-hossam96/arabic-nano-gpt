"""
Training nano gpt2 based model on arabic wikipedia 2023 dump.
"""

import os
import wandb
import torch
import pathlib
import argparse
from rich import print
from dotenv import load_dotenv
from collections import defaultdict

# huggingface
import huggingface_hub
from datasets import load_dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argument Parser for LM training.")
    # basic arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for model training and data splitting. (default: 42)",
    )
    # data arguments
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
    # model arguments
    parser.add_argument(
        "--base_model_ckpt",
        type=str,
        default="openai-community/gpt2",
        help="Base model check point. (default: 'openai-community/gpt2')",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=256,
        help="Embedding dimension. (default: 256)",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=4,
        help="Number of attention heads. (default: 4)",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=4,
        help="Number of attention blocks. (default: 4)",
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
    # training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=60,
        help="Number of training epcohs. (default: 60)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size. (default: 32)",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps. (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. (default: 1e-3)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-5,
        help="Weight decay. (default: 1e-5)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.01,
        help="Warmup ratio. (default: 0.01)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluation steps. (default: 100)",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
        help="Logging steps. (default: 100)",
    )
    # wandb argument
    parser.add_argument(
        "--project_name",
        type=str,
        default="Arabic-Nano-GPT",
        help="Project name on W&B. (Default: 'Arabic-Nano-GPT')",
    )
    parser.add_argument(
        "--job_type",
        type=str,
        default="LM-Modeling",
        help="Project phase for W&B grouping. (Default: 'LM-Modeling')",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"{{project_name}}",
        help="Run name of current training. (Default: '{project_name}')",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Notes for current run. (Default: '')",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Notes for current run. (Default: '')",
    )
    parser.add_argument(
        "--config",
        type=dict,
        default=defaultdict(dict),
        help="Extra configuration parameters. (Default: 'defaultdict(dict)')",
    )
    args = parser.parse_args()
    # Handle dynamic default for `processed_data_file_path` based on `data_ckpt`
    if args.processed_data_file_path == pathlib.Path(f"data/{{data_ckpt}}.csv"):
        args.processed_data_file_path = pathlib.Path(f"data/{args.data_ckpt}.csv")
    # Handle dynamic default for `target_model_path` based on `model_name`
    if args.target_model_path == pathlib.Path(f"models/{{model_name}}"):
        args.target_model_path = pathlib.Path(f"models/{args.model_name}")
    # Handle dynamic default for `run_name` based on `project_name`
    if args.run_name == f"{{project_name}}":
        args.run_name = args.project_name
    return args


def tokenize_batch(batch: dict, tokenizer: PreTrainedTokenizerFast) -> dict:
    outputs = tokenizer(
        batch["text"],
        truncation=True,
        return_overflowing_tokens=True,
    )
    return {"input_ids": outputs["input_ids"]}


def build_model(
    tokenizer: PreTrainedTokenizerFast, args: argparse.Namespace
) -> torch.nn.Module:
    model_config = AutoConfig.from_pretrained(
        args.base_model_ckpt,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_positions=tokenizer.model_max_length,
        n_ctx=tokenizer.model_max_length,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
    )
    model = AutoModelForCausalLM.from_config(model_config)
    return model


def main(args: argparse.Namespace) -> None:
    # setup the tokens
    load_dotenv()
    wandb.login()
    huggingface_hub.login(
        token=os.getenv("HF_TOKEN"),
        add_to_git_credential=True,
        write_permission=True,
    )  # HF
    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    # loading the dataset
    dataset = load_dataset(
        "csv",
        data_files=[str(args.processed_data_file_path)],
        split=f"{args.split_name}[:{args.split_size}]",
    )
    print(f"Loaded dataset\n{dataset}")
    print(f"Sample record\n{dataset[0]}")
    # splitting dataset into train/valid/test
    dataset = dataset.train_test_split(train_size=0.9, seed=args.seed)
    test_data = dataset["test"].train_test_split(test_size=0.5, seed=args.seed)
    dataset["valid"] = test_data["train"]
    dataset["test"] = test_data["test"]
    del test_data
    print(f"Data splits\n{dataset}")
    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    print(f"Tokenizer\n{tokenizer}")
    # tokenization
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )
    print(f"Tokenized data\n{tokenized_dataset}")
    # defining the model
    model = build_model(tokenizer, args)
    print("Model Size in MBs:", model.get_memory_footprint() / 1_000_000)
    print("Num Model Params:", model.num_parameters() / 1_000_000, "M")
    print(f"Molel config\n{model.config}")
    print(model)
    # defining training args
    total_steps = (
        len(tokenized_dataset["train"])
        * args.num_epochs
        // (args.batch_size * args.accum_steps)
    )
    print(f"Total training steps: {total_steps}")
    print(f"Effective training batch size: {args.batch_size * args.accum_steps}")
    training_args = TrainingArguments(
        output_dir=args.target_model_path,
        run_name=args.run_name,
        report_to="wandb",
        save_strategy="no",
        eval_strategy="steps",
        gradient_accumulation_steps=args.accum_steps,
        overwrite_output_dir=True,
        data_seed=args.seed,
        seed=args.seed,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_ratio=args.warmup,
        eval_steps=args.eval_steps,
        logging_steps=args.log_steps,
        log_level="error",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    run = wandb.init(
        project=args.project_name,
        job_type=args.job_type,
        name=args.run_name,
        notes=args.notes,
        tags=args.tags.split(","),
        config=args.config,
    )
    _ = trainer.train()
    print(trainer.evaluate(tokenized_dataset["test"], metric_key_prefix="test"))
    run.finish()
    trainer.save_model(args.target_model_path)
    _ = trainer.push_to_hub()


if __name__ == "__main__":
    args = get_args()
    main(args)
