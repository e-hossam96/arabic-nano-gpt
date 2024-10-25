#!/bin/bash

model_name="arabic-nano-gpt-v1"

# python src/preprocess_data.py

python src/build_tokenizer.py \
    --model_name=$model_name

python src/train_causal_lm.py \
    --split_size=500 \
    --n_embd=384 \
    --n_head=6 \
    --num_epochs=100 \
    --lr=6e-4 \
    --wd=0.0 \
    --warmup=0.0 \
    --batch_size=32 \
    --accum_steps=1 \
    --eval_steps=100 \
    --log_steps=50 \
    --model_name=$model_name \
    --run_name="Overfitting-Small-Batch-v1" \
    --notes="LM Training on Arabic Data using Nano GPT2 Model Architecture" \
    --tags="Modeling,Transformers,GPT2,Language-Modeling,Arabic-Wikipedia" \
