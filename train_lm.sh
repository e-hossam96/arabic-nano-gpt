#!/bin/bash

model_name="arabic-nano-gpt-v1"

python src/preprocess_data.py

python src/build_tokenizer.py \
    --model_name=$model_name

python src/train_causal_lm.py \
    --n_embd=384 \
    --n_head=6 \
    --num_epochs=24 \
    --lr=2e-4 \
    --wd=1e-5 \
    --warmup=0.01 \
    --batch_size=64 \
    --accum_steps=8 \
    --eval_steps=5000 \
    --log_steps=2000 \
    --model_name=$model_name \
    --run_name="Arabic-NanoGPT-LM-on-Wikipedia-Docs-23-V1" \
    --notes="LM Training on Arabic Data using Nano GPT2 Model Architecture" \
    --tags="Modeling,Transformers,GPT2,Language-Modeling,Arabic-Wikipedia"
