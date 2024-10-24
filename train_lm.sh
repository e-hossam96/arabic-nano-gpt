#!/bin/bash
python src/preprocess_data.py
python src/build_tokenizer.py
python src/train_causal_lm.py \
    --split_size=200000 \
    --num_epochs=2 \
    --lr=1e-3 \
    --run_name="Arabic-NanoGPT-LM-on-200K-Docs" \
    --notes="LM Training on Arabic Data using Nano GPT2 Model Architecture" \
    --tags="Modeling,Transformers,GPT2,Language-Modeling,Arabic"
