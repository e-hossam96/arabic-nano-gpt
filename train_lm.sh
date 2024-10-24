#!/bin/bash
# python src/preprocess_data.py
# python src/build_tokenizer.py
python src/train_causal_lm.py \
    --num_epochs=24 \
    --lr=1e-3 \
    --batch_size=64 \
    --accum_steps=4 \
    --eval_steps=1000 \
    --log_steps=200 \
    --run_name="Arabic-NanoGPT-LM-on-Wikipedia-Docs-23" \
    --notes="LM Training on Arabic Data using Nano GPT2 Model Architecture" \
    --tags="Modeling,Transformers,GPT2,Language-Modeling,Arabic-Wikipedia" \
    --torch_compile
