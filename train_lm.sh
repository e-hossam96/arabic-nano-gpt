#!/bin/bash

# data config
DATA_CKPT=wikimedia/wikipedia
SUB_DATA=20231101.ar
PROCESSED_DATA_PATH=data/$DATA_CKPT.csv

# tokenizer config
BASE_MODEL=openai-community/gpt2
MODEL_NAME=arabic-nano-gpt-v2
MODEL_PATH=models/$MODEL_NAME
MODEL_MAX_LENGTH=1024
VOCAB_SIZE=16384

# model config
EMBED_SIZE=384
NUM_ATT_HEAD=6
NUM_ATT_LAYERS=8

# training config
NUM_EPOCHS=5
BATCH_SIZE=32
ACCUM_STEPS=8
EVAL_STEPS=1000
LOG_STEPS=400
LR=0.0001
WD=0.000001
WARMUP=0.01

# weights & biases config
PROJECT_NAME=Arabic-Nano-GPT
JOB_TYPE=LM-Modeling
RUN_NAME=Arabic-NanoGPT-LM-on-Wikipedia-Docs-23-V2
NOTES="LM Training on Arabic Data using Nano GPT2 Model Architecture"
TAGS=Modeling,Transformers,GPT2,Language-Modeling,Arabic-Wikipedia

python src/preprocess_data.py \
    --data_ckpt=$DATA_CKPT \
    --sub_data=$SUB_DATA \
    --split_name=train \
    --processed_data_file_path=$PROCESSED_DATA_PATH

python src/build_tokenizer.py \
    --model_ckpt=$BASE_MODEL \
    --data_ckpt=$DATA_CKPT \
    --processed_data_file_path=$PROCESSED_DATA_PATH \
    --model_max_length=$MODEL_MAX_LENGTH \
    --vocab_size=$VOCAB_SIZE \
    --model_name=$MODEL_NAME \
    --target_model_path=$MODEL_PATH

python src/train_causal_lm.py \
    --n_embd=$EMBED_SIZE \
    --n_head=$NUM_ATT_HEAD \
    --n_layer=$NUM_ATT_LAYERS \
    --num_epochs=$NUM_EPOCHS \
    --lr=$LR \
    --wd=$WD \
    --warmup=$WARMUP \
    --batch_size=$BATCH_SIZE \
    --accum_steps=$ACCUM_STEPS \
    --eval_steps=$EVAL_STEPS \
    --log_steps=$LOG_STEPS \
    --torch_compile \
    --model_name=$MODEL_NAME \
    --run_name=$RUN_NAME \
    --notes="$NOTES" \
    --tags=$TAGS
