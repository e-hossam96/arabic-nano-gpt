#!/bin/bash
python src/preprocess_data.py
python src/build_tokenizer.py
python src/train_causal_lm.py
