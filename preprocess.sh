#!/bin/bash

python ./data_preprocess.py \
    --dataset_name vgm \
    --dataset_folder prompts \
    --output prompts \
    --model_config ./src/llama_recipes/configs/model_config.json \
    --train_test_split_file None --train_ratio 0 --ts_threshold None