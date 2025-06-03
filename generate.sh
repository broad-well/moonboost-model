#!/bin/bash

python recipes/inference/custom_music_generation/unconditional_music_generation.py \
    --csv_file ../moonboost/dataset/train_test_split.csv \
    --top_p 0.95 \
    --temperature 0.9 \
    --model_config_path src/llama_recipes/configs/model_config.json \
    --ckpt_dir moonbeam_309M.pt \
    --finetuned_PEFT_weight_path checkpoints/uncon/1-0.safetensors/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_gen_len 512 \
    --max_batch_size 16 \
    --num_test_data 40 \
    --prompt_len 50