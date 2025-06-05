#!/bin/bash

python recipes/inference/custom_music_generation/unconditional_music_generation.py \
    --csv_file ../moonboost/dataset/train_test_split.csv \
    --top_p 0.97 \
    --temperature 1.15 \
    --model_config_path src/llama_recipes/configs/model_config.json \
    --ckpt_dir moonbeam_839M.pt \
    --finetuned_PEFT_weight_path 5-1000.safetensors-confin \
    --tokenizer_path tokenizer.model \
    --max_seq_len 1024 \
    --max_gen_len 1024 \
    --max_batch_size 16 \
    --num_test_data 40 \
    --prompt_len 80 \
    --custom_prompts prompts/