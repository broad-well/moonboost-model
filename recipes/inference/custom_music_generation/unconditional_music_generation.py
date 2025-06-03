import glob
from typing import List, Optional

import fire
import pandas as pd
import numpy as np
import os
import re
from generation import MusicLlama
import random
import ast
import json

def main(
    ckpt_dir: str,
    csv_file: str,
    tokenizer_path: str,
    model_config_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    prompt_len: int = 5,
    num_test_data: int = 50,
    max_gen_len: Optional[int] = None,
    finetuned_PEFT_weight_path: Optional[str] = None,
    custom_prompts: Optional[str] = None,
):

    generator = MusicLlama.build(
        ckpt_dir=ckpt_dir,
        model_config_path = model_config_path, 
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        finetuned_PEFT_weight_path = finetuned_PEFT_weight_path) 
    
    df = pd.read_csv(csv_file)
    split = "test"
    test_filenames = df[df['split'] == split]['file_base_name'].tolist()
    test_files_sampled = random.sample(test_filenames, num_test_data)
    prompts = []

    if not custom_prompts:
        for filename in test_files_sampled:
            test_data = np.load(os.path.join(os.path.dirname(csv_file), 'processed', filename))
            test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
            prompts.append(test_data_with_sos[:prompt_len])
    else:
        for path in glob.glob(custom_prompts):
            test_data = np.load(path)
            test_data_with_sos = generator.tokenizer.encode_series(test_data, if_add_sos = True, if_add_eos = False)
            prompts.append(test_data_with_sos[:prompt_len])

    results = generator.music_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    
    save_folder = os.path.join(finetuned_PEFT_weight_path, os.path.basename(ckpt_dir), f"temperature_{temperature}_top_p_{top_p}")
    os.makedirs(save_folder, exist_ok=True)

    for i, (dialog, result) in enumerate(zip(prompts, results)):
        epoch_step = '6-0'
        save_path = f'{save_folder}/{epoch_step}_{str(i)}.mid'
        result['generation']['content'].save(save_path)
        result['generation']['prompt'].save(save_path.replace(".mid", "_prompt.mid"))
        print(f"Midi and prompt saved to {save_path} and {save_path.replace('.mid', '_prompt.mid')}")
        print("\n==================================\n")
if __name__ == "__main__":
    fire.Fire(main)
