import copy
import json

import torch
from torch.utils.data import Dataset
import glob
import mido
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
import tqdm
import numpy as np
import pandas as pd
import os

class PlayerClassificationDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        assert partition=="train" or partition=="test"
        self.data_dir = dataset_config.data_dir
        self.seq_len = dataset_config.seq_len #max duration in seconds
        split_data = pd.read_csv(dataset_config.csv_file)
        file_basenames = split_data['file_base_name'].values
        splits = split_data['split'].values     
        labels = split_data['label'].values   
        self.file_basenames = [f for f, s in zip(file_basenames, splits) if s == partition]
        self.labels = [l for l, s in zip(labels, splits) if s == partition]
        if partition == "train":
            self.chunked_files, self.chunked_labels = self.get_chunked_files_labels()
        else:
            self.chunked_files, self.chunked_labels = [np.load(os.path.join(self.data_dir, "processed",filename))  for filename in self.file_basenames], self.labels
        self.tokenizer = tokenizer
    def get_chunked_files_labels(self):
        self.chunked_files, self.chunked_labels = [], []
        for label, filename in zip(self.labels, self.file_basenames):
            raw_tokens = np.load(os.path.join(self.data_dir, "processed",filename))    
            # Split the raw tokens into chunks of length self.seq_len
            num_chunks = len(raw_tokens) // self.seq_len
            for i in range(num_chunks):
                chunk = raw_tokens[i * self.seq_len:(i + 1) * self.seq_len]
                self.chunked_files.append(chunk)
                self.chunked_labels.append(label)
            
            # Handle the remainder, if any
            remainder = len(raw_tokens) % self.seq_len
            if remainder > 0:
                chunk = raw_tokens[-remainder:]
                self.chunked_files.append(chunk)
                self.chunked_labels.append(label)
        return self.chunked_files, self.chunked_labels
    def __len__(self):
        return len(self.chunked_files)

    def __getitem__(self, index):
        # encoded_tokens = self.tokenizer.encode_series(raw_tokens, if_add_sos = True, if_add_eos = True) #SOS and EOS are added in the tokenizer
        encoded_tokens = self.tokenizer.encode_series_player_classification(self.chunked_files[index], if_add_sos = True, if_add_eos = True, if_add_classification_token=True) #SOS and EOS are added in the tokenizer

        classification_labels = [self.chunked_labels[index] for _ in range(len(encoded_tokens))] 

        return {
            "input_ids": encoded_tokens,
            "labels": classification_labels,
            "attention_mask":[0 for _ in range(len(encoded_tokens))] #dummy attention mask
        }
