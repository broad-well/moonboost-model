# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Optional
    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class lakhmidi_dataset:
    dataset: str = "lakhmidi_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/data/scratch/acw753/lakhmidi_processed"
    csv_file: str = "/data/scratch/acw753/lakhmidi_processed/train_test_split.csv"

@dataclass
class merge_dataset:
    dataset: str = "merge_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/data/scratch/acw753/processed_midi"
    csv_file: str = "/data/scratch/acw753/processed_midi/train_test_split.csv"

@dataclass
class player_classification_dataset: #Pijama, individual_eval=False, seq_dur = 15, non_overlap_ratio = 0.125
    dataset: str = "player_classification_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/data/scratch/acw753/finetune/Giant_Piano_MIDI_processed_top30_seqlen_4096"
    csv_file: str = "/data/scratch/acw753/finetune/Giant_Piano_MIDI_processed_top30_seqlen_4096/train_test_split.csv"
    seq_len: Optional[int] = 1000#fixed sequence length during training, if seq_len and seq_dur are both None, concat all events in midi 
    seq_dur: Optional[str] = None #fixed sequence duration during training, if seq_len and seq_dur are both None, concat all events in midi 
    non_overlap_ratio: float = 0.25 #allowed range (0, 1]
    individual_eval: bool = True