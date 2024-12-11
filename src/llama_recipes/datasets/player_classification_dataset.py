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
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import random
def analyze_dataset_statistics(chunked_files, chunked_labels, split="Dataset", save_dir="plots"):
    """
    Analyze statistics of chunked_files and chunked_labels and save plots to local directory.
    
    Parameters:
        chunked_files (list of lists): Nested list where each sublist represents a file chunk (e.g., frames or samples).
        chunked_labels (list): List of labels corresponding to chunked files.
        split (str): Name of the dataset (e.g., "Train", "Test") for display purposes.
        save_dir (str): Directory where plots will be saved.
    
    Returns:
        None
    """
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Analyze chunked_files length distribution
    file_lengths = [len(chunk) for chunk in chunked_files]
    
    # Analyze chunked_labels class distribution
    label_distribution = Counter(chunked_labels)
    
    # Display statistics
    print(f"--- {split} Statistics ---")
    print(f"Number of files: {len(chunked_files)}")
    print(f"File length (min, max, mean, std): {min(file_lengths)}, {max(file_lengths)}, {np.mean(file_lengths):.2f}, {np.std(file_lengths):.2f}")
    print(f"Label distribution: {label_distribution}")
    
    # Visualize file lengths distribution
    plt.figure(figsize=(12, 6))
    plt.hist(file_lengths, bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.title(f"{split}: Chunked File Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    file_length_plot_path = os.path.join(save_dir, f"{split}_file_length_distribution.png")
    plt.savefig(file_length_plot_path, format='png', dpi=300)
    plt.close()  # Close the figure to free up memory
    
    # Visualize label distribution
    labels, counts = zip(*label_distribution.items())
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, color='orange', edgecolor='black', alpha=0.75)
    plt.title(f"{split}: Chunked Label Distribution")
    plt.xlabel("Labels")
    plt.ylabel("Counts")
    plt.grid(axis='y')
    label_dist_plot_path = os.path.join(save_dir, f"{split}_label_distribution.png")
    plt.savefig(label_dist_plot_path, format='png', dpi=300)
    plt.close()  # Close the figure to free up memory
    
    print(f"Plots saved to directory: {save_dir}")


def process_file_lazy(args):
    """Helper function to process a single file lazily."""
    label, filename, data_dir, seq_len, partition = args
    raw_tokens = np.load(os.path.join(data_dir, "processed", filename))
    
    chunked_files, chunked_labels = [], []
    slow, fast = 0, 0

    while fast < len(raw_tokens):
        # Calculate the duration of the current token
        onset_time_slow = raw_tokens[slow][0] / 100  # Convert to seconds
        onset_time_fast = raw_tokens[fast][0] / 100  # Convert to seconds
        current_duration = onset_time_fast - onset_time_slow
        
        if current_duration <= seq_len:
            fast += 1
        else:
            abs_onset = raw_tokens[slow][0]
            raw_tokens_adjust_onset = np.array([[event[0]-abs_onset]+event[1:] for event in raw_tokens[slow:fast].tolist()])
            # chunked_files.append(raw_tokens[slow:fast])
            chunked_files.append(raw_tokens_adjust_onset)
            chunked_labels.append(label)
            
            if partition == "train":  # Create overlap
                # slow_new = int((fast + slow) / 2)
                slow_new = int(slow + (fast - slow) / 8)
                if slow_new == slow:  # Ensure progress
                    print(f"breaking the loop to avoid infinite loop: {slow_new} slow old{slow}, len(raw_tokens): {len(raw_tokens)}")
                    slow+=1
                    # break
                else:
                    slow = slow_new
                fast = slow
            elif partition == "test":  # No overlap
                slow = fast
    
    return chunked_files, chunked_labels 
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
        self.partition = partition
        """if partition == "train":
            self.chunked_files, self.chunked_labels = self.get_chunked_files_labels()
        else:
            self.chunked_files, self.chunked_labels = [np.load(os.path.join(self.data_dir, "processed",filename))  for filename in self.file_basenames], self.labels"""
        
        self.chunked_files, self.chunked_labels = self.get_chunked_files_labels_based_on_dur_parallel() #self.get_chunked_files_labels_based_on_dur()
        # analyze_dataset_statistics(self.chunked_files, self.chunked_labels, split=partition, save_dir="/data/home/acw753/musicllama/archive_logs")
        # if partition == "train":
        #     self.balance_data()
        analyze_dataset_statistics(self.chunked_files, self.chunked_labels, split=partition, save_dir="/data/home/acw753/musicllama/archive_logs")

        self.tokenizer = tokenizer
    def balance_data(self, target_count=None):
        """
        Balance the dataset by adjusting the number of labels per class.
        
        Parameters:
            target_count (int, optional): The desired number of samples per class. If None, it uses the minimum class count.
        
        Returns:
            None: Modifies the dataset in place (self.chunked_files and self.chunked_labels).
        """
        # Count the number of labels in each class
        label_distribution = Counter(self.chunked_labels)
        print(f"Original label distribution: {label_distribution}")
        
        # Determine the target number of samples per class
        if target_count is None:
            target_count = int(np.median(list(label_distribution.values())))  # Use the median class size as the target
        print(f"Target count per class: {target_count}")
        
        # Group files and labels by class
        grouped_data = {label: [] for label in label_distribution.keys()}
        for file, label in zip(self.chunked_files, self.chunked_labels):
            grouped_data[label].append(file)
        
        # Resample the data
        balanced_files = []
        balanced_labels = []
        for label, files in grouped_data.items():
            if len(files) > target_count:
                # Undersample if there are too many samples
                sampled_files = random.sample(files, target_count)
            else:
                # Oversample if there are too few samples
                sampled_files = files + random.choices(files, k=target_count - len(files))
            
            balanced_files.extend(sampled_files)
            balanced_labels.extend([label] * target_count)
        
        # Update the dataset
        self.chunked_files = balanced_files
        self.chunked_labels = balanced_labels
        
        # Display the new label distribution
        new_label_distribution = Counter(self.chunked_labels)
        print(f"Balanced label distribution: {new_label_distribution}")    
    
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
    
    def get_chunked_files_labels_based_on_dur(self):
        self.chunked_files, self.chunked_labels = [], []

        for label, filename in zip(self.labels, self.file_basenames):
            raw_tokens = np.load(os.path.join(self.data_dir, "processed", filename))
            
            slow = 0  # Slow counter for the starting index
            fast = 0  # Fast counter for the ending index
            current_duration = 0  # Track the duration of the current chunk

            while fast < len(raw_tokens):
                # Calculate the duration of the current token
                onset_time_slow = raw_tokens[slow][0] / 100  # Convert to seconds
                onset_time_fast = raw_tokens[fast][0] / 100  # Convert to seconds
                current_duration = onset_time_fast - onset_time_slow
                
                if current_duration <= self.seq_len:
                    # Increment the fast counter to include the next token
                    fast += 1
                else:
                    # Append the tokens between slow and fast as a chunk
                    self.chunked_files.append(raw_tokens[slow:fast])
                    self.chunked_labels.append(label)
                    
                    if self.partition == "train": #this creates overlap 
                        slow_new = int((fast + slow) / 2)
                        if slow_new == slow:  # Ensure progress
                            print(f"breaking the loop to avoid infinite loop")
                            break
                        else:
                            slow = slow_new
                        fast = slow
                    elif self.partition == "test": #this doesnt creates overlap 
                        slow = fast

        return self.chunked_files, self.chunked_labels



    def get_chunked_files_labels_based_on_dur_parallel(self):
        import os
        import numpy as np
        from multiprocessing import Pool, cpu_count
    
        self.chunked_files, self.chunked_labels = [], []
        args = [
            (label, filename, self.data_dir, self.seq_len, self.partition)
            for label, filename in zip(self.labels, self.file_basenames)
        ]
        
        with Pool(cpu_count()) as pool:
            # Use `pool.imap` for lazy iteration
            for chunked_files, chunked_labels in pool.imap(process_file_lazy, args):
                self.chunked_files.extend(chunked_files)
                self.chunked_labels.extend(chunked_labels)
        
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
