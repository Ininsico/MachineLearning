"""
Custom dataset class for language modeling
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path


class TextDataset(Dataset):
    """Dataset for language modeling with tokenized text"""
    
    def __init__(self, data_path, max_length=2048, stride=1024):
        """
        Args:
            data_path: Path to tokenized data directory
            max_length: Maximum sequence length
            stride: Stride for creating overlapping sequences
        """
        self.max_length = max_length
        self.stride = stride
        
        # Load tokenized data
        self.data = self._load_data(data_path)
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _load_data(self, data_path):
        """Load all tokenized files"""
        data_files = list(Path(data_path).glob('*.npy'))
        
        if not data_files:
            raise ValueError(f"No .npy files found in {data_path}")
        
        all_tokens = []
        for file_path in data_files:
            tokens = np.load(file_path)
            all_tokens.append(tokens)
        
        # Concatenate all tokens
        return np.concatenate(all_tokens)
    
    def _create_sequences(self):
        """Create overlapping sequences from data"""
        sequences = []
        
        for i in range(0, len(self.data) - self.max_length, self.stride):
            seq = self.data[i:i + self.max_length + 1]
            sequences.append(seq)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a single sequence"""
        seq = self.sequences[idx]
        
        # Input and labels (shifted by 1)
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class StreamingTextDataset(Dataset):
    """Memory-efficient streaming dataset for large corpora"""
    
    def __init__(self, data_path, max_length=2048, buffer_size=10000):
        self.data_path = data_path
        self.max_length = max_length
        self.buffer_size = buffer_size
        
        # Get file list
        self.files = list(Path(data_path).glob('*.npy'))
        
        # Calculate total sequences
        self.total_sequences = self._count_sequences()
    
    def _count_sequences(self):
        """Count total number of sequences across all files"""
        total = 0
        for file_path in self.files:
            file_size = os.path.getsize(file_path)
            # Estimate sequences (assuming int32 tokens)
            num_tokens = file_size // 4
            num_seqs = max(1, (num_tokens - self.max_length) // self.max_length)
            total += num_seqs
        
        return total
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        """Load sequence on-demand"""
        # Determine which file and offset
        file_idx = idx % len(self.files)
        local_idx = idx // len(self.files)
        
        # Load file
        tokens = np.load(self.files[file_idx])
        
        # Get sequence
        start_idx = local_idx * self.max_length
        end_idx = start_idx + self.max_length + 1
        
        if end_idx > len(tokens):
            # Wrap around or pad
            seq = tokens[start_idx:]
            padding = np.zeros(end_idx - len(tokens), dtype=tokens.dtype)
            seq = np.concatenate([seq, padding])
        else:
            seq = tokens[start_idx:end_idx]
        
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


if __name__ == "__main__":
    # Test dataset
    print("Dataset module loaded successfully")
