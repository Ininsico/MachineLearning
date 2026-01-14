"""
Tokenizer training script using BPE
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import json
from pathlib import Path
from tqdm import tqdm


class CustomTokenizer:
    """Custom BPE tokenizer for the language model"""
    
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.tokenizer = None
        
    def train(self, files, output_dir):
        """Train tokenizer on text files"""
        print(f"Training tokenizer with vocab size {self.vocab_size}...")
        
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # Pre-tokenizer
        tokenizer.pre_tokenizer = Whitespace()
        
        # Trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=3,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        )
        
        # Train
        tokenizer.train(files, trainer)
        
        # Post-processor
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )
        
        self.tokenizer = tokenizer
        
        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tokenizer.save(str(output_path / "tokenizer.json"))
        
        # Save vocab
        vocab = tokenizer.get_vocab()
        with open(output_path / "vocab.json", 'w') as f:
            json.dump(vocab, f, indent=2)
        
        print(f"Tokenizer saved to {output_dir}")
        print(f"Vocabulary size: {len(vocab)}")
        
        return tokenizer
    
    def load(self, tokenizer_path):
        """Load trained tokenizer"""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        return self.tokenizer
    
    def encode(self, text):
        """Encode text to token IDs"""
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids):
        """Decode token IDs to text"""
        return self.tokenizer.decode(ids)
    
    def batch_encode(self, texts):
        """Encode batch of texts"""
        return [self.encode(text) for text in texts]


def tokenize_dataset(input_dir, output_dir, tokenizer_path, max_length=2048):
    """Tokenize entire dataset and save as numpy arrays"""
    import numpy as np
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Tokenizing dataset...")
    
    file_idx = 0
    current_tokens = []
    
    for text_file in tqdm(list(input_path.glob("*.txt"))):
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Tokenize
                encoding = tokenizer.encode(line.strip())
                tokens = encoding.ids
                
                current_tokens.extend(tokens)
                
                # Save in chunks
                if len(current_tokens) >= 1_000_000:  # 1M tokens per file
                    output_file = output_path / f"tokens_{file_idx:05d}.npy"
                    np.save(output_file, np.array(current_tokens, dtype=np.int32))
                    
                    file_idx += 1
                    current_tokens = []
    
    # Save remaining tokens
    if current_tokens:
        output_file = output_path / f"tokens_{file_idx:05d}.npy"
        np.save(output_file, np.array(current_tokens, dtype=np.int32))
    
    print(f"Tokenization complete! Saved {file_idx + 1} files to {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Tokenizer module loaded")
