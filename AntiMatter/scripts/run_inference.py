#!/usr/bin/env python3
"""
Quick script to run inference with the trained model
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model.architecture import CustomLM
from preprocessing.tokenizer import CustomTokenizer


def main():
    print("Loading model...")
    model = CustomLM.from_pretrained('checkpoints/checkpoint_step_75000.pt')
    model.eval()
    
    print("Loading tokenizer...")
    tokenizer = CustomTokenizer()
    tokenizer.load('data/vocab/tokenizer.json')
    
    print("\n" + "="*60)
    print("Custom 300M Language Model - Interactive Demo")
    print("="*60)
    
    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        # Encode prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        
        # Generate
        print("\nGenerating...")
        output_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50
        )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        print("\nGenerated:")
        print("-" * 60)
        print(generated_text)
        print("-" * 60)
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
