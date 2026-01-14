#!/usr/bin/env python3
"""
Script to evaluate model on test set
"""

import torch
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model.architecture import CustomLM
from utils.evaluation import ModelEvaluator
from preprocessing.dataset import TextDataset
from torch.utils.data import DataLoader


def main():
    print("="*60)
    print("Model Evaluation Script")
    print("="*60)
    
    # Load model
    print("\n[1/4] Loading model...")
    checkpoint_path = 'checkpoints/checkpoint_step_75000.pt'
    model = CustomLM.from_pretrained(checkpoint_path)
    model.eval()
    print(f"✓ Loaded model with {model.get_num_params():,} parameters")
    
    # Load test dataset
    print("\n[2/4] Loading test dataset...")
    test_dataset = TextDataset('data/processed/validation', max_length=2048)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    # Initialize evaluator
    print("\n[3/4] Running evaluation...")
    evaluator = ModelEvaluator(model, None, device='cuda')
    
    # Evaluate perplexity
    results = evaluator.evaluate_perplexity(test_loader)
    
    print("\n[4/4] Evaluation Results:")
    print("-"*60)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Total Tokens: {results['total_tokens']:,}")
    print("-"*60)
    
    # Save results
    output_path = 'results/evaluations/test_evaluation.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
