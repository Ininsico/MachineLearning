#!/usr/bin/env python3
"""
Script to convert checkpoint to HuggingFace format
"""

import torch
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))


def convert_checkpoint(checkpoint_path, output_dir):
    """Convert custom checkpoint to HuggingFace format"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    print("Saving model weights...")
    model_path = output_path / 'pytorch_model.bin'
    torch.save(checkpoint['model_state_dict'], model_path)
    
    # Save config
    print("Saving configuration...")
    config = checkpoint['config']
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save training info
    print("Saving training info...")
    training_info = {
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }
    info_path = output_path / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✓ Checkpoint converted successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - pytorch_model.bin")
    print(f"  - config.json")
    print(f"  - training_info.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert checkpoint to HuggingFace format')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    convert_checkpoint(args.checkpoint, args.output)
