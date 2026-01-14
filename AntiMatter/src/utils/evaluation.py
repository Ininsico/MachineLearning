"""
Model evaluation utilities
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import json


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_perplexity(self, dataloader):
        """Calculate perplexity on dataset"""
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, loss = self.model(input_ids, labels=labels)
                
                # Count valid tokens
                mask = labels != -100
                num_tokens = mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens
        }
    
    def generate_samples(self, prompts: List[str], max_length=100, temperature=0.8, top_k=50):
        """Generate text samples from prompts"""
        samples = []
        
        for prompt in tqdm(prompts, desc="Generating"):
            # Encode prompt
            input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
            
            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            # Decode
            generated_text = self.tokenizer.decode(output_ids[0].tolist())
            samples.append({
                'prompt': prompt,
                'generated': generated_text
            })
        
        return samples
    
    def evaluate_generation_quality(self, num_samples=100):
        """Evaluate generation quality with various metrics"""
        prompts = [
            "The future of artificial intelligence",
            "Once upon a time",
            "In the field of machine learning",
            "The most important thing to remember is",
            "Scientists have discovered"
        ] * (num_samples // 5)
        
        samples = self.generate_samples(prompts[:num_samples])
        
        # Calculate metrics
        lengths = [len(s['generated'].split()) for s in samples]
        
        metrics = {
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'num_samples': num_samples
        }
        
        return metrics, samples
    
    def benchmark_speed(self, batch_size=32, seq_length=512, num_iterations=100):
        """Benchmark inference speed"""
        import time
        
        # Create dummy input
        dummy_input = torch.randint(0, 50257, (batch_size, seq_length)).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = (batch_size * seq_length * num_iterations) / total_time
        
        return {
            'avg_time_per_batch': avg_time,
            'throughput_tokens_per_sec': throughput,
            'batch_size': batch_size,
            'seq_length': seq_length
        }
    
    def save_evaluation_report(self, output_path, results):
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation report saved to {output_path}")


if __name__ == "__main__":
    print("Evaluation module loaded")
