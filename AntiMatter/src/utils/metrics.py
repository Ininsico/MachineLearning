"""
Metrics calculation utilities
"""

import torch
import numpy as np
from typing import List, Dict
import math


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(loss)


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate token-level accuracy"""
    predictions = torch.argmax(logits, dim=-1)
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy


def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """Calculate BLEU score"""
    from nltk.translate.bleu_score import corpus_bleu
    
    # Tokenize
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]
    
    score = corpus_bleu(ref_tokens, pred_tokens)
    return score


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    
    # Average scores
    avg_scores = {key: np.mean(values) for key, values in scores.items()}
    return avg_scores


def calculate_diversity(texts: List[str]) -> Dict[str, float]:
    """Calculate text diversity metrics"""
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.split())
    
    unique_tokens = set(all_tokens)
    
    metrics = {
        'unique_tokens': len(unique_tokens),
        'total_tokens': len(all_tokens),
        'diversity_ratio': len(unique_tokens) / len(all_tokens) if all_tokens else 0,
        'avg_length': np.mean([len(text.split()) for text in texts])
    }
    
    return metrics


class MetricsTracker:
    """Track and aggregate metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_average(self, key: str) -> float:
        """Get average value for a metric"""
        if key not in self.metrics:
            return 0.0
        return self.metrics[key] / self.counts[key]
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get all average metrics"""
        return {key: self.get_average(key) for key in self.metrics.keys()}
    
    def get_summary(self) -> str:
        """Get formatted summary"""
        averages = self.get_all_averages()
        summary = " | ".join([f"{k}: {v:.4f}" for k, v in averages.items()])
        return summary
