"""
Data preprocessing utilities for text datasets
"""

import re
import unicodedata
from typing import List
import ftfy


class TextPreprocessor:
    """Text cleaning and preprocessing"""
    
    def __init__(self, lowercase=False, remove_urls=True, remove_emails=True):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Fix unicode issues
        text = ftfy.fix_text(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def filter_quality(self, text: str, min_length=50, max_length=10000) -> bool:
        """Filter low-quality text"""
        # Length check
        if len(text) < min_length or len(text) > max_length:
            return False
        
        # Check for minimum word count
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for excessive repetition
        if self._has_excessive_repetition(text):
            return False
        
        # Check for minimum alphabetic ratio
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.6:
            return False
        
        return True
    
    def _has_excessive_repetition(self, text: str, threshold=0.3) -> bool:
        """Check for excessive character repetition"""
        if len(text) < 10:
            return False
        
        # Check for repeated characters
        repeated = sum(1 for i in range(len(text)-1) if text[i] == text[i+1])
        ratio = repeated / len(text)
        
        return ratio > threshold
    
    def deduplicate_lines(self, lines: List[str]) -> List[str]:
        """Remove duplicate lines while preserving order"""
        seen = set()
        unique_lines = []
        
        for line in lines:
            line_hash = hash(line.strip())
            if line_hash not in seen:
                seen.add(line_hash)
                unique_lines.append(line)
        
        return unique_lines


def process_dataset(input_path: str, output_path: str, preprocessor: TextPreprocessor):
    """Process entire dataset"""
    import os
    from tqdm import tqdm
    
    print(f"Processing dataset: {input_path}")
    
    total_docs = 0
    filtered_docs = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile):
            total_docs += 1
            
            # Clean text
            cleaned = preprocessor.clean_text(line)
            
            # Filter quality
            if preprocessor.filter_quality(cleaned):
                outfile.write(cleaned + '\n')
            else:
                filtered_docs += 1
    
    print(f"Total documents: {total_docs}")
    print(f"Filtered out: {filtered_docs} ({filtered_docs/total_docs*100:.2f}%)")
    print(f"Remaining: {total_docs - filtered_docs}")


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor(
        lowercase=False,
        remove_urls=True,
        remove_emails=True
    )
    
    # Test text
    test_text = "This is a TEST text with URLs http://example.com and emails test@example.com!!!"
    cleaned = preprocessor.clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
