import pytest
from src.preprocessing.prompt_processor import PromptProcessor

def test_prompt_processing():
    """Test prompt cleaning"""
    processor = PromptProcessor()
    
    # Test whitespace removal
    result = processor.process("  test   prompt  ")
    assert result == "test prompt"
    
    # Test length limit
    long_prompt = "a" * 1000
    result = processor.process(long_prompt)
    assert len(result) <= processor.max_length

def test_prompt_enhancement():
    """Test prompt enhancement"""
    processor = PromptProcessor()
    
    result = processor.enhance("a cat")
    assert "high quality" in result
    assert "detailed" in result
    assert "a cat" in result
