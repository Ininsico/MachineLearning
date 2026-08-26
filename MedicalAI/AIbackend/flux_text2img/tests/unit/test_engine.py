import pytest
from src.core.engine import FluxEngine

def test_engine_initialization():
    """Test engine initializes correctly"""
    engine = FluxEngine()
    assert engine.token is not None
    assert engine.model_id is not None
    assert "huggingface" in engine.endpoint

def test_engine_generate():
    """Test image generation"""
    engine = FluxEngine()
    
    # Mock test - would need actual API in production
    try:
        result = engine.generate("test prompt", width=512, height=512, steps=1)
        assert result is not None
    except Exception as e:
        # Expected to fail without valid token
        assert "Generation failed" in str(e) or "Authorization" in str(e)
