# Usage Guide

## Basic Usage

### Python API

```python
from src.core.pipeline import FluxPipeline

# Initialize pipeline
pipeline = FluxPipeline()

# Generate image
image = pipeline(
    "a beautiful sunset over mountains",
    output_path="sunset.png",
    width=1024,
    height=1024,
    steps=4,
    guidance=3.5
)
```

### Command Line

```bash
python main.py
```

### REST API

Start server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Generate image:

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat in space",
    "width": 1024,
    "height": 1024
  }'
```

## Batch Processing

```python
from src.inference.batch.processor import BatchProcessor

processor = BatchProcessor(batch_size=4)

prompts = [
    "a dog in a park",
    "a cat on a roof",
    "a bird in the sky"
]

processor.process_batch(prompts, output_dir="./outputs")
```

## Advanced Features

### Custom Pipeline

```python
from src.core.engine import FluxEngine
from src.preprocessing.prompt_processor import PromptProcessor

engine = FluxEngine()
processor = PromptProcessor()

# Enhanced prompt
prompt = processor.enhance("a landscape")

# Generate
image = engine.generate(prompt, width=1024, height=1024)
```

### Evaluation

```python
from src.evaluation.metrics.fid import FIDCalculator

calculator = FIDCalculator()
fid = calculator.compute_fid(real_images, generated_images)
print(f"FID Score: {fid:.2f}")
```

## Best Practices

1. **Prompt Engineering**: Be specific and descriptive
2. **Resolution**: Start with 512x512 for testing
3. **Steps**: 4 steps for speed, 20+ for quality
4. **Guidance**: 3.5-7.5 for most cases

## Examples

See `examples/` directory for more examples.
