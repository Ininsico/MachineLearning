# Shakespeare GPT - Trained Model

This model was trained from scratch on Shakespeare's complete works.

## Files:
- model.pt: Trained weights (10.65M parameters)
- model.py: Transformer implementation
- sample.py: Text generation script
- config.py: Model configuration
- prepare.py: Data processing code
- meta.pkl: Vocabulary metadata

## Quick Start:
1. Install: pip install torch tiktoken
2. Run: python run_shakespeare.py "Thy Victory"

## Example Prompts:
- "ROMEO:"
- "JULIET:"
- "HAMLET:"
- "To be or not to be"

## Training Details:
- 10.65 million parameters
- Trained for 2000 iterations
- Final loss: 1.12
- Character-level model
