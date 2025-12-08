# WORKING_TEST_FIXED.py
import torch
import pickle
import sys
import os
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPT

print("=" * 60)
print("🚀 SHAKESPEARE GPT - WORKING TEST (FIXED)")
print("=" * 60)

# Load model
print("\n📥 Loading model...")
checkpoint = torch.load('model.pt', map_location='cpu')
config = GPTConfig(**checkpoint['model_args'])
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()

print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Load vocabulary
print("\n🔤 Loading vocabulary...")
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
print(f"✅ Vocabulary: {len(stoi)} characters")

# Get prompt from command line
if len(sys.argv) > 1:
    prompt = sys.argv[1]
else:
    prompt = "ROMEO:"

print(f"\n🎭 Generating: '{prompt}'")
print("-" * 50)

# Encode
try:
    idx = torch.tensor([stoi[ch] for ch in prompt], dtype=torch.long).unsqueeze(0)
except KeyError as e:
    print(f"❌ Character not in vocabulary: {e}")
    print(f"Available characters: {''.join(stoi.keys())}")
    sys.exit(1)

# Generate with typing effect
print(prompt, end="", flush=True)

for i in range(150):  # Generate 150 characters
    with torch.no_grad():
        # GET LOGITS FROM TUPLE
        outputs = model(idx)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Temperature
        logits = logits[:, -1, :] / 0.8
        
        # Softmax and sample
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append
        idx = torch.cat((idx, idx_next), dim=1)
    
    # Print character
    next_char = itos[idx_next.item()]
    print(next_char, end="", flush=True)
    
    # Small delay for effect
    time.sleep(0.03)

print("\n" + "=" * 50)
print("GENERATION COMPLETE!")