# Installation Guide

## Requirements

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- 8GB+ RAM (16GB recommended)
- 10GB disk space

## Quick Install

```bash
# Clone repository
git clone https://github.com/yourorg/flux-text2img.git
cd flux-text2img

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Docker Install

```bash
# Build image
docker build -t flux-t2i .

# Run container
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  flux-t2i
```

## Configuration

Create `.env` file:

```bash
HF_TOKEN=your_huggingface_token
MODEL_ID=black-forest-labs/FLUX.1-schnell
OUTPUT_DIR=./outputs
LOG_LEVEL=INFO
```

## Verify Installation

```bash
python -c "from src.core.pipeline import FluxPipeline; print('✓ Installation successful')"
```

## GPU Setup

### NVIDIA

```bash
# Install CUDA toolkit
# https://developer.nvidia.com/cuda-downloads

# Verify
nvidia-smi
```

### AMD (ROCm)

```bash
# Install ROCm
# https://rocmdocs.amd.com/

# Verify
rocm-smi
```

## Troubleshooting

### Import Errors

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### CUDA Out of Memory

Reduce batch size or image resolution in config.

### API Connection Issues

Check HuggingFace token and network connection.
