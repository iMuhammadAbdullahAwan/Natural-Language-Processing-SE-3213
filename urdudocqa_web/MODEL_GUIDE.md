# Local Model Configuration Guide

## Available Models

### 1. FLAN-T5 Small (Recommended)
- **Model**: `google/flan-t5-small`
- **Size**: 77M parameters (~300 MB)
- **RAM Required**: 1-2 GB
- **Speed**: Very Fast
- **Quality**: Good for basic Q&A
- **Best for**: Quick responses, limited resources

### 2. FLAN-T5 Base
- **Model**: `google/flan-t5-base`
- **Size**: 250M parameters (~1 GB)
- **RAM Required**: 3-4 GB
- **Speed**: Moderate
- **Quality**: Better understanding and responses
- **Best for**: Balanced performance and quality

### 3. DistilGPT2
- **Model**: `distilgpt2`
- **Size**: 82M parameters (~350 MB)
- **RAM Required**: 1-2 GB
- **Speed**: Fast
- **Quality**: Good for text generation
- **Best for**: Creative responses, text completion

## System Requirements

### Minimum Requirements
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **CPU**: Any modern processor
- **GPU**: Optional (CPU mode available)

### Recommended Requirements
- **RAM**: 8 GB or more
- **Storage**: 5 GB free space
- **CPU**: Multi-core processor
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for faster processing)

## Performance Tips

1. **For Low-End Systems**: Use `flan-t5-small`
2. **For Better Quality**: Use `flan-t5-base`
3. **GPU Acceleration**: Models automatically use GPU if available
4. **CPU Optimization**: Models fallback to CPU if no GPU
5. **Memory Management**: Models use float16 precision on GPU to save memory

## Model Download

Models are automatically downloaded on first use:
- Downloaded to: `~/.cache/huggingface/transformers/`
- Internet required for first download only
- Subsequent runs work offline

## Switching Models

Change the `MODEL_TYPE` in your `.env` file:
```bash
MODEL_TYPE=flan-t5-small  # Default
MODEL_TYPE=flan-t5-base   # Better quality
MODEL_TYPE=distilgpt2     # Alternative option
```

## Troubleshooting

### Out of Memory Errors
1. Use smaller model (`flan-t5-small`)
2. Reduce `CHUNK_SIZE` in config
3. Reduce `SEARCH_K` in config
4. Close other applications

### Slow Performance
1. Enable GPU acceleration
2. Use smaller chunks
3. Reduce number of retrieved documents
4. Use `flan-t5-small` for speed

### Model Download Issues
1. Check internet connection
2. Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`
3. Restart application
