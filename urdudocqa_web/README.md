# اردو دستاویز سوالات و جوابات (Urdu Document Q&A)

An advanced RAG (Retrieval-Augmented Generation) system for asking questions about Urdu PDF documents using local open-source models.

## ✨ Features

- **Local Models**: Uses lightweight Hugging Face models (no API keys required)
- **Multilingual Support**: Optimized for Urdu text processing
- **Modern UI**: ChatGPT-like interface with Urdu fonts
- **Vector Search**: ChromaDB for efficient document retrieval
- **GPU Acceleration**: Automatic GPU detection and usage
- **Lightweight**: Runs on modest hardware

## 🎯 Available Models

| Model | Size | RAM Required | Speed | Quality | Best For |
|-------|------|--------------|-------|---------|----------|
| **flan-t5-small** | 77M | 1-2 GB | ⚡ Very Fast | ✅ Good | Quick responses |
| **flan-t5-base** | 250M | 3-4 GB | 🔄 Moderate | ⭐ Better | Balanced performance |
| **distilgpt2** | 82M | 1-2 GB | ⚡ Fast | ✅ Good | Creative responses |

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo>
cd urdudocqa_web

# Run automated setup
python setup.py
```

### 2. Manual Setup (Alternative)
```bash
# Install dependencies
pip install -r requirements.txt

# Copy configuration
cp .env.example .env

# Create directories
mkdir data chroma_db logs
```

### 3. Configure Model
Edit `.env` file:
```bash
# Choose your model (recommended: flan-t5-small)
MODEL_TYPE=flan-t5-small

# Adjust settings for your hardware
CHUNK_SIZE=800
SEARCH_K=3
```

### 4. Run Application
```bash
# Option 1: Direct
python app.py

# Option 2: With uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open Browser
Visit: `http://localhost:8000`

## 💻 System Requirements

### Minimum
- **Python**: 3.8+
- **RAM**: 4 GB
- **Storage**: 2 GB free
- **CPU**: Any modern processor

### Recommended
- **RAM**: 8 GB+
- **Storage**: 5 GB free
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CPU**: Multi-core processor

## 📋 Usage Instructions

1. **Upload PDF**: Click "فائل منتخب کریں" and select an Urdu PDF
2. **Wait for Processing**: Document will be chunked and indexed
3. **Ask Questions**: Type your question in Urdu in the input field
4. **Get Answers**: AI will respond based on document content

## 🔧 Configuration Options

### Model Selection
```bash
# Fast and lightweight (default)
MODEL_TYPE=flan-t5-small

# Better quality responses
MODEL_TYPE=flan-t5-base

# Creative text generation
MODEL_TYPE=distilgpt2
```

### Performance Tuning
```bash
# For low-end systems
CHUNK_SIZE=500
SEARCH_K=2
SCORE_THRESHOLD=0.2

# For high-end systems
CHUNK_SIZE=1000
SEARCH_K=5
SCORE_THRESHOLD=0.4
```

## 🛠️ Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Solution 1: Use smaller model
MODEL_TYPE=flan-t5-small

# Solution 2: Reduce chunk size
CHUNK_SIZE=500
SEARCH_K=2
```

#### Slow Performance
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode (if needed)
export CUDA_VISIBLE_DEVICES=""
```

#### Model Download Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python setup.py
```

### Performance Tips

1. **GPU Acceleration**: Ensure CUDA is installed for GPU support
2. **Memory Management**: Close other applications when running
3. **Model Choice**: Start with `flan-t5-small` for testing
4. **Document Size**: Split large PDFs into smaller files

## 📁 Project Structure

```
urdudocqa_web/
├── app.py                 # FastAPI application
├── rag_pipeline.py        # RAG pipeline with local models
├── preprocess.py          # PDF processing and text cleaning
├── templates/
│   └── index.html         # Web interface
├── fonts/                 # Urdu fonts
├── data/                  # Uploaded PDFs
├── chroma_db/            # Vector database
├── requirements.txt      # Dependencies
├── setup.py             # Automated setup
├── MODEL_GUIDE.md       # Detailed model information
└── .env.example         # Configuration template
```

## 🔍 API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload PDF document
- `POST /query`: Ask questions
- `GET /health`: System status and model info
- `GET /config`: Current configuration

## 🌟 Advanced Features

### Custom Prompts
Models use optimized prompts for Urdu Q&A:
- FLAN-T5: Instruction-tuned for better comprehension
- Context-aware response generation
- Fallback handling for unclear questions

### Vector Search
- Multilingual embeddings for better Urdu support
- Similarity-based document retrieval
- Configurable search parameters

### Error Handling
- Graceful degradation to simpler models
- Comprehensive logging
- User-friendly error messages in Urdu

## 📊 Performance Benchmarks

| System | Model | Processing Time | Memory Usage |
|--------|-------|----------------|--------------|
| CPU Only | flan-t5-small | ~3-5 seconds | 1.5 GB |
| GPU (4GB) | flan-t5-small | ~1-2 seconds | 2 GB |
| GPU (8GB) | flan-t5-base | ~2-3 seconds | 3.5 GB |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test with different models
5. Submit pull request

## 📝 License

This project is open source. See LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review MODEL_GUIDE.md
3. Open GitHub issue
4. Check logs in `logs/` directory

---

**Made with ❤️ for Urdu NLP**
