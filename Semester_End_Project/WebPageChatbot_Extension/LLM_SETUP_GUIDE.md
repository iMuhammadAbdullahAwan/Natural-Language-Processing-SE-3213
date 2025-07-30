# LLM Integration Setup Guide

## Overview
This Chrome extension now supports multiple LLM backends for intelligent responses:

1. **Hugging Face Inference API** (Free tier - no setup required)
2. **Ollama** (Local installation - completely free)
3. **Enhanced Rule-based Fallback** (Always available)

## Setup Options

### Option 1: Hugging Face API (Easiest)
The extension works out-of-the-box with Hugging Face's free inference API.

**Features:**
- No setup required
- Uses Microsoft DialoGPT and Google Flan-T5 models
- Free tier with rate limits
- Automatic retry mechanism

**To get better performance:**
1. Sign up at https://huggingface.co/
2. Get your API token from https://huggingface.co/settings/tokens
3. Edit `llm-service.js` and add your token:
   ```javascript
   'Authorization': 'Bearer YOUR_HF_TOKEN'
   ```

### Option 2: Ollama (Best Performance)
For the best experience, install Ollama locally for unlimited, fast responses.

**Installation Steps:**

1. **Download Ollama:**
   - Windows: Download from https://ollama.ai/download
   - Run the installer and follow the setup wizard

2. **Install a Model:**
   Open Command Prompt or PowerShell and run:
   ```bash
   ollama pull llama2
   # OR for a smaller, faster model:
   ollama pull mistral
   # OR for coding tasks:
   ollama pull codellama
   ```

3. **Start Ollama:**
   ```bash
   ollama serve
   ```
   This runs on http://localhost:11434

4. **Verify Installation:**
   ```bash
   ollama list
   ```
   Should show your installed models.

### Option 3: OpenAI API (Premium)
If you have an OpenAI API key, you can modify the LLM service:

1. Edit `llm-service.js`
2. Add OpenAI endpoint:
   ```javascript
   this.openaiEndpoint = 'https://api.openai.com/v1/chat/completions';
   this.openaiKey = 'YOUR_OPENAI_API_KEY';
   ```

## LangChain-Inspired Features

The extension implements LangChain-like functionality:

### Document Processing
- **Text Chunking**: Splits page content into manageable chunks
- **Overlap Strategy**: Maintains context between chunks
- **Relevance Scoring**: Finds most relevant content for questions

### Retrieval-Augmented Generation (RAG)
- **Context Extraction**: Identifies relevant page sections
- **Question Matching**: Scores content based on question relevance
- **Response Generation**: Uses retrieved context for accurate answers

### Advanced NLP Processing
- **Named Entity Recognition**: Identifies people, places, dates
- **Keyword Extraction**: Finds important terms and phrases
- **Sentiment Analysis**: Understands content tone
- **Text Summarization**: Creates concise page summaries

## Usage Examples

### Before (Rule-based):
**Q:** "When was OpenAI founded?"
**A:** "I'd be happy to help you understand this page better! Request information about page sections."

### After (LLM-powered):
**Q:** "When was OpenAI founded?"
**A:** "Based on the page content: OpenAI was founded in December 2015 by Sam Altman, Elon Musk, and others as a non-profit AI research company."

## Model Comparison

| Model | Speed | Quality | Cost | Setup |
|-------|-------|---------|------|-------|
| Hugging Face Free | Medium | Good | Free* | None |
| Ollama Llama2 | Fast | Excellent | Free | Easy |
| Ollama Mistral | Very Fast | Good | Free | Easy |
| OpenAI GPT-3.5 | Fast | Excellent | Paid | API Key |
| Rule-based Fallback | Instant | Basic | Free | None |

*Free tier has rate limits

## Troubleshooting

### Hugging Face Issues:
- **503 Error**: Model is loading, wait and retry (automatic)
- **Rate Limited**: Wait a few minutes or upgrade to paid tier
- **No Response**: Check internet connection

### Ollama Issues:
- **Connection Failed**: Make sure `ollama serve` is running
- **Model Not Found**: Run `ollama pull <model-name>`
- **Slow Responses**: Try a smaller model like `mistral`

### General Issues:
- **Generic Responses**: Fallback mode is active, check LLM configuration
- **No Context**: Extension can't access page content, try refreshing
- **Extension Errors**: Check browser console for detailed error messages

## Configuration

Edit `llm-service.js` to customize:

```javascript
// Model preferences (try in order)
this.models = ['llama2', 'mistral', 'codellama'];

// Response settings
this.maxTokens = 200;
this.temperature = 0.7;

// Retry settings
this.maxRetries = 3;
this.retryDelay = 1000;
```

## Privacy & Security

- **Local Processing**: Ollama runs entirely on your machine
- **Hugging Face**: Data sent to HF servers (check their privacy policy)
- **No Data Storage**: Extension doesn't store your conversations externally
- **Page Content**: Only sent to chosen LLM service for processing

## Performance Tips

1. **Use Ollama** for best performance and privacy
2. **Choose Smaller Models** (mistral) for faster responses
3. **Limit Context** - extension automatically limits content to 1000 chars
4. **Enable Caching** in browser for faster popup loading

---

**Need Help?** Check the browser console (F12) for detailed error messages and debugging information.
