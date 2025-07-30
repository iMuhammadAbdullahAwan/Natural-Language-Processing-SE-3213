# WebPage AI Chatbot Chrome Extension

A sophisticated Chrome extension that provides real-time AI-powered chatbot functionality for any webpage. Users can ask questions and get intelligent responses about the content of the currently opened webpage using advanced NLP techniques.

## 🚀 Features

### Core Functionality
- **Real-time Page Analysis**: Automatically analyzes webpage content when loaded
- **Intelligent Chatbot**: AI-powered responses about webpage content
- **Multiple Chat Interfaces**: Both popup and embedded widget options
- **Content Extraction**: Smart extraction of text, headings, links, and metadata
- **Context-Aware Responses**: Tailored answers based on page content

### NLP & AI Features
- **Text Preprocessing**: Advanced text cleaning and normalization
- **Keyword Extraction**: Automatic identification of key topics and terms
- **Sentiment Analysis**: Analyzes the tone and sentiment of webpage content
- **Summarization**: Generates concise summaries of webpage content
- **Entity Recognition**: Identifies dates, emails, URLs, and phone numbers
- **Readability Analysis**: Calculates content complexity and reading level
- **Topic Modeling**: Extracts main themes and topics from content

### User Experience
- **Floating Chat Indicator**: Subtle indicator showing chatbot availability
- **Context Menus**: Right-click options for selected text analysis
- **Chat History**: Persistent conversation history per webpage
- **Quick Actions**: Pre-defined question buttons for common queries
- **Responsive Design**: Works seamlessly across different websites

## 📁 Project Structure

```
WebPageChatbot_Extension/
├── manifest.json          # Extension configuration
├── popup.html             # Extension popup interface
├── popup.js               # Popup functionality and chat logic
├── content.js             # Content script for webpage integration
├── background.js          # Service worker for extension management
├── icons/                 # Extension icons (16px, 32px, 48px, 128px)
├── README.md              # This file
└── docs/                  # Additional documentation
    ├── installation.md    # Installation instructions
    ├── usage.md          # Usage guide
    └── development.md    # Development setup
```

## 🛠️ Technologies Used

### Frontend
- **HTML5**: Modern semantic markup
- **CSS3**: Advanced styling with gradients and animations
- **JavaScript ES6+**: Modern JavaScript features
- **Chrome Extension APIs**: Chrome runtime, storage, scripting APIs

### NLP & AI Techniques
- **Text Processing**: Tokenization, normalization, stop-word removal
- **Frequency Analysis**: TF-IDF for keyword extraction
- **Sentiment Analysis**: Rule-based sentiment classification
- **Extractive Summarization**: Sentence scoring and selection
- **Named Entity Recognition**: Pattern-based entity extraction
- **Readability Metrics**: Flesch Reading Ease calculation

### Architecture
- **Service Worker**: Background processing and event handling
- **Content Scripts**: DOM manipulation and content extraction
- **Message Passing**: Communication between extension components
- **Local Storage**: Persistent data storage

## 🔧 Installation & Setup

### 1. Download or Clone
```bash
git clone [repository-url]
cd WebPageChatbot_Extension
```

### 2. Load in Chrome
1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (top right toggle)
3. Click "Load unpacked"
4. Select the `WebPageChatbot_Extension` folder
5. The extension icon should appear in your toolbar

### 3. Generate Icons (Optional)
Create PNG icons in the `icons/` folder:
- `icon16.png` (16x16 pixels)
- `icon32.png` (32x32 pixels)
- `icon48.png` (48x48 pixels)
- `icon128.png` (128x128 pixels)

## 📖 Usage Guide

### Basic Usage
1. **Navigate to any webpage**
2. **Click the extension icon** or look for the floating chat indicator
3. **Start chatting** about the webpage content
4. **Use quick action buttons** for common questions

### Available Commands
- **"What is this page about?"** - Get a general summary
- **"Summarize this content"** - Get a concise overview
- **"What are the main topics?"** - Extract key themes
- **"Show me the links"** - List important links
- **"Analyze the tone"** - Get sentiment analysis

### Advanced Features
- **Right-click selected text** → "Ask AI about selected text"
- **Toggle widget** - Embedded chat window on the page
- **Persistent history** - Conversations saved per webpage
- **Context awareness** - Responses tailored to specific content

## 🧠 NLP Implementation Details

### Text Preprocessing Pipeline
1. **Content Extraction**: DOM traversal to extract meaningful content
2. **Text Cleaning**: Remove extra whitespace, normalize encoding
3. **Tokenization**: Split text into words and sentences
4. **Stop-word Filtering**: Remove common words for better analysis

### Analysis Features
- **Keyword Frequency**: Statistical analysis of term importance
- **Sentiment Scoring**: Positive/negative word counting
- **Topic Extraction**: Frequency-based topic identification
- **Readability Metrics**: Syllable counting and sentence complexity
- **Entity Recognition**: Pattern matching for structured data

### Response Generation
- **Template-based**: Structured responses for common queries
- **Context-aware**: Responses incorporate page-specific information
- **Multi-modal**: Text, links, and structured data in responses

## 🔮 Future Enhancements

### AI Integration
- [ ] OpenAI GPT API integration
- [ ] Google Gemini API support
- [ ] Custom fine-tuned models
- [ ] Vector embeddings for semantic search

### Advanced NLP
- [ ] Advanced topic modeling (LDA)
- [ ] Transformer-based embeddings
- [ ] Multi-language support
- [ ] Custom entity recognition

### Features
- [ ] Voice interaction
- [ ] Export conversations
- [ ] Collaborative annotations
- [ ] Custom prompt templates

## 🎓 Educational Value

This project demonstrates several key NLP and web development concepts:

### NLP Concepts
- **Text preprocessing and normalization**
- **Statistical text analysis**
- **Sentiment analysis techniques**
- **Information extraction**
- **Automatic summarization**

### Web Development
- **Chrome Extension Architecture**
- **DOM manipulation and content extraction**
- **Asynchronous JavaScript programming**
- **Client-side data storage**
- **User interface design**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Development Team

**Student**: [Your Name]  
**Course**: Natural Language Processing (SE-3213)  
**Semester**: 6th Semester  
**Institution**: University of Azad Jammu & Kashmir  

## 🙏 Acknowledgments

- Course instructor and teaching assistants
- Chrome Extension documentation and community
- Open-source NLP libraries and resources
- Web development community and tutorials

---

**Note**: This is an educational project developed as part of the NLP course curriculum. For production use, consider integrating with professional AI APIs and implementing additional security measures.
