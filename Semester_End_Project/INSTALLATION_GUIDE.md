# WebPage AI Chatbot - Installation & Usage Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Creating Extension Icons](#creating-extension-icons)
4. [Loading the Extension](#loading-the-extension)
5. [Usage Guide](#usage-guide)
6. [Features Overview](#features-overview)
7. [Troubleshooting](#troubleshooting)
8. [Development Setup](#development-setup)

## 🔧 Prerequisites

Before installing the WebPage AI Chatbot extension, ensure you have:

- **Google Chrome Browser** (version 88 or higher)
- **Developer Mode** enabled in Chrome Extensions
- **Basic understanding** of Chrome extensions (helpful but not required)

## 📦 Installation Steps

### Step 1: Download the Extension Files

1. Navigate to the project folder:
   ```
   C:\Users\Hp\OneDrive\desktop\UAJK\Six Semester\02- NLP\Natural-Language-Processing-SE-3213\Semester_End_Project\WebPageChatbot_Extension
   ```

2. Ensure all required files are present:
   - `manifest.json`
   - `popup.html`
   - `popup.js`
   - `content.js`
   - `background.js`
   - `welcome.html`
   - `icons/` folder (with icon files)

### Step 2: Create Extension Icons (Optional)

If you want custom icons, create PNG files with the following dimensions:
- `icon16.png` - 16x16 pixels
- `icon32.png` - 32x32 pixels  
- `icon48.png` - 48x48 pixels
- `icon128.png` - 128x128 pixels

Place these files in the `icons/` folder. If you don't have custom icons, the extension will use default Chrome icons.

### Step 3: Load the Extension in Chrome

1. **Open Chrome Extensions Page**:
   - Type `chrome://extensions/` in the address bar
   - Or go to Chrome Menu → More Tools → Extensions

2. **Enable Developer Mode**:
   - Toggle the "Developer mode" switch in the top-right corner

3. **Load Unpacked Extension**:
   - Click "Load unpacked" button
   - Navigate to and select the `WebPageChatbot_Extension` folder
   - Click "Select Folder"

4. **Verify Installation**:
   - You should see "WebPage AI Chatbot" in your extensions list
   - The extension icon should appear in your Chrome toolbar
   - A welcome page should open automatically

## 🎯 Usage Guide

### Basic Usage

1. **Navigate to Any Webpage**:
   - Visit any website (e.g., news articles, blogs, documentation)
   - The extension automatically analyzes the page content

2. **Access the Chatbot**:
   - **Method 1**: Click the extension icon in the toolbar
   - **Method 2**: Look for the floating chat indicator (🤖) on the page
   - **Method 3**: Right-click on the page → "Chat about this page"

3. **Start Chatting**:
   - Type your question in the chat input
   - Press Enter or click the send button
   - Get intelligent responses about the webpage content

### Quick Action Examples

Try these sample questions:
- "What is this page about?"
- "Summarize the main content"
- "What are the key topics discussed?"
- "Show me important links"
- "What's the tone of this content?"
- "How difficult is this to read?"

### Advanced Features

#### 1. **Selected Text Analysis**
- Highlight any text on the webpage
- Right-click → "Ask AI about selected text"
- Get specific analysis of the selected content

#### 2. **Embedded Chat Widget**
- Click "Open Chat Widget on Page" in the popup
- A floating chat window appears on the webpage
- Chat without leaving the page context

#### 3. **Context Menus**
- Right-click anywhere on a page
- Select "Chat about this page" for instant access

#### 4. **Persistent Chat History**
- Conversations are saved per webpage
- Return to the same page to continue previous chats
- History is automatically managed

## 🌟 Features Overview

### NLP Capabilities
- **Text Preprocessing**: Automatic content cleaning and normalization
- **Sentiment Analysis**: Understands positive, negative, or neutral tone
- **Keyword Extraction**: Identifies important terms and topics
- **Text Summarization**: Generates concise content summaries
- **Entity Recognition**: Finds emails, URLs, phone numbers, dates
- **Readability Analysis**: Calculates content complexity levels

### User Interface
- **Popup Chat**: Clean, modern interface in the extension popup
- **Floating Widget**: Embedded chat window on webpages
- **Quick Actions**: Pre-defined question buttons for common queries
- **Visual Indicators**: Subtle notifications when chatbot is available

### Technical Features
- **Real-time Analysis**: Instant processing of webpage content
- **Context Awareness**: Responses tailored to specific page content
- **Multi-modal Responses**: Text, links, and structured information
- **Background Processing**: Efficient analysis without slowing browsing

## 🔍 Example Conversations

### Example 1: News Article Analysis
```
User: "What is this article about?"
Bot: "This article is about recent developments in artificial intelligence research, focusing on new breakthrough techniques in natural language processing and their potential applications in various industries."

User: "What's the main topic?"
Bot: "The main topics include: artificial intelligence, machine learning, natural language processing, research developments, and industry applications."

User: "What's the tone of the article?"
Bot: "The overall tone is positive with optimistic language about technological advancement and future possibilities."
```

### Example 2: Technical Documentation
```
User: "How difficult is this to read?"
Bot: "The readability level is fairly difficult (Flesch score: 45.2). The content uses technical terminology and complex sentence structures typical of professional documentation."

User: "Show me the key sections"
Bot: "The main sections are: Installation Guide, API Reference, Configuration Options, Troubleshooting, and Examples."

User: "Are there any contact details?"
Bot: "I found these contact details: support@example.com, and the documentation link https://docs.example.com/support"
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Extension Not Loading
**Problem**: Extension doesn't appear after installation
**Solutions**:
- Ensure Developer Mode is enabled
- Check that all required files are in the folder
- Reload the extension from chrome://extensions/
- Check the Console for error messages

#### Chat Not Responding
**Problem**: Chatbot doesn't respond to queries
**Solutions**:
- Refresh the webpage and try again
- Check if the page has loaded completely
- Try a different webpage to test functionality
- Disable other extensions that might interfere

#### Floating Indicator Not Appearing
**Problem**: Chat indicator doesn't show on pages
**Solutions**:
- Check if the page allows content scripts
- Ensure the page is not a chrome:// internal page
- Refresh the page after enabling the extension
- Check if the page has restrictive Content Security Policy

#### Performance Issues
**Problem**: Browser becomes slow with extension
**Solutions**:
- The extension is optimized for performance
- Try disabling other resource-heavy extensions
- Clear browser cache and restart Chrome
- Check available system memory

### Error Messages

#### "Cannot access page content"
- The page might be restricted (chrome://, extension pages)
- Try the extension on regular websites (news, blogs, etc.)

#### "Analysis failed"
- The page content might be too short or empty
- Ensure the page has loaded completely
- Try refreshing the page

### Browser Permissions

The extension requires these permissions:
- **activeTab**: To analyze current webpage content
- **scripting**: To inject content scripts for analysis
- **storage**: To save chat history and preferences
- **host_permissions**: To work on all websites

## 🔧 Development Setup

### For Developers and Advanced Users

#### Project Structure
```
WebPageChatbot_Extension/
├── manifest.json          # Extension configuration
├── popup.html             # Main user interface
├── popup.js               # Popup logic and chat handling
├── content.js             # Content analysis and extraction
├── background.js          # Background processing and events
├── welcome.html           # First-time user welcome page
├── icons/                 # Extension icons
└── README.md              # Documentation
```

#### Key Components

1. **manifest.json**: Defines extension permissions and structure
2. **popup.js**: Handles user interactions and chat logic
3. **content.js**: Extracts and analyzes webpage content
4. **background.js**: Manages extension lifecycle and advanced processing

#### Customization Options

**Modify Response Templates** (in popup.js):
```javascript
// Add custom response patterns
if (questionLower.includes('your-keyword')) {
    return 'Your custom response logic here';
}
```

**Adjust Analysis Parameters** (in content.js):
```javascript
// Modify content extraction limits
const MAX_CONTENT_LENGTH = 10000; // Adjust as needed
const MAX_LINKS = 50; // Change link extraction limit
```

**Enhance NLP Features** (in background.js):
```javascript
// Add new analysis features
analyzeCustomFeature(content) {
    // Your custom analysis logic
}
```

### Testing the Extension

1. **Load in Developer Mode**: Follow installation steps above
2. **Test on Various Sites**: Try different types of websites
3. **Check Console**: Open DevTools to see any error messages
4. **Monitor Performance**: Use Chrome's Task Manager to check resource usage

### Debugging Tips

- Use `console.log()` statements in the JavaScript files
- Check the Extension's background page console
- Monitor network requests in DevTools
- Test with different webpage types and content lengths

## 📚 Additional Resources

- [Chrome Extension Documentation](https://developer.chrome.com/docs/extensions/)
- [Natural Language Processing Concepts](https://en.wikipedia.org/wiki/Natural_language_processing)
- [JavaScript Web APIs](https://developer.mozilla.org/en-US/docs/Web/API)

## 🎓 Educational Context

This extension was developed as part of the Natural Language Processing (SE-3213) course at the University of Azad Jammu & Kashmir. It demonstrates practical application of NLP techniques including:

- Text preprocessing and tokenization
- Sentiment analysis algorithms
- Extractive text summarization
- Named entity recognition
- Readability analysis
- Interactive natural language interfaces

## 🤝 Support and Feedback

For questions, issues, or suggestions:
1. Check this troubleshooting guide first
2. Review the project README.md file
3. Consult your course instructor or teaching assistants
4. Collaborate with classmates on understanding the implementation

---

**Happy Chatting! 🤖💬**

*Remember: This extension works best on content-rich websites like news articles, blogs, documentation, and educational resources.*
