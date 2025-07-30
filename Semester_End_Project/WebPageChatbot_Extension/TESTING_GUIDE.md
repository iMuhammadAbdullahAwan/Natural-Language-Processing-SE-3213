# WebPage AI Chatbot Extension - Testing Guide

## Installation Steps

1. **Open Chrome Browser**
   - Make sure you're using Google Chrome

2. **Access Extensions Page**
   - Go to `chrome://extensions/`
   - Or click the three dots menu → More tools → Extensions

3. **Enable Developer Mode**
   - Toggle the "Developer mode" switch in the top right corner

4. **Load the Extension**
   - Click "Load unpacked" button
   - Navigate to this folder: `C:\Users\Hp\OneDrive\desktop\UAJK\Six Semester\02- NLP\Natural-Language-Processing-SE-3213\Semester_End_Project\WebPageChatbot_Extension`
   - Select the folder and click "Select Folder"

## Testing the Extension

### Step 1: Check Installation
- Look for "WebPage AI Chatbot" in your extensions list
- Make sure it's enabled (toggle should be blue)
- You should see the extension icon in the Chrome toolbar

### Step 2: Test on a Website
- Go to any website (e.g., `https://wikipedia.org`, `https://news.google.com`)
- Click the extension icon in the toolbar
- The popup should open with the chat interface

### Step 3: Test Chat Functionality
- Type a question about the webpage in the chat input
- Press Enter or click Send
- You should see your message and a bot response

### Step 4: Test Widget Toggle
- Click the "Open Chat Widget" button in the popup
- A floating chat widget should appear on the webpage
- Click the × button to close the widget

## Troubleshooting

### If Extension Won't Load:
1. Check that all files are in the correct folder
2. Make sure Developer mode is enabled
3. Try refreshing the extensions page

### If Extension Loads but Doesn't Work:
1. Open Chrome DevTools (F12)
2. Check the Console tab for errors
3. Try reloading the extension in chrome://extensions/

### If Chat Doesn't Work:
1. Make sure you're not on a chrome:// page (these are restricted)
2. Check that the website allows extensions
3. Try a different website like Google or Wikipedia

## Extension Features to Test

✅ **Popup Interface**: Clean chat interface opens when clicking extension icon
✅ **Content Analysis**: Extension analyzes webpage content for context
✅ **NLP Processing**: Uses sentiment analysis, summarization, and keyword extraction
✅ **Widget Mode**: Floating chat widget that can be embedded on any page
✅ **Chat History**: Saves conversation history per website
✅ **Real-time Responses**: Simulated AI responses to user questions

## Error Messages You Might See

- **"Cannot access this page"**: Normal for chrome:// pages
- **"Error extracting content"**: Some websites block content extraction
- **"Could not open widget"**: Widget might be blocked on certain sites

## Success Indicators

✅ Extension icon appears in toolbar
✅ Popup opens with chat interface
✅ Can type messages and see responses
✅ Widget toggle button works
✅ No console errors on regular websites

## Next Steps

Once basic functionality is confirmed:
1. Test on various website types (news, blogs, e-commerce)
2. Try different types of questions
3. Check that NLP analysis is working in the background
4. Verify chat history persistence

---

**Note**: This is a demonstration version. The AI responses are currently simulated. In a production version, you would integrate with actual AI services like OpenAI GPT or Google's Gemini API.
