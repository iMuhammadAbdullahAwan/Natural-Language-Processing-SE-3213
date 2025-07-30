// content.js - Runs on every webpage to enable chatbot functionality
class WebPageChatbot {
  constructor() {
    this.pageContent = null;
    this.initialized = false;
    this.init();
  }

  init() {
    if (this.initialized) return;

    // Wait for page to load completely
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => this.setup());
    } else {
      this.setup();
    }
  }

  setup() {
    this.extractPageContent();
    this.setupMessageListener();
    this.initialized = true;

    // Add subtle indicator that chatbot is available
    this.addChatbotIndicator();
  }

  extractPageContent() {
    try {
      this.pageContent = {
        title: document.title,
        url: window.location.href,
        domain: window.location.hostname,
        text: this.getMainContent(),
        headings: this.getHeadings(),
        links: this.getLinks(),
        images: this.getImages(),
        metadata: this.getMetadata(),
        lastUpdated: new Date().toISOString(),
      };

      // Store in extension storage for popup access
      this.storePageContent();
    } catch (error) {
      console.error("Error extracting page content:", error);
    }
  }

  getMainContent() {
    // Try to get main content, excluding navigation, ads, etc.
    const contentSelectors = [
      "main",
      "article",
      '[role="main"]',
      ".content",
      ".main-content",
      "#content",
      "#main",
    ];

    let mainContent = "";

    for (const selector of contentSelectors) {
      const element = document.querySelector(selector);
      if (element) {
        mainContent = element.innerText || element.textContent || "";
        break;
      }
    }

    // Fallback to body content if no main content found
    if (!mainContent) {
      mainContent = document.body.innerText || document.body.textContent || "";
    }

    // Clean and limit content
    return this.cleanText(mainContent).substring(0, 10000);
  }

  cleanText(text) {
    return text.replace(/\s+/g, " ").replace(/\n+/g, " ").trim();
  }

  getHeadings() {
    return Array.from(document.querySelectorAll("h1, h2, h3, h4, h5, h6"))
      .map((h) => ({
        level: parseInt(h.tagName.charAt(1)),
        text: h.textContent.trim(),
      }))
      .filter((h) => h.text.length > 0)
      .slice(0, 20);
  }

  getLinks() {
    return Array.from(document.querySelectorAll("a[href]"))
      .map((a) => ({
        text: a.textContent.trim(),
        href: a.href,
        internal: a.href.includes(window.location.hostname),
      }))
      .filter((link) => link.text.length > 0)
      .slice(0, 50);
  }

  getImages() {
    return Array.from(document.querySelectorAll("img"))
      .map((img) => ({
        alt: img.alt || "",
        src: img.src,
        title: img.title || "",
      }))
      .filter((img) => img.alt || img.title)
      .slice(0, 20);
  }

  getMetadata() {
    const metadata = {};

    // Get meta tags
    const metaTags = document.querySelectorAll("meta");
    metaTags.forEach((meta) => {
      const name = meta.getAttribute("name") || meta.getAttribute("property");
      const content = meta.getAttribute("content");
      if (name && content) {
        metadata[name] = content;
      }
    });

    // Get structured data
    const jsonLdScripts = document.querySelectorAll(
      'script[type="application/ld+json"]'
    );
    const structuredData = [];
    jsonLdScripts.forEach((script) => {
      try {
        const data = JSON.parse(script.textContent);
        structuredData.push(data);
      } catch (e) {
        // Ignore malformed JSON-LD
      }
    });

    metadata.structuredData = structuredData;
    return metadata;
  }

  async storePageContent() {
    try {
      const key = `pageContent_${window.location.href}`;
      await chrome.storage.local.set({ [key]: this.pageContent });
    } catch (error) {
      console.error("Error storing page content:", error);
    }
  }

  setupMessageListener() {
    // Listen for messages from popup or background script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.type === "getPageContent") {
        sendResponse(this.pageContent);
      } else if (message.type === "analyzeContent") {
        const analysis = this.analyzeContent(message.query);
        sendResponse(analysis);
      } else if (message.type === "searchContent") {
        const results = this.searchInContent(message.query);
        sendResponse(results);
      }
    });
  }

  analyzeContent(query) {
    if (!this.pageContent) return null;

    const analysis = {
      relevantSections: this.findRelevantSections(query),
      keyTerms: this.extractKeyTerms(query),
      summary: this.generateSummary(query),
      relatedLinks: this.findRelatedLinks(query),
    };

    return analysis;
  }

  findRelevantSections(query) {
    const queryTerms = query
      .toLowerCase()
      .split(" ")
      .filter((term) => term.length > 2);
    const text = this.pageContent.text.toLowerCase();
    const sentences = text.split(/[.!?]+/);

    const relevantSentences = sentences
      .map((sentence, index) => {
        const relevanceScore = queryTerms.reduce((score, term) => {
          return score + (sentence.includes(term) ? 1 : 0);
        }, 0);
        return { sentence: sentence.trim(), score: relevanceScore, index };
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    return relevantSentences.map((item) => item.sentence);
  }

  extractKeyTerms(query) {
    const queryTerms = query.toLowerCase().split(" ");
    const text = this.pageContent.text.toLowerCase();

    // Simple term frequency analysis
    const words = text.match(/\b\w{4,}\b/g) || [];
    const frequency = {};

    words.forEach((word) => {
      if (!this.isStopWord(word)) {
        frequency[word] = (frequency[word] || 0) + 1;
      }
    });

    // Sort by frequency and relevance to query
    const sortedTerms = Object.entries(frequency)
      .map(([term, freq]) => ({
        term,
        frequency: freq,
        relevance: queryTerms.some((qterm) => term.includes(qterm)) ? 2 : 1,
      }))
      .sort((a, b) => b.frequency * b.relevance - a.frequency * a.relevance)
      .slice(0, 10)
      .map((item) => item.term);

    return sortedTerms;
  }

  isStopWord(word) {
    const stopWords = new Set([
      "the",
      "a",
      "an",
      "and",
      "or",
      "but",
      "in",
      "on",
      "at",
      "to",
      "for",
      "of",
      "with",
      "by",
      "this",
      "that",
      "these",
      "those",
      "is",
      "are",
      "was",
      "were",
      "be",
      "been",
      "being",
      "have",
      "has",
      "had",
      "do",
      "does",
      "did",
      "will",
      "would",
      "could",
      "should",
      "may",
      "might",
      "must",
      "shall",
      "can",
      "cannot",
      "could",
      "would",
    ]);
    return stopWords.has(word.toLowerCase());
  }

  generateSummary(query) {
    const relevantSections = this.findRelevantSections(query);
    if (relevantSections.length === 0) {
      return (
        this.pageContent.metadata.description ||
        this.pageContent.text.substring(0, 200) + "..."
      );
    }

    return relevantSections.slice(0, 2).join(" ");
  }

  findRelatedLinks(query) {
    const queryTerms = query.toLowerCase().split(" ");

    return this.pageContent.links
      .filter((link) => {
        const linkText = link.text.toLowerCase();
        return queryTerms.some((term) => linkText.includes(term));
      })
      .slice(0, 5);
  }

  searchInContent(query) {
    const results = {
      directMatches: [],
      fuzzyMatches: [],
      relatedHeadings: [],
    };

    const queryLower = query.toLowerCase();
    const text = this.pageContent.text.toLowerCase();

    // Direct matches
    const sentences = text.split(/[.!?]+/);
    sentences.forEach((sentence, index) => {
      if (sentence.includes(queryLower)) {
        results.directMatches.push({
          text: sentence.trim(),
          position: index,
        });
      }
    });

    // Related headings
    this.pageContent.headings.forEach((heading) => {
      if (heading.text.toLowerCase().includes(queryLower)) {
        results.relatedHeadings.push(heading);
      }
    });

    return results;
  }

  addChatbotIndicator() {
    // Add a subtle floating indicator
    const indicator = document.createElement("div");
    indicator.id = "webpage-chatbot-indicator";
    indicator.innerHTML = "🤖";
    indicator.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            cursor: pointer;
            z-index: 9999;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            opacity: 0.8;
        `;

    indicator.addEventListener("mouseenter", () => {
      indicator.style.transform = "scale(1.1)";
      indicator.style.opacity = "1";
    });

    indicator.addEventListener("mouseleave", () => {
      indicator.style.transform = "scale(1)";
      indicator.style.opacity = "0.8";
    });

    indicator.addEventListener("click", () => {
      this.openChatWidget();
    });

    // Add tooltip
    indicator.title = "Click to chat about this webpage";

    document.body.appendChild(indicator);
  }

  openChatWidget() {
    // Remove existing widget if present
    const existingWidget = document.getElementById("webpage-chatbot-widget");
    if (existingWidget) {
      existingWidget.remove();
      return;
    }

    // Create embedded chat widget
    const widget = document.createElement("div");
    widget.id = "webpage-chatbot-widget";
    widget.innerHTML = `
            <div style="
                position: fixed;
                bottom: 80px;
                right: 20px;
                width: 350px;
                height: 450px;
                background: white;
                border: none;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                z-index: 10000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            ">
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="font-weight: bold;">🤖 WebPage Chat</div>
                        <div style="font-size: 12px; opacity: 0.9;">${this.pageContent.domain}</div>
                    </div>
                    <button onclick="this.closest('#webpage-chatbot-widget').remove()" style="
                        background: rgba(255,255,255,0.2);
                        border: none;
                        color: white;
                        font-size: 18px;
                        cursor: pointer;
                        padding: 5px 8px;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">×</button>
                </div>
                <div style="
                    flex: 1;
                    padding: 20px;
                    overflow-y: auto;
                    background: #f8f9fa;
                " id="widget-chat-container">
                    <div style="
                        background: #e3f2fd;
                        padding: 15px;
                        border-radius: 15px;
                        margin-bottom: 15px;
                        font-size: 14px;
                        border-left: 4px solid #2196f3;
                    ">
                        <strong>👋 Welcome!</strong><br>
                        I can help you understand this webpage. Ask me about:
                        <ul style="margin: 10px 0 0 0; padding-left: 20px;">
                            <li>Page summary</li>
                            <li>Key topics</li>
                            <li>Specific content</li>
                            <li>Links and sections</li>
                        </ul>
                    </div>
                </div>
                <div style="
                    padding: 20px;
                    border-top: 1px solid #eee;
                    background: white;
                ">
                    <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                        <input type="text" placeholder="Ask about this webpage..." style="
                            flex: 1;
                            padding: 12px 16px;
                            border: 2px solid #e0e0e0;
                            border-radius: 25px;
                            outline: none;
                            font-size: 14px;
                            transition: border-color 0.3s;
                        " id="widget-input-field" maxlength="500">
                        <button style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            border: none;
                            border-radius: 50%;
                            width: 45px;
                            height: 45px;
                            cursor: pointer;
                            font-size: 16px;
                            transition: transform 0.2s;
                        " onclick="window.sendWidgetMessage()" id="widget-send-btn">▶</button>
                    </div>
                    <div style="display: flex; gap: 5px; flex-wrap: wrap;">
                        <button onclick="window.sendWidgetMessage('What is this page about?')" style="
                            background: #f0f0f0;
                            border: 1px solid #ddd;
                            padding: 5px 10px;
                            border-radius: 15px;
                            font-size: 12px;
                            cursor: pointer;
                            transition: background 0.2s;
                        ">Summary</button>
                        <button onclick="window.sendWidgetMessage('What are the main topics?')" style="
                            background: #f0f0f0;
                            border: 1px solid #ddd;
                            padding: 5px 10px;
                            border-radius: 15px;
                            font-size: 12px;
                            cursor: pointer;
                            transition: background 0.2s;
                        ">Topics</button>
                        <button onclick="window.sendWidgetMessage('Show me the key links')" style="
                            background: #f0f0f0;
                            border: 1px solid #ddd;
                            padding: 5px 10px;
                            border-radius: 15px;
                            font-size: 12px;
                            cursor: pointer;
                            transition: background 0.2s;
                        ">Links</button>
                    </div>
                </div>
            </div>
        `;

    document.body.appendChild(widget);

    // Focus on input field
    setTimeout(() => {
      document.getElementById("widget-input-field").focus();
    }, 100);

    // Add enhanced widget functionality
    this.setupWidgetInteraction();
  }

  setupWidgetInteraction() {
    const input = document.getElementById("widget-input-field");
    const chatContainer = document.getElementById("widget-chat-container");

    // Enhanced send message function
    window.sendWidgetMessage = (predefinedMessage) => {
      const message = predefinedMessage || input.value.trim();
      if (!message) return;

      this.addWidgetMessage(message, "user");
      input.value = "";

      // Show typing indicator
      this.showTypingIndicator();

      // Process message with actual content analysis
      setTimeout(() => {
        this.hideTypingIndicator();
        const response = this.generateIntelligentResponse(message);
        this.addWidgetMessage(response, "bot");
      }, 1500);
    };

    // Enter key support
    input.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        window.sendWidgetMessage();
      }
    });

    // Focus styling
    input.addEventListener("focus", () => {
      input.style.borderColor = "#667eea";
    });

    input.addEventListener("blur", () => {
      input.style.borderColor = "#e0e0e0";
    });
  }

  addWidgetMessage(message, type) {
    const chatContainer = document.getElementById("widget-chat-container");
    const messageDiv = document.createElement("div");

    const isUser = type === "user";
    messageDiv.style.cssText = `
            background: ${
              isUser
                ? "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                : "#ffffff"
            };
            color: ${isUser ? "white" : "#333"};
            padding: 12px 16px;
            border-radius: ${
              isUser ? "20px 20px 5px 20px" : "20px 20px 20px 5px"
            };
            margin: 8px 0;
            margin-${isUser ? "left" : "right"}: ${isUser ? "50px" : "20px"};
            font-size: 14px;
            line-height: 1.4;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            word-wrap: break-word;
            animation: slideIn 0.3s ease-out;
        `;

    messageDiv.textContent = message;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Add slide-in animation
    if (!document.getElementById("widget-animations")) {
      const style = document.createElement("style");
      style.id = "widget-animations";
      style.textContent = `
                @keyframes slideIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            `;
      document.head.appendChild(style);
    }
  }

  showTypingIndicator() {
    const chatContainer = document.getElementById("widget-chat-container");
    const typingDiv = document.createElement("div");
    typingDiv.id = "typing-indicator";
    typingDiv.style.cssText = `
            background: #f0f0f0;
            padding: 12px 16px;
            border-radius: 20px 20px 20px 5px;
            margin: 8px 20px 8px 0;
            font-size: 14px;
            color: #666;
            animation: pulse 1.5s infinite;
        `;
    typingDiv.innerHTML = "🤔 Thinking...";

    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Add pulse animation
    if (!document.getElementById("pulse-animation")) {
      const style = document.createElement("style");
      style.id = "pulse-animation";
      style.textContent = `
                @keyframes pulse {
                    0%, 100% { opacity: 0.6; }
                    50% { opacity: 1; }
                }
            `;
      document.head.appendChild(style);
    }
  }

  hideTypingIndicator() {
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }

  generateIntelligentResponse(question) {
    const analysis = this.analyzeContent(question);
    const questionLower = question.toLowerCase();

    // Enhanced response generation based on content analysis
    if (
      questionLower.includes("what") &&
      (questionLower.includes("about") || questionLower.includes("page"))
    ) {
      return `This page is about "${this.pageContent.title}". ${
        analysis.summary ||
        "It contains information and resources related to the topic."
      }`;
    }

    if (
      questionLower.includes("summary") ||
      questionLower.includes("summarize")
    ) {
      const keyPoints = analysis.relevantSections.slice(0, 2).join(" ");
      return (
        keyPoints ||
        `This is "${this.pageContent.title}" - ${
          this.pageContent.metadata.description ||
          "a webpage with various content and information."
        }`
      );
    }

    if (questionLower.includes("topic") || questionLower.includes("subject")) {
      const topics = analysis.keyTerms.slice(0, 5).join(", ");
      return `The main topics discussed on this page include: ${topics}. These seem to be the key areas of focus based on the content.`;
    }

    if (questionLower.includes("link") || questionLower.includes("url")) {
      const relatedLinks = analysis.relatedLinks;
      if (relatedLinks.length > 0) {
        return `I found these relevant links: ${relatedLinks
          .map((l) => l.text)
          .join(", ")}. These appear to be related to your query.`;
      }
      return `This page has ${this.pageContent.links.length} links total. Most seem to be navigation or reference links within the site.`;
    }

    if (
      questionLower.includes("heading") ||
      questionLower.includes("section")
    ) {
      const headings = this.pageContent.headings
        .slice(0, 5)
        .map((h) => h.text)
        .join(", ");
      return headings
        ? `The main sections are: ${headings}`
        : "I couldn't identify clear section headings on this page.";
    }

    // Search for specific content
    const searchResults = this.searchInContent(question);
    if (searchResults.directMatches.length > 0) {
      return `I found this relevant information: "${searchResults.directMatches[0].text.substring(
        0,
        200
      )}..."`;
    }

    if (searchResults.relatedHeadings.length > 0) {
      return `This seems related to the "${searchResults.relatedHeadings[0].text}" section of the page.`;
    }

    // If no specific match, provide general help
    const suggestions = [
      "Try asking about the page summary",
      "Ask about specific topics or keywords",
      "Request information about page sections",
      "Ask about links or resources mentioned",
    ];

    return `I'd be happy to help you understand this page better! ${
      suggestions[Math.floor(Math.random() * suggestions.length)]
    }. The page covers "${this.pageContent.title}" and contains ${Math.floor(
      this.pageContent.text.length / 100
    )} sections of content.`;
  }
}

// Initialize the chatbot when content script loads
if (typeof window !== "undefined") {
  let webPageChatbot;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      webPageChatbot = new WebPageChatbot();
    });
  } else {
    webPageChatbot = new WebPageChatbot();
  }
}
