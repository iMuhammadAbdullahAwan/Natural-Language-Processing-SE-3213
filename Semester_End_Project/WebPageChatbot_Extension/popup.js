// popup.js - Handles the extension popup functionality

// Function to extract page content (runs in page context)
function extractPageContent() {
  const content = {
    title: document.title,
    url: window.location.href,
    text: document.body.innerText || document.body.textContent || "",
    headings: Array.from(
      document.querySelectorAll("h1, h2, h3, h4, h5, h6")
    ).map((h) => h.textContent.trim()),
    links: Array.from(document.querySelectorAll("a[href]"))
      .map((a) => ({
        text: a.textContent.trim(),
        href: a.href,
      }))
      .slice(0, 10), // Limit to first 10 links
    images: Array.from(document.querySelectorAll("img[alt]"))
      .map((img) => img.alt)
      .slice(0, 5),
    metaDescription:
      document.querySelector('meta[name="description"]')?.content || "",
  };

  return content;
}

// Function to toggle chat widget (runs in page context)
function toggleWidget() {
  // This function runs in the page context
  const existingWidget = document.getElementById("webpage-chatbot-widget");

  if (existingWidget) {
    existingWidget.remove();
    return;
  }

  // Create widget iframe
  const widget = document.createElement("div");
  widget.id = "webpage-chatbot-widget";
  // Widget implementation continues...
  // (The full widget code would go here)
}

class PopupChatbot {
  constructor() {
    this.llmService = new LLMService();
    this.initializeElements();
    this.setupEventListeners();
    this.loadChatHistory();
    this.updatePageInfo();
  }

  initializeElements() {
    this.chatContainer = document.getElementById("chatContainer");
    this.userInput = document.getElementById("userInput");
    this.sendBtn = document.getElementById("sendBtn");
    this.loading = document.getElementById("loading");
    this.error = document.getElementById("error");
    this.pageStatus = document.getElementById("pageStatus");
    this.toggleWidget = document.getElementById("toggleWidget");
  }

  setupEventListeners() {
    this.sendBtn.addEventListener("click", () => this.sendMessage());
    this.userInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") this.sendMessage();
    });
    this.toggleWidget.addEventListener("click", () => this.togglePageWidget());
  }

  async updatePageInfo() {
    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true,
      });
      if (tab) {
        const domain = new URL(tab.url).hostname;
        this.pageStatus.textContent = `Chatting about: ${domain}`;
      }
    } catch (error) {
      console.error("Error getting page info:", error);
    }
  }

  async sendMessage() {
    const message = this.userInput.value.trim();
    if (!message) return;

    this.addMessage(message, "user");
    this.userInput.value = "";
    this.setLoading(true);
    this.clearError();

    try {
      const response = await this.processMessage(message);
      this.addMessage(response, "bot");
    } catch (error) {
      this.showError("Sorry, I encountered an error. Please try again.");
      console.error("Error processing message:", error);
    } finally {
      this.setLoading(false);
    }
  }

  async processMessage(message) {
    // Get current page content
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });

    // Check if tab URL is accessible
    if (
      tab.url.startsWith("chrome://") ||
      tab.url.startsWith("chrome-extension://")
    ) {
      return "Sorry, I can't analyze this type of page. Please try on a regular website.";
    }

    try {
      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: extractPageContent,
      });

      const pageContent = results[0].result;

      // Process with NLP techniques
      const processedContent = this.preprocessText(pageContent);
      const response = await this.generateResponse(message, processedContent);

      return response;
    } catch (error) {
      console.error("Error executing script:", error);
      return "Sorry, I couldn't analyze this page. Please try refreshing the page or try on a different website.";
    }
  }

  preprocessText(content) {
    // Basic NLP preprocessing
    const text = content.text.substring(0, 5000); // Limit text length

    // Remove extra whitespace and normalize
    const cleaned = text.replace(/\s+/g, " ").trim();

    // Extract key information
    const processed = {
      title: content.title,
      description: content.metaDescription,
      mainText: cleaned,
      headings: content.headings,
      keyPhrases: this.extractKeyPhrases(cleaned),
      summary: this.createSummary(cleaned),
      links: content.links,
      images: content.images,
    };

    return processed;
  }

  extractKeyPhrases(text) {
    // Simple keyword extraction using frequency analysis
    const words = text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((word) => word.length > 3);

    const stopWords = new Set([
      "that",
      "this",
      "with",
      "have",
      "will",
      "from",
      "they",
      "know",
      "want",
      "been",
      "good",
      "much",
      "some",
      "time",
      "very",
      "when",
      "come",
      "here",
      "just",
      "like",
      "long",
      "make",
      "many",
      "over",
      "such",
      "take",
      "than",
      "them",
      "well",
      "were",
    ]);

    const frequency = {};
    words.forEach((word) => {
      if (!stopWords.has(word)) {
        frequency[word] = (frequency[word] || 0) + 1;
      }
    });

    return Object.entries(frequency)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([word]) => word);
  }

  createSummary(text) {
    // Simple extractive summarization
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 20);
    if (sentences.length <= 3) return text;

    // Return first 2 sentences as summary
    return sentences.slice(0, 2).join(". ") + ".";
  }

  async generateResponse(question, processedContent) {
    // Use LLM service for intelligent responses
    try {
      const response = await this.llmService.generateResponse(
        question,
        processedContent,
        {
          title: processedContent.title,
          url: window.location?.href || "current page",
        }
      );
      return response;
    } catch (error) {
      console.error("LLM service error:", error);

      // Fallback to enhanced rule-based if LLM fails
      return this.enhancedFallbackResponse(question, processedContent);
    }
  }

  enhancedFallbackResponse(question, processedContent) {
    const questionLower = question.toLowerCase();
    const fullText = processedContent.mainText;

    // Handle specific date/founding questions
    if (
      questionLower.includes("when") &&
      (questionLower.includes("found") ||
        questionLower.includes("start") ||
        questionLower.includes("establish") ||
        questionLower.includes("creat"))
    ) {
      const dateInfo = this.findDateInformation(fullText, questionLower);
      if (dateInfo) {
        return dateInfo;
      }
    }

    // Handle who questions
    if (
      questionLower.includes("who") &&
      (questionLower.includes("found") ||
        questionLower.includes("ceo") ||
        questionLower.includes("founder") ||
        questionLower.includes("start"))
    ) {
      const personInfo = this.findPersonInformation(fullText, questionLower);
      if (personInfo) {
        return personInfo;
      }
    }

    // Handle location questions
    if (
      questionLower.includes("where") &&
      (questionLower.includes("locat") ||
        questionLower.includes("headquart") ||
        questionLower.includes("base"))
    ) {
      const locationInfo = this.findLocationInformation(
        fullText,
        questionLower
      );
      if (locationInfo) {
        return locationInfo;
      }
    }

    // Enhanced content search for specific questions
    const relevantContent = this.findRelevantContentEnhanced(
      question,
      fullText
    );
    if (relevantContent) {
      return relevantContent;
    }

    // Default response
    return `I'd be happy to help you understand this page better! Ask about specific topics, dates, people, or locations mentioned on "${processedContent.title}".`;
  }

  findDateInformation(text, question) {
    // Look for date patterns in the text
    const datePatterns = [
      /founded?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /established?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /created?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /started?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /launched?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /(?:founded|established|created|started|launched).*?(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})/i,
      /(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})/i,
      /(\d{4})/g, // Fallback for years
    ];

    for (const pattern of datePatterns) {
      const matches = text.match(pattern);
      if (matches) {
        const sentences = text.split(/[.!?]+/);
        const relevantSentence = sentences.find(
          (sentence) =>
            pattern.test(sentence) &&
            (sentence.toLowerCase().includes("found") ||
              sentence.toLowerCase().includes("establish") ||
              sentence.toLowerCase().includes("start") ||
              sentence.toLowerCase().includes("creat") ||
              sentence.toLowerCase().includes("launch"))
        );

        if (relevantSentence) {
          return `Based on the page content: ${relevantSentence.trim()}.`;
        }
      }
    }
    return null;
  }

  findPersonInformation(text, question) {
    // Look for people mentioned with founding/CEO context
    const personPatterns = [
      /founded?\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/i,
      /ceo\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/i,
      /founder\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/i,
      /([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+founded/i,
    ];

    for (const pattern of personPatterns) {
      const matches = text.match(pattern);
      if (matches) {
        const sentences = text.split(/[.!?]+/);
        const relevantSentence = sentences.find((sentence) =>
          pattern.test(sentence)
        );
        if (relevantSentence) {
          return `Based on the page content: ${relevantSentence.trim()}.`;
        }
      }
    }
    return null;
  }

  findLocationInformation(text, question) {
    // Look for location information
    const locationPatterns = [
      /headquarter(?:ed|s)?\s+(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/i,
      /located?\s+(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/i,
      /based?\s+(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/i,
    ];

    for (const pattern of locationPatterns) {
      const matches = text.match(pattern);
      if (matches) {
        const sentences = text.split(/[.!?]+/);
        const relevantSentence = sentences.find((sentence) =>
          pattern.test(sentence)
        );
        if (relevantSentence) {
          return `Based on the page content: ${relevantSentence.trim()}.`;
        }
      }
    }
    return null;
  }

  findRelevantContentEnhanced(question, text) {
    // Enhanced content search with better context matching
    const questionWords = question
      .toLowerCase()
      .split(/\s+/)
      .filter(
        (word) =>
          word.length > 2 &&
          ![
            "the",
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
            "is",
            "are",
            "was",
            "were",
            "when",
            "what",
            "who",
            "where",
            "why",
            "how",
          ].includes(word)
      );

    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 20);

    // Score sentences based on question word matches
    const scoredSentences = sentences.map((sentence) => {
      const sentenceLower = sentence.toLowerCase();
      let score = 0;

      questionWords.forEach((word) => {
        if (sentenceLower.includes(word)) {
          score += 1;
          // Bonus for exact matches
          if (
            sentenceLower.includes(" " + word + " ") ||
            sentenceLower.startsWith(word + " ") ||
            sentenceLower.endsWith(" " + word)
          ) {
            score += 0.5;
          }
        }
      });

      return { sentence: sentence.trim(), score };
    });

    // Sort by score and get the best match
    scoredSentences.sort((a, b) => b.score - a.score);

    if (scoredSentences.length > 0 && scoredSentences[0].score > 0) {
      return `I found this relevant information: "${scoredSentences[0].sentence}."`;
    }

    return null;
  }

  findRelevantContent(searchTerms, text) {
    const sentences = text.split(/[.!?]+/);
    const relevantSentences = sentences.filter((sentence) =>
      searchTerms.some((term) =>
        sentence.toLowerCase().includes(term.toLowerCase())
      )
    );

    if (relevantSentences.length > 0) {
      return relevantSentences[0].trim() + ".";
    }

    return null;
  }

  addMessage(message, type) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = message;
    this.chatContainer.appendChild(messageDiv);
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;

    this.saveChatHistory();
  }

  setLoading(isLoading) {
    this.loading.style.display = isLoading ? "block" : "none";
    this.sendBtn.disabled = isLoading;
    this.userInput.disabled = isLoading;
  }

  showError(message) {
    this.error.textContent = message;
    setTimeout(() => this.clearError(), 5000);
  }

  clearError() {
    this.error.textContent = "";
  }

  async togglePageWidget() {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });

    // Check if tab URL is accessible
    if (
      tab.url.startsWith("chrome://") ||
      tab.url.startsWith("chrome-extension://")
    ) {
      this.showError(
        "Can't open widget on this type of page. Please try on a regular website."
      );
      return;
    }

    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: toggleWidget,
      });
    } catch (error) {
      console.error("Error toggling widget:", error);
      this.showError("Could not open widget on this page.");
    }
  }

  async saveChatHistory() {
    const messages = Array.from(this.chatContainer.children).map((msg) => ({
      text: msg.textContent,
      type: msg.classList.contains("user-message") ? "user" : "bot",
    }));

    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });
    const key = `chat_${tab.url}`;

    chrome.storage.local.set({ [key]: messages });
  }

  async loadChatHistory() {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });
    const key = `chat_${tab.url}`;

    const result = await chrome.storage.local.get(key);
    const messages = result[key] || [];

    // Clear container except welcome message
    this.chatContainer.innerHTML = `
            <div class="message bot-message">
                👋 Hi! I can answer questions about the content of this webpage. What would you like to know?
            </div>
        `;

    // Add saved messages
    messages.slice(1).forEach((msg) => {
      // Skip the welcome message
      if (
        msg.text !==
        "👋 Hi! I can answer questions about the content of this webpage. What would you like to know?"
      ) {
        this.addMessageWithoutSaving(msg.text, msg.type);
      }
    });
  }

  addMessageWithoutSaving(message, type) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = message;
    this.chatContainer.appendChild(messageDiv);
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }
}

// Initialize when popup loads
document.addEventListener("DOMContentLoaded", () => {
  new PopupChatbot();
});

// Standalone function for widget toggle (used in content injection)
function toggleWidget() {
  // This function runs in the page context
  const existingWidget = document.getElementById("webpage-chatbot-widget");

  if (existingWidget) {
    existingWidget.remove();
    return;
  }

  // Create widget iframe
  const widget = document.createElement("div");
  widget.id = "webpage-chatbot-widget";
  widget.innerHTML = `
    <div style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 300px;
        height: 400px;
        background: white;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 10000;
        font-family: Arial, sans-serif;
        display: flex;
        flex-direction: column;
    ">
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px 10px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span>🤖 WebPage Chat</span>
            <button onclick="this.closest('#webpage-chatbot-widget').remove()" style="
                background: none;
                border: none;
                color: white;
                font-size: 18px;
                cursor: pointer;
                padding: 0;
                width: 20px;
                height: 20px;
            ">×</button>
        </div>
        <div style="
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background: #f8f9fa;
        " id="widget-chat">
            <div style="
                background: #e9ecef;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                font-size: 14px;
            ">
                Hi! I'm your webpage assistant. Ask me anything about this page!
            </div>
        </div>
        <div style="
            padding: 15px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        ">
            <input type="text" placeholder="Ask about this page..." style="
                flex: 1;
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 20px;
                outline: none;
                font-size: 14px;
            " id="widget-input">
            <button style="
                background: #007bff;
                color: white;
                border: none;
                border-radius: 50%;
                width: 35px;
                height: 35px;
                cursor: pointer;
                font-size: 14px;
            " onclick="sendWidgetMessage()">▶</button>
        </div>
    </div>
  `;

  document.body.appendChild(widget);

  // Add widget functionality
  window.sendWidgetMessage = function () {
    const input = document.getElementById("widget-input");
    const chat = document.getElementById("widget-chat");
    const message = input.value.trim();

    if (message) {
      // Add user message
      const userMsg = document.createElement("div");
      userMsg.style.cssText = `
        background: #007bff;
        color: white;
        padding: 8px 12px;
        border-radius: 15px;
        margin: 5px 0;
        margin-left: 50px;
        font-size: 14px;
      `;
      userMsg.textContent = message;
      chat.appendChild(userMsg);

      input.value = "";
      chat.scrollTop = chat.scrollHeight;

      // Simulate bot response
      setTimeout(() => {
        const botMsg = document.createElement("div");
        botMsg.style.cssText = `
          background: #e9ecef;
          color: #333;
          padding: 8px 12px;
          border-radius: 15px;
          margin: 5px 0;
          margin-right: 50px;
          font-size: 14px;
        `;
        botMsg.textContent = `I understand you're asking about "${message}". This is a demo response!`;
        chat.appendChild(botMsg);
        chat.scrollTop = chat.scrollHeight;
      }, 1000);
    }
  };

  // Allow Enter key to send message
  document
    .getElementById("widget-input")
    .addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        sendWidgetMessage();
      }
    });
}
