// background.js - Service worker for the Chrome extension
class ChatbotBackground {
  constructor() {
    this.setupEventListeners();
    this.initializeExtension();
  }

  setupEventListeners() {
    // Handle extension installation
    chrome.runtime.onInstalled.addListener((details) => {
      this.handleInstallation(details);
    });

    // Handle tab updates to refresh page content
    chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
      this.handleTabUpdate(tabId, changeInfo, tab);
    });

    // Handle messages from content scripts and popup
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleMessage(message, sender, sendResponse);
      return true; // Keep the message channel open for async responses
    });

    // Remove the onClicked listener since we have a popup
    // chrome.action.onClicked.addListener((tab) => {
    //     this.handleIconClick(tab);
    // });

    // Handle context menu items (if we add them)
    chrome.contextMenus.onClicked.addListener((info, tab) => {
      this.handleContextMenu(info, tab);
    });
  }
  initializeExtension() {
    // Set up context menus
    chrome.contextMenus.removeAll(() => {
      chrome.contextMenus.create({
        id: "chatbot-analyze-selection",
        title: "Ask AI about selected text",
        contexts: ["selection"],
      });

      chrome.contextMenus.create({
        id: "chatbot-analyze-page",
        title: "Chat about this page",
        contexts: ["page"],
      });
    });

    // Initialize storage with default settings
    this.initializeStorage();
  }

  async initializeStorage() {
    const defaultSettings = {
      chatHistory: {},
      userPreferences: {
        theme: "light",
        responseLength: "medium",
        autoAnalyze: true,
        showIndicator: true,
      },
      apiSettings: {
        useOpenAI: false,
        apiKey: "",
        model: "gpt-3.5-turbo",
      },
    };

    try {
      const stored = await chrome.storage.local.get("chatbotSettings");
      if (!stored.chatbotSettings) {
        await chrome.storage.local.set({ chatbotSettings: defaultSettings });
      }
    } catch (error) {
      console.error("Error initializing storage:", error);
    }
  }

  handleInstallation(details) {
    if (details.reason === "install") {
      // First time installation
      console.log("WebPage AI Chatbot installed successfully!");

      // Open welcome page
      chrome.tabs.create({
        url: chrome.runtime.getURL("welcome.html"),
      });
    } else if (details.reason === "update") {
      // Extension updated
      console.log(
        "WebPage AI Chatbot updated to version",
        chrome.runtime.getManifest().version
      );
    }
  }

  async handleTabUpdate(tabId, changeInfo, tab) {
    // Only process when the page is completely loaded
    if (changeInfo.status === "complete" && tab.url) {
      try {
        // Skip chrome:// and extension pages
        if (
          tab.url.startsWith("chrome://") ||
          tab.url.startsWith("chrome-extension://") ||
          tab.url.startsWith("moz-extension://") ||
          tab.url.startsWith("edge-extension://")
        ) {
          return;
        }

        // Update badge to show chatbot is available
        try {
          chrome.action.setBadgeText({
            text: "AI",
            tabId: tabId,
          });

          chrome.action.setBadgeBackgroundColor({
            color: "#667eea",
            tabId: tabId,
          });
        } catch (badgeError) {
          // Badge setting might fail on some pages, continue silently
          console.log("Could not set badge for tab:", tabId);
        }

        // Don't inject content script here - let it be handled by manifest
      } catch (error) {
        console.error("Error handling tab update:", error);
      }
    }
  }

  async ensureContentScript(tabId) {
    try {
      // Skip injection for restricted pages
      const tab = await chrome.tabs.get(tabId);
      if (
        tab.url.startsWith("chrome://") ||
        tab.url.startsWith("chrome-extension://")
      ) {
        return;
      }

      // Check if content script is already injected
      const results = await chrome.scripting.executeScript({
        target: { tabId },
        func: () => window.webPageChatbotInjected || false,
      });

      if (!results[0].result) {
        // Inject content script
        await chrome.scripting.executeScript({
          target: { tabId },
          files: ["content.js"],
        });

        // Mark as injected
        await chrome.scripting.executeScript({
          target: { tabId },
          func: () => {
            window.webPageChatbotInjected = true;
          },
        });
      }
    } catch (error) {
      console.error("Error ensuring content script:", error);
    }
  }

  async handleMessage(message, sender, sendResponse) {
    try {
      switch (message.type) {
        case "getPageAnalysis":
          const analysis = await this.analyzePageContent(message.content);
          sendResponse({ success: true, data: analysis });
          break;

        case "generateResponse":
          const response = await this.generateAIResponse(
            message.query,
            message.context
          );
          sendResponse({ success: true, data: response });
          break;

        case "saveConversation":
          await this.saveConversation(message.tabId, message.conversation);
          sendResponse({ success: true });
          break;

        case "getConversation":
          const conversation = await this.getConversation(message.tabId);
          sendResponse({ success: true, data: conversation });
          break;

        case "updateSettings":
          await this.updateSettings(message.settings);
          sendResponse({ success: true });
          break;

        case "getSettings":
          const settings = await this.getSettings();
          sendResponse({ success: true, data: settings });
          break;

        default:
          sendResponse({ success: false, error: "Unknown message type" });
      }
    } catch (error) {
      console.error("Error handling message:", error);
      sendResponse({ success: false, error: error.message });
    }
  }

  handleIconClick(tab) {
    // This is handled by the popup, but we can add additional logic here if needed
    console.log("Extension icon clicked for tab:", tab.id);
  }

  async handleContextMenu(info, tab) {
    try {
      switch (info.menuItemId) {
        case "chatbot-analyze-selection":
          await this.analyzeSelection(info.selectionText, tab);
          break;

        case "chatbot-analyze-page":
          await this.showChatbot(tab);
          break;
      }
    } catch (error) {
      console.error("Error handling context menu:", error);
    }
  }

  async analyzeSelection(selectedText, tab) {
    try {
      // Skip restricted pages
      if (
        tab.url.startsWith("chrome://") ||
        tab.url.startsWith("chrome-extension://")
      ) {
        return;
      }

      // Inject a temporary chat widget with the selected text
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        args: [selectedText],
        func: (text) => {
          // Create a temporary analysis popup
          const popup = document.createElement("div");
          popup.id = "selection-analysis-popup";
          popup.style.cssText = `
                        position: fixed;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 400px;
                        max-height: 300px;
                        background: white;
                        border: none;
                        border-radius: 10px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                        z-index: 10001;
                        font-family: Arial, sans-serif;
                        overflow: hidden;
                    `;

          popup.innerHTML = `
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 15px;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        ">
                            <span>🔍 Text Analysis</span>
                            <button onclick="this.closest('#selection-analysis-popup').remove()" style="
                                background: none;
                                border: none;
                                color: white;
                                font-size: 18px;
                                cursor: pointer;
                            ">×</button>
                        </div>
                        <div style="padding: 15px;">
                            <div style="
                                background: #f0f0f0;
                                padding: 10px;
                                border-radius: 5px;
                                margin-bottom: 10px;
                                font-size: 14px;
                                max-height: 80px;
                                overflow-y: auto;
                            ">"${text}"</div>
                            <div style="font-size: 14px; color: #666;">
                                Selected text analysis:<br>
                                • Length: ${text.length} characters<br>
                                • Words: ${text.split(/\s+/).length}<br>
                                • Appears to be ${
                                  text.length > 100
                                    ? "detailed content"
                                    : "brief text"
                                }<br>
                                <button onclick="
                                    this.closest('#selection-analysis-popup').remove();
                                    document.getElementById('webpage-chatbot-indicator')?.click();
                                " style="
                                    background: #007bff;
                                    color: white;
                                    border: none;
                                    padding: 8px 16px;
                                    border-radius: 5px;
                                    cursor: pointer;
                                    margin-top: 10px;
                                ">Chat about this text</button>
                            </div>
                        </div>
                    `;

          document.body.appendChild(popup);

          // Auto-remove after 10 seconds
          setTimeout(() => {
            if (popup.parentNode) {
              popup.remove();
            }
          }, 10000);
        },
      });
    } catch (error) {
      console.error("Error analyzing selection:", error);
    }
  }

  async showChatbot(tab) {
    try {
      // Skip restricted pages
      if (
        tab.url.startsWith("chrome://") ||
        tab.url.startsWith("chrome-extension://")
      ) {
        return;
      }

      // Activate the chatbot widget on the page
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          const indicator = document.getElementById(
            "webpage-chatbot-indicator"
          );
          if (indicator) {
            indicator.click();
          } else {
            // If indicator doesn't exist, create it
            if (typeof WebPageChatbot !== "undefined") {
              new WebPageChatbot();
            }
          }
        },
      });
    } catch (error) {
      console.error("Error showing chatbot:", error);
    }
  }

  async analyzePageContent(content) {
    // Enhanced NLP analysis
    const analysis = {
      sentiment: this.analyzeSentiment(content.text),
      topics: this.extractTopics(content.text),
      entities: this.extractEntities(content.text),
      readability: this.calculateReadability(content.text),
      structure: this.analyzeStructure(content),
      summary: this.generateSummary(content.text),
    };

    return analysis;
  }

  analyzeSentiment(text) {
    // Simple sentiment analysis
    const positiveWords = [
      "good",
      "great",
      "excellent",
      "amazing",
      "wonderful",
      "fantastic",
      "awesome",
      "best",
      "love",
      "perfect",
    ];
    const negativeWords = [
      "bad",
      "terrible",
      "awful",
      "horrible",
      "worst",
      "hate",
      "disgusting",
      "disappointing",
      "poor",
      "fail",
    ];

    const words = text.toLowerCase().split(/\W+/);
    let positiveScore = 0;
    let negativeScore = 0;

    words.forEach((word) => {
      if (positiveWords.includes(word)) positiveScore++;
      if (negativeWords.includes(word)) negativeScore++;
    });

    const total = positiveScore + negativeScore;
    if (total === 0) return "neutral";

    const positiveRatio = positiveScore / total;
    if (positiveRatio > 0.6) return "positive";
    if (positiveRatio < 0.4) return "negative";
    return "neutral";
  }

  extractTopics(text) {
    // Topic extraction using keyword frequency
    const words = text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((word) => word.length > 4);

    const frequency = {};
    words.forEach((word) => {
      frequency[word] = (frequency[word] || 0) + 1;
    });

    return Object.entries(frequency)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([word, freq]) => ({ topic: word, frequency: freq }));
  }

  extractEntities(text) {
    // Simple named entity recognition
    const entities = {
      dates: text.match(/\b\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}\b/g) || [],
      emails:
        text.match(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g) ||
        [],
      urls: text.match(/https?:\/\/[^\s]+/g) || [],
      phoneNumbers: text.match(/\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g) || [],
    };

    return entities;
  }

  calculateReadability(text) {
    // Simple readability score
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);
    const words = text.split(/\s+/).filter((w) => w.length > 0);
    const syllables = words.reduce(
      (total, word) => total + this.countSyllables(word),
      0
    );

    if (sentences.length === 0 || words.length === 0) return "unknown";

    const avgWordsPerSentence = words.length / sentences.length;
    const avgSyllablesPerWord = syllables / words.length;

    // Simplified Flesch Reading Ease
    const score =
      206.835 - 1.015 * avgWordsPerSentence - 84.6 * avgSyllablesPerWord;

    if (score >= 90) return "very easy";
    if (score >= 80) return "easy";
    if (score >= 70) return "fairly easy";
    if (score >= 60) return "standard";
    if (score >= 50) return "fairly difficult";
    if (score >= 30) return "difficult";
    return "very difficult";
  }

  countSyllables(word) {
    word = word.toLowerCase();
    if (word.length <= 3) return 1;
    word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, "");
    word = word.replace(/^y/, "");
    const matches = word.match(/[aeiouy]{1,2}/g);
    return matches ? matches.length : 1;
  }

  analyzeStructure(content) {
    return {
      headingLevels: content.headings.map((h) => h.level),
      linkCount: content.links.length,
      imageCount: content.images.length,
      contentLength: content.text.length,
      hasNavigation: content.headings.length > 3,
    };
  }

  generateSummary(text) {
    // Extractive summarization
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 20);
    if (sentences.length <= 2) return text.substring(0, 200) + "...";

    // Score sentences by position and keyword frequency
    const scoredSentences = sentences.map((sentence, index) => {
      let score = 0;

      // Position score (first and last sentences are important)
      if (index === 0) score += 3;
      if (index === sentences.length - 1) score += 2;

      // Length score (prefer medium-length sentences)
      const wordCount = sentence.split(/\s+/).length;
      if (wordCount >= 10 && wordCount <= 25) score += 2;

      return { sentence: sentence.trim(), score, index };
    });

    // Return top 2 sentences
    return (
      scoredSentences
        .sort((a, b) => b.score - a.score)
        .slice(0, 2)
        .sort((a, b) => a.index - b.index)
        .map((item) => item.sentence)
        .join(". ") + "."
    );
  }

  async generateAIResponse(query, context) {
    // This would integrate with actual AI APIs in a production version
    // For now, we'll use enhanced rule-based responses

    const analysis = await this.analyzePageContent(context);

    // Generate contextual response
    let response = this.generateContextualResponse(query, context, analysis);

    return {
      text: response,
      confidence: 0.8,
      sources: ["page content analysis"],
      analysis: analysis,
    };
  }

  generateContextualResponse(query, context, analysis) {
    const queryLower = query.toLowerCase();

    // Sentiment-based responses
    if (
      queryLower.includes("feel") ||
      queryLower.includes("tone") ||
      queryLower.includes("mood")
    ) {
      return `The overall tone of this page appears to be ${
        analysis.sentiment
      }. ${this.getSentimentExplanation(analysis.sentiment)}`;
    }

    // Topic-based responses
    if (
      queryLower.includes("topic") ||
      queryLower.includes("about") ||
      queryLower.includes("theme")
    ) {
      const topTopics = analysis.topics
        .slice(0, 5)
        .map((t) => t.topic)
        .join(", ");
      return `The main topics covered include: ${topTopics}. These appear most frequently in the content and likely represent the key themes.`;
    }

    // Structure-based responses
    if (
      queryLower.includes("structure") ||
      queryLower.includes("organized") ||
      queryLower.includes("layout")
    ) {
      return `This page has ${
        analysis.structure.headingLevels.length
      } headings, ${analysis.structure.linkCount} links, and ${
        analysis.structure.imageCount
      } images. The content is ${
        analysis.structure.hasNavigation
          ? "well-structured with clear navigation"
          : "presented in a simple format"
      }.`;
    }

    // Readability responses
    if (
      queryLower.includes("difficult") ||
      queryLower.includes("easy") ||
      queryLower.includes("understand")
    ) {
      return `The readability level is ${
        analysis.readability
      }. ${this.getReadabilityExplanation(analysis.readability)}`;
    }

    // Default enhanced response
    return `Based on my analysis of "${context.title}", this page focuses on ${
      analysis.topics[0]?.topic || "the main topic"
    }. ${analysis.summary} The content appears to be ${
      analysis.readability
    } to read with a ${analysis.sentiment} tone overall.`;
  }

  getSentimentExplanation(sentiment) {
    switch (sentiment) {
      case "positive":
        return "The language used is generally optimistic and favorable.";
      case "negative":
        return "The content contains some critical or negative language.";
      default:
        return "The language is balanced and neutral in tone.";
    }
  }

  getReadabilityExplanation(readability) {
    switch (readability) {
      case "very easy":
        return "The content should be easily understood by most readers.";
      case "easy":
        return "The content is accessible and straightforward.";
      case "standard":
        return "The content is at a typical reading level.";
      case "difficult":
        return "The content may require careful reading and attention.";
      default:
        return "The content complexity varies throughout the page.";
    }
  }

  async saveConversation(tabId, conversation) {
    try {
      const key = `conversation_${tabId}`;
      await chrome.storage.local.set({ [key]: conversation });
    } catch (error) {
      console.error("Error saving conversation:", error);
    }
  }

  async getConversation(tabId) {
    try {
      const key = `conversation_${tabId}`;
      const result = await chrome.storage.local.get(key);
      return result[key] || [];
    } catch (error) {
      console.error("Error getting conversation:", error);
      return [];
    }
  }

  async updateSettings(settings) {
    try {
      const stored = await chrome.storage.local.get("chatbotSettings");
      const updated = { ...stored.chatbotSettings, ...settings };
      await chrome.storage.local.set({ chatbotSettings: updated });
    } catch (error) {
      console.error("Error updating settings:", error);
    }
  }

  async getSettings() {
    try {
      const result = await chrome.storage.local.get("chatbotSettings");
      return result.chatbotSettings || {};
    } catch (error) {
      console.error("Error getting settings:", error);
      return {};
    }
  }
}

// Initialize the background service
new ChatbotBackground();
