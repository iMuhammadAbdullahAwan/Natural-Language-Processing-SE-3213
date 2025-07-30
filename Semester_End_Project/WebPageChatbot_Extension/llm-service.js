// llm-service.js - LLM service using free APIs and LangChain-like functionality

class LLMService {
  constructor() {
    // Use configuration if available
    this.config = window.LLMConfig || {};

    // Initialize endpoints with config values
    this.apiEndpoint =
      this.config.huggingFace?.models?.primary ||
      "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium";
    this.fallbackEndpoint =
      this.config.huggingFace?.models?.fallback ||
      "https://api-inference.huggingface.co/models/google/flan-t5-base";
    this.ollamaEndpoint =
      this.config.ollama?.endpoint || "http://localhost:11434/api/generate";

    // Initialize with retry mechanism from config
    this.maxRetries = this.config.reliability?.maxRetries || 3;
    this.retryDelay = this.config.reliability?.retryDelay || 1000;
    this.timeoutMs = this.config.reliability?.timeoutMs || 10000;

    // Debug settings
    this.debug = this.config.debug?.enableLogging || false;
    this.showPrompts = this.config.debug?.showPrompts || false;
  }

  // Main method to generate response using LLM
  async generateResponse(question, context, pageContent) {
    if (this.debug) {
      console.log("🤖 LLM Service: Processing question:", question);
    }

    // Check for simulation mode
    if (this.config.debug?.simulateFailure) {
      return this.enhancedRuleBasedResponse(question, context, pageContent);
    }

    const prompt = this.constructPrompt(question, context, pageContent);

    if (this.showPrompts) {
      console.log("📝 Prompt sent to LLM:", prompt);
    }

    const primaryService = this.config.primaryService || "huggingface";

    // Try primary service first
    try {
      let response = null;

      switch (primaryService) {
        case "openai":
          if (this.config.openai?.enabled && this.config.openai?.apiKey) {
            response = await this.queryOpenAI(prompt);
          }
          break;
        case "ollama":
          response = await this.queryOllama(prompt);
          break;
        case "huggingface":
        default:
          response = await this.queryHuggingFace(prompt);
          break;
      }

      if (response) {
        if (this.debug) {
          console.log(
            "✅ LLM Response received:",
            response.substring(0, 100) + "..."
          );
        }
        return this.postProcessResponse(response, question);
      }
    } catch (error) {
      if (this.debug) {
        console.log(
          `❌ Primary service (${primaryService}) failed:`,
          error.message
        );
      }
    }

    // Try fallback services
    const fallbackServices = ["huggingface", "ollama", "openai"].filter(
      (s) => s !== primaryService
    );

    for (const service of fallbackServices) {
      try {
        let response = null;

        switch (service) {
          case "openai":
            if (this.config.openai?.enabled && this.config.openai?.apiKey) {
              response = await this.queryOpenAI(prompt);
            }
            break;
          case "ollama":
            response = await this.queryOllama(prompt);
            break;
          case "huggingface":
            response = await this.queryHuggingFace(prompt);
            break;
        }

        if (response) {
          if (this.debug) {
            console.log(`✅ Fallback service (${service}) succeeded`);
          }
          return this.postProcessResponse(response, question);
        }
      } catch (error) {
        if (this.debug) {
          console.log(
            `❌ Fallback service (${service}) failed:`,
            error.message
          );
        }
      }
    }

    // Enhanced rule-based fallback with better context understanding
    if (this.debug) {
      console.log("🔄 Using enhanced rule-based fallback");
    }
    return this.enhancedRuleBasedResponse(question, context, pageContent);
  }

  constructPrompt(question, context, pageContent) {
    const maxLength = this.config.processing?.maxContextLength || 1000;
    const contextText = context.mainText.substring(0, maxLength);

    const prompt = `Context: You are an AI assistant helping users understand web page content.

Page Information:
- Title: ${pageContent.title}
- URL: ${pageContent.url}
- Content Summary: ${context.summary}
- Key Topics: ${context.keyPhrases.join(", ")}

Page Content (excerpt):
${contextText}

User Question: ${question}

Please provide a helpful, accurate answer based on the page content. If the information isn't available on the page, say so clearly. Keep the response concise and relevant.

Answer:`;

    return prompt;
  }

  async queryOpenAI(prompt) {
    if (!this.config.openai?.enabled || !this.config.openai?.apiKey) {
      throw new Error("OpenAI not configured");
    }

    try {
      const response = await fetch(this.config.openai.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.config.openai.apiKey}`,
        },
        body: JSON.stringify({
          model: this.config.openai.model || "gpt-3.5-turbo",
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
          max_tokens: this.config.openai.parameters?.max_tokens || 200,
          temperature: this.config.openai.parameters?.temperature || 0.7,
          top_p: this.config.openai.parameters?.top_p || 0.9,
        }),
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status}`);
      }

      const data = await response.json();
      return data.choices?.[0]?.message?.content || null;
    } catch (error) {
      if (this.debug) {
        console.error("OpenAI API error:", error);
      }
      throw error;
    }
  }

  async queryHuggingFace(prompt, retries = 0) {
    try {
      const headers = {
        "Content-Type": "application/json",
      };

      // Add authorization if token is configured
      if (this.config.huggingFace?.apiToken) {
        headers["Authorization"] = `Bearer ${this.config.huggingFace.apiToken}`;
      }

      const parameters = this.config.huggingFace?.parameters || {
        max_length: 200,
        temperature: 0.7,
        do_sample: true,
        top_p: 0.9,
      };

      const response = await fetch(this.apiEndpoint, {
        method: "POST",
        headers,
        body: JSON.stringify({
          inputs: prompt,
          parameters,
        }),
      });

      if (!response.ok) {
        if (response.status === 503 && retries < this.maxRetries) {
          // Model is loading, wait and retry
          if (this.debug) {
            console.log(
              `⏳ HuggingFace model loading, retry ${retries + 1}/${
                this.maxRetries
              }`
            );
          }
          await this.delay(this.retryDelay * (retries + 1));
          return this.queryHuggingFace(prompt, retries + 1);
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (Array.isArray(data) && data[0]?.generated_text) {
        return data[0].generated_text.replace(prompt, "").trim();
      }

      return null;
    } catch (error) {
      if (this.debug) {
        console.error("Hugging Face API error:", error);
      }
      if (retries < this.maxRetries) {
        await this.delay(this.retryDelay);
        return this.queryHuggingFace(prompt, retries + 1);
      }
      throw error;
    }
  }

  async queryOllama(prompt) {
    try {
      const model = this.config.ollama?.preferredModel || "llama2";
      const options = this.config.ollama?.options || {
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 200,
      };

      const response = await fetch(this.ollamaEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: model,
          prompt: prompt,
          stream: false,
          options: options,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Ollama API error: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();
      return data.response;
    } catch (error) {
      if (this.debug) {
        console.error("Ollama API error:", error);
      }
      throw error;
    }
  }

  // LangChain-inspired text processing
  processWithLangChain(question, context, pageContent) {
    // Implement LangChain-like document processing
    const documents = this.splitIntoChunks(context.mainText);
    const relevantChunks = this.findRelevantChunks(question, documents);
    const retrievedContext = relevantChunks.join("\n\n");

    return this.generateContextualResponse(
      question,
      retrievedContext,
      pageContent
    );
  }

  splitIntoChunks(text, chunkSize = null, overlap = null) {
    const configuredChunkSize =
      chunkSize || this.config.processing?.chunkSize || 500;
    const configuredOverlap =
      overlap || this.config.processing?.chunkOverlap || 50;

    const chunks = [];
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);

    let currentChunk = "";
    let currentLength = 0;

    for (const sentence of sentences) {
      if (
        currentLength + sentence.length > configuredChunkSize &&
        currentChunk
      ) {
        chunks.push(currentChunk.trim());
        // Keep some overlap
        const words = currentChunk.split(" ");
        currentChunk =
          words.slice(-configuredOverlap).join(" ") + " " + sentence;
        currentLength = currentChunk.length;
      } else {
        currentChunk += sentence + ". ";
        currentLength = currentChunk.length;
      }
    }

    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }

    return chunks;
  }

  findRelevantChunks(question, chunks) {
    const maxChunks = this.config.processing?.maxRelevantChunks || 3;
    const questionWords = question
      .toLowerCase()
      .split(/\s+/)
      .filter((word) => word.length > 3 && !this.getStopWords().has(word));

    const scoredChunks = chunks.map((chunk) => {
      const chunkLower = chunk.toLowerCase();
      let score = 0;

      questionWords.forEach((word) => {
        const regex = new RegExp(`\\b${word}\\b`, "gi");
        const matches = chunkLower.match(regex);
        if (matches) {
          score += matches.length;
        }
      });

      return { chunk, score };
    });

    return scoredChunks
      .sort((a, b) => b.score - a.score)
      .slice(0, maxChunks)
      .filter((item) => item.score > 0)
      .map((item) => item.chunk);
  }

  generateContextualResponse(question, context, pageContent) {
    const questionLower = question.toLowerCase();

    // Use the retrieved context to generate better responses
    if (context.trim().length === 0) {
      return "I couldn't find specific information about that on this page. Could you try rephrasing your question or asking about something else from the page?";
    }

    // Extract the most relevant sentence from context
    const sentences = context
      .split(/[.!?]+/)
      .filter((s) => s.trim().length > 20);
    const questionWords = questionLower
      .split(/\s+/)
      .filter((w) => w.length > 3);

    let bestSentence = sentences[0];
    let bestScore = 0;

    sentences.forEach((sentence) => {
      const sentenceLower = sentence.toLowerCase();
      let score = 0;

      questionWords.forEach((word) => {
        if (sentenceLower.includes(word)) {
          score++;
        }
      });

      if (score > bestScore) {
        bestScore = score;
        bestSentence = sentence;
      }
    });

    return `Based on the page content: ${bestSentence.trim()}.`;
  }

  enhancedRuleBasedResponse(question, context, pageContent) {
    const questionLower = question.toLowerCase();

    // Enhanced pattern matching with context
    if (
      questionLower.includes("when") &&
      (questionLower.includes("found") ||
        questionLower.includes("start") ||
        questionLower.includes("establish"))
    ) {
      return this.findDateInformation(context.mainText);
    }

    if (
      questionLower.includes("who") &&
      (questionLower.includes("found") ||
        questionLower.includes("ceo") ||
        questionLower.includes("founder"))
    ) {
      return this.findPersonInformation(context.mainText);
    }

    if (
      questionLower.includes("where") &&
      (questionLower.includes("locat") || questionLower.includes("headquart"))
    ) {
      return this.findLocationInformation(context.mainText);
    }

    if (questionLower.includes("what") && questionLower.includes("about")) {
      return `This page is about "${pageContent.title}". ${context.summary}`;
    }

    // Use LangChain-like processing as fallback
    return this.processWithLangChain(question, context, pageContent);
  }

  findDateInformation(text) {
    const datePatterns = [
      /founded?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /established?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /created?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /started?\s+(?:in\s+|on\s+)?(\d{4})/i,
      /(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})/i,
    ];

    for (const pattern of datePatterns) {
      const match = text.match(pattern);
      if (match) {
        const sentences = text.split(/[.!?]+/);
        const relevantSentence = sentences.find(
          (sentence) =>
            pattern.test(sentence) &&
            (sentence.toLowerCase().includes("found") ||
              sentence.toLowerCase().includes("establish") ||
              sentence.toLowerCase().includes("start") ||
              sentence.toLowerCase().includes("creat"))
        );

        if (relevantSentence) {
          return `Based on the page content: ${relevantSentence.trim()}.`;
        }
      }
    }
    return "I couldn't find specific founding date information on this page.";
  }

  findPersonInformation(text) {
    const personPatterns = [
      /founded?\s+by\s+([A-Z][a-zA-Z\s]+)/i,
      /founder[s]?\s*:?\s*([A-Z][a-zA-Z\s]+)/i,
      /ceo[s]?\s*:?\s*([A-Z][a-zA-Z\s]+)/i,
    ];

    for (const pattern of personPatterns) {
      const match = text.match(pattern);
      if (match) {
        const sentences = text.split(/[.!?]+/);
        const relevantSentence = sentences.find((sentence) =>
          pattern.test(sentence)
        );
        if (relevantSentence) {
          return `Based on the page content: ${relevantSentence.trim()}.`;
        }
      }
    }
    return "I couldn't find specific founder or CEO information on this page.";
  }

  findLocationInformation(text) {
    const locationPatterns = [
      /headquarter(?:ed|s)?\s+(?:in|at)\s+([A-Z][a-zA-Z\s,]+)/i,
      /located?\s+(?:in|at)\s+([A-Z][a-zA-Z\s,]+)/i,
      /based?\s+(?:in|at)\s+([A-Z][a-zA-Z\s,]+)/i,
    ];

    for (const pattern of locationPatterns) {
      const match = text.match(pattern);
      if (match) {
        const sentences = text.split(/[.!?]+/);
        const relevantSentence = sentences.find((sentence) =>
          pattern.test(sentence)
        );
        if (relevantSentence) {
          return `Based on the page content: ${relevantSentence.trim()}.`;
        }
      }
    }
    return "I couldn't find specific location information on this page.";
  }

  postProcessResponse(response, question) {
    // Clean up and format the LLM response
    let cleaned = response.trim();

    // Remove any prompt leakage
    const promptMarkers = [
      "Context:",
      "Page Information:",
      "User Question:",
      "Answer:",
      "Assistant:",
      "Human:",
    ];
    promptMarkers.forEach((marker) => {
      const index = cleaned.indexOf(marker);
      if (index !== -1) {
        cleaned = cleaned.substring(0, index).trim();
      }
    });

    // Ensure response ends properly
    if (cleaned && !cleaned.match(/[.!?]$/)) {
      cleaned += ".";
    }

    // Limit response length based on configuration
    const maxLength = this.config.processing?.maxResponseLength || 300;
    if (cleaned.length > maxLength) {
      cleaned = cleaned.substring(0, maxLength - 3) + "...";
    }

    // Log final response if debug mode
    if (this.debug) {
      console.log("🎯 Final processed response:", cleaned);
    }

    return (
      cleaned ||
      "I'm having trouble processing that question. Could you try rephrasing it?"
    );
  }

  getStopWords() {
    return new Set([
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
      "been",
      "be",
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
      "can",
      "shall",
      "must",
      "this",
      "that",
      "these",
      "those",
      "i",
      "you",
      "he",
      "she",
      "it",
      "we",
      "they",
      "me",
      "him",
      "her",
      "us",
      "them",
      "my",
      "your",
      "his",
      "her",
      "its",
      "our",
      "their",
    ]);
  }

  delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// Export for use in popup.js
window.LLMService = LLMService;
