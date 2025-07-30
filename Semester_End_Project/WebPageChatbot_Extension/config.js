// config.js - Configuration for LLM services
// Edit this file to customize your AI assistant

window.LLMConfig = {
  // Primary LLM service preference (huggingface, ollama, fallback)
  primaryService: "huggingface",

  // Hugging Face settings
  huggingFace: {
    // Get your free token from https://huggingface.co/settings/tokens
    apiToken: "", // Leave empty to use free tier (with rate limits)

    // Model endpoints (don't change unless you know what you're doing)
    models: {
      primary:
        "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
      fallback:
        "https://api-inference.huggingface.co/models/google/flan-t5-base",
      qa: "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2",
    },

    // Generation parameters
    parameters: {
      max_length: 200,
      temperature: 0.7,
      do_sample: true,
      top_p: 0.9,
    },
  },

  // Ollama settings (requires local installation)
  ollama: {
    endpoint: "http://localhost:11434/api/generate",

    // Available models (install with: ollama pull <model-name>)
    preferredModel: "llama2", // Options: llama2, mistral, codellama, neural-chat

    // Generation parameters
    options: {
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 200,
      stop: ["Human:", "Assistant:", "\n\n"],
    },
  },

  // OpenAI settings (requires API key and credits)
  openai: {
    enabled: false, // Set to true if you want to use OpenAI
    apiKey: "", // Your OpenAI API key
    endpoint: "https://api.openai.com/v1/chat/completions",
    model: "gpt-3.5-turbo", // or 'gpt-4' for better quality

    parameters: {
      temperature: 0.7,
      max_tokens: 200,
      top_p: 0.9,
    },
  },

  // Response processing settings
  processing: {
    maxContextLength: 1000, // Max characters from page to send to LLM
    maxResponseLength: 300, // Max response length
    enableCache: true, // Cache responses for similar questions

    // LangChain-like settings
    chunkSize: 500, // Size of text chunks for processing
    chunkOverlap: 50, // Overlap between chunks
    maxRelevantChunks: 3, // Number of relevant chunks to include
  },

  // Retry and error handling
  reliability: {
    maxRetries: 3,
    retryDelay: 1000, // milliseconds
    timeoutMs: 10000, // 10 seconds
    enableFallback: true, // Use rule-based responses if LLM fails
  },

  // Debug and development
  debug: {
    enableLogging: true, // Console logging for troubleshooting
    showPrompts: false, // Show the prompts sent to LLM (for debugging)
    simulateFailure: false, // Force fallback mode for testing
  },
};

// Quick setup functions
window.LLMConfig.setupHuggingFace = function (token) {
  this.huggingFace.apiToken = token;
  this.primaryService = "huggingface";
  console.log("✅ Hugging Face configured with token");
};

window.LLMConfig.setupOllama = function (model = "llama2") {
  this.ollama.preferredModel = model;
  this.primaryService = "ollama";
  console.log(`✅ Ollama configured with model: ${model}`);
};

window.LLMConfig.setupOpenAI = function (apiKey, model = "gpt-3.5-turbo") {
  this.openai.apiKey = apiKey;
  this.openai.model = model;
  this.openai.enabled = true;
  this.primaryService = "openai";
  console.log(`✅ OpenAI configured with model: ${model}`);
};

// Validation
window.LLMConfig.validate = function () {
  const errors = [];

  if (this.primaryService === "openai" && !this.openai.apiKey) {
    errors.push("OpenAI API key is required when using OpenAI service");
  }

  if (this.processing.maxContextLength > 2000) {
    errors.push("Context length too large, may cause API errors");
  }

  if (errors.length > 0) {
    console.warn("LLM Configuration issues:", errors);
    return false;
  }

  console.log("✅ LLM Configuration is valid");
  return true;
};
