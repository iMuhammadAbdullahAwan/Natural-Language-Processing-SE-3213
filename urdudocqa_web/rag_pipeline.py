import os
import logging
import time
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import torch
from huggingface_hub import snapshot_download
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FallbackRetriever(BaseRetriever):
    """A retriever that tries a primary retriever first and falls back to another if no docs are returned."""

    primary: BaseRetriever
    fallback: BaseRetriever

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Avoid passing run_manager to keep compatibility across langchain versions
        docs = self.primary.get_relevant_documents(query)
        if not docs:
            return self.fallback.get_relevant_documents(query)
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Avoid passing run_manager to keep compatibility across langchain versions
        docs = await self.primary.aget_relevant_documents(query)
        if not docs:
            return await self.fallback.aget_relevant_documents(query)
        return docs

class LocalRAGPipeline:
    """
    Advanced RAG Pipeline using local lightweight Hugging Face models
    """
    
    def __init__(self, model_type: str = "flan-t5-small"):
        self.model_type = model_type
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.llm = None
        self.conv_chain = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.persist_root = "chroma_db"
        self.current_persist_dir = None

        logger.info(f"Initializing LocalRAG with device: {self.device}")

        # Initialize embeddings
        self._initialize_embeddings()

        # Initialize local LLM
        self._initialize_local_llm()
    
    def _initialize_embeddings(self):
        """Initialize lightweight multilingual embedding model"""
        try:
            # Use consistent embedding model with known dimensions
            # This model has 384 dimensions - keep consistent
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': self.device,
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 16
                }
            )
            logger.info(f"Embeddings initialized: {model_name} (384 dimensions)")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            # Fallback to the same model
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': self.device}
                )
            except Exception as e2:
                logger.error(f"Fallback embeddings failed: {str(e2)}")
                raise
    
    def _initialize_local_llm(self):
        """Initialize lightweight local LLM models"""
        try:
            if self.model_type == "flan-t5-small":
                self._setup_flan_t5_small()
            elif self.model_type == "distilgpt2":
                self._setup_distilgpt2()
            elif self.model_type == "flan-t5-base":
                self._setup_flan_t5_base()
            else:
                self._setup_flan_t5_small()  # Default fallback
                
            logger.info(f"Local LLM initialized: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            self._setup_simple_qa_model()
    
    def _setup_flan_t5_small(self):
        """Setup Google's FLAN-T5 Small (77M parameters) - Good for Q&A"""
        model_name = "google/flan-t5-small"
        
        # Create text2text generation pipeline
        text_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if self.device == "cuda" else -1,
            model_kwargs={"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        )
        # Prefer constrained, deterministic decoding with enough headroom for output tokens
        self.llm = HuggingFacePipeline(
            pipeline=text_pipeline,
            model_kwargs={
                "max_new_tokens": 128,
                "min_new_tokens": 16,
                "do_sample": False,
                "num_beams": 2,
                "clean_up_tokenization_spaces": True,
            },
        )
    
    def _setup_flan_t5_base(self):
        """Setup Google's FLAN-T5 Base (250M parameters) - Better quality"""
        model_name = "google/flan-t5-base"
        
        text_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if self.device == "cuda" else -1,
            model_kwargs={"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        )
        self.llm = HuggingFacePipeline(
            pipeline=text_pipeline,
            model_kwargs={
                "max_new_tokens": 128,
                "min_new_tokens": 16,
                "do_sample": False,
                "num_beams": 2,
                "clean_up_tokenization_spaces": True,
            },
        )
    
    def _setup_distilgpt2(self):
        """Setup DistilGPT2 (82M parameters) - Fast generation"""
        model_name = "distilgpt2"
        
        text_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        self.llm = HuggingFacePipeline(
            pipeline=text_pipeline,
            model_kwargs={
                "max_new_tokens": 64,
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.95,
                "pad_token_id": 50256,
                "clean_up_tokenization_spaces": True,
            },
        )
    
    def _setup_simple_qa_model(self):
        """Fallback to a very simple Q&A model"""
        try:
            model_name = "distilbert-base-uncased-distilled-squad"
            
            qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1
            )
            
            # Wrap QA pipeline for compatibility
            class QAPipelineWrapper:
                def __init__(self, qa_pipeline):
                    self.qa_pipeline = qa_pipeline
                
                def __call__(self, prompt):
                    try:
                        # Extract question and context from prompt
                        if "سوال:" in prompt and "سیاق و سباق:" in prompt:
                            context_start = prompt.find("سیاق و سباق:") + len("سیاق و سباق:")
                            question_start = prompt.find("سوال:") + len("سوال:")
                            
                            context = prompt[context_start:prompt.find("سوال:")].strip()
                            question = prompt[question_start:prompt.find("ہدایات:")].strip()
                            
                            if context and question:
                                result = self.qa_pipeline(question=question, context=context)
                                return result['answer']
                        
                        return "میں اس سوال کا جواب نہیں دے سکتا۔"
                    except:
                        return "میں اس سوال کا جواب نہیں دے سکتا۔"
            
            self.llm = QAPipelineWrapper(qa_pipeline)
            
        except Exception as e:
            logger.error(f"Simple QA model failed: {str(e)}")
            self.llm = SimpleLLM()
    
    def create_vector_store(self, documents: List[Document], persist_dir: str = "chroma_db") -> Chroma:
        """Create and persist vector store using a unique subfolder to avoid Windows file locks."""
        t0 = time.time()
        try:
            # Ensure root exists and create a unique subdirectory for this index
            os.makedirs(persist_dir, exist_ok=True)
            unique_dir = os.path.join(persist_dir, str(uuid.uuid4()))
            os.makedirs(unique_dir, exist_ok=True)
            # Build fresh store at unique path
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=unique_dir,
            )
            # Persist call is optional on newer Chroma; keep for compatibility
            try:
                self.vector_store.persist()
            except Exception:
                pass
            self.current_persist_dir = unique_dir
            self.persist_root = persist_dir
            # Cleanup older vector stores to limit disk usage
            self._cleanup_old_vector_stores(persist_dir, max_keep=3)
            logger.info(f"Vector store created with {len(documents)} documents in {int((time.time()-t0)*1000)} ms")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            try:
                # Try once more with a new unique dir
                os.makedirs(persist_dir, exist_ok=True)
                unique_dir = os.path.join(persist_dir, str(uuid.uuid4()))
                os.makedirs(unique_dir, exist_ok=True)
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=unique_dir,
                )
                try:
                    self.vector_store.persist()
                except Exception:
                    pass
                self.current_persist_dir = unique_dir
                self.persist_root = persist_dir
                logger.info("Vector store recreated successfully after cleanup")
                return self.vector_store
            except Exception as e2:
                logger.error(f"Failed to recreate vector store: {str(e2)}")
                raise
    
    def load_vector_store(self, persist_dir: str = "chroma_db") -> Chroma:
        """Load the most recent vector store under the root and validate via a small search."""
        try:
            path = self._latest_vector_store_dir(persist_dir)
            if path is None:
                raise FileNotFoundError(f"No persisted vector stores found under {persist_dir}")
            self.vector_store = Chroma(
                persist_directory=path,
                embedding_function=self.embeddings,
            )
            _ = self.vector_store.similarity_search("test", k=1)
            logger.info("Vector store loaded successfully")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            if "dimension" in str(e).lower():
                logger.info("Dimension mismatch detected. Vector store needs to be recreated.")
                self._safe_rmtree(persist_dir)
                logger.info("Vector store root cleared. Please upload your PDF again.")
            raise

    def _latest_vector_store_dir(self, root: str) -> Optional[str]:
        try:
            if not os.path.isdir(root):
                return None
            subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            if not subdirs:
                return None
            subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return subdirs[0]
        except Exception:
            return None

    def _safe_rmtree(self, path: str, retries: int = 5, delay: float = 0.3) -> None:
        import shutil, time, stat
        if not os.path.exists(path):
            return
        def onerror(func, p, exc_info):
            try:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            except Exception:
                pass
        for i in range(retries):
            try:
                shutil.rmtree(path, onerror=onerror)
                return
            except Exception as e:
                if i == retries - 1:
                    logger.warning(f"Failed to remove {path} after retries: {e}")
                    return
                time.sleep(delay)

    def _cleanup_old_vector_stores(self, root: str, max_keep: int = 3) -> None:
        try:
            if not os.path.isdir(root):
                return
            subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            if len(subdirs) <= max_keep:
                return
            subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for old in subdirs[max_keep:]:
                self._safe_rmtree(old)
        except Exception as e:
            logger.warning(f"Cleanup of old vector stores failed: {e}")
    
    def setup_qa_chain(self, search_kwargs: Dict[str, Any] = None) -> RetrievalQA:
        """Setup QA chain with optimized prompt for local models"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        # Default search parameters
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        # Create retriever; if thresholded search is requested, add a fallback to plain similarity
        if search_kwargs.get("score_threshold", 0) > 0:
            primary = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": search_kwargs.get("k", 5),
                    "score_threshold": search_kwargs["score_threshold"],
                },
            )
            fallback = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": search_kwargs.get("k", 5)},
            )
            retriever = FallbackRetriever(primary=primary, fallback=fallback)
        else:
            # Use regular similarity search without threshold
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": search_kwargs.get("k", 5)}
            )

        # Keep prompt Urdu-focused and concise
        prompt_template = """
سیاق و سباق:
{context}

سوال: {question}

ہدایات:
- صرف دیے گئے سیاق و سباق کی بنیاد پر جواب دیں
- اگر جواب موجود نہیں تو کہیں: "میں اس سوال کا جواب دستیاب معلومات سے نہیں دے سکتا"
- جامع مگر مختصر جواب اردو میں دیں

جواب:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        # Create QA chain
        t0 = time.time()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        # Setup conversational chain using same retriever and prompt
        try:
            self.conv_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT},
            )
        except Exception as e:
            logger.warning(f"Conversational chain setup failed, falling back to simple QA chain only: {e}")

        logger.info(f"QA chain setup completed in {int((time.time()-t0)*1000)} ms")
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system and include basic timing info."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
        
        try:
            t0 = time.time()
            # Use modern invoke API if available; fallback to call
            try:
                result = self.qa_chain.invoke({"query": question})
            except Exception:
                result = self.qa_chain({"query": question})
            total_ms = int((time.time() - t0) * 1000)
            
            # Enhanced response with metadata
            answer_text = result.get("result") or ""
            # Fallback: if empty, try to synthesize a brief extractive answer from sources
            if not answer_text.strip():
                srcs = result.get("source_documents", [])
                if srcs:
                    answer_text = (srcs[0].page_content or "").strip()[:400]
            if not answer_text.strip():
                answer_text = "میں دیے گئے سیاق و سباق کی بنیاد پر واضح جواب تشکیل نہیں دے سکا۔ براہ کرم سوال مزید واضح کریں۔"

            response = {
                "answer": answer_text,
                "source_documents": result.get("source_documents", []),
                "sources_count": len(result.get("source_documents", [])),
                "confidence_score": self._calculate_confidence(result),
                "model_used": self.model_type,
                "timings_ms": {"qa_total_ms": total_ms},
            }
            
            logger.info(f"Query processed successfully with {self.model_type}")
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"خرابی: {str(e)}",
                "source_documents": [],
                "sources_count": 0,
                "confidence_score": 0.0,
                "model_used": self.model_type,
                "timings_ms": {"qa_total_ms": 0},
            }

    def query_with_history(self, question: str, chat_history: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        """Query using conversational chain with chat history [(user, ai), ...]."""
        if not self.conv_chain:
            # Fallback to stateless query
            return self.query(question)

        history = chat_history or []
        try:
            t0 = time.time()
            # LangChain expects a list of [user, ai] pairs or list of tuples
            # Convert [[user, ai], ...] into list of tuples as required by LC
            hist_tuples = []
            for pair in history:
                try:
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        u = str(pair[0]) if pair[0] is not None else ""
                        a = str(pair[1]) if pair[1] is not None else ""
                        hist_tuples.append((u, a))
                except Exception:
                    continue
            payload = {"question": question, "chat_history": hist_tuples}
            try:
                result = self.conv_chain.invoke(payload)
            except Exception:
                result = self.conv_chain(payload)
            total_ms = int((time.time() - t0) * 1000)

            answer_text = result.get("answer") or result.get("result") or ""
            # If conversational answer is empty, retry once with stateless QA as fallback
            if not answer_text.strip():
                try:
                    stateless = self.query(question)
                    answer_text = stateless.get("answer", "")
                    result = {**result, "source_documents": stateless.get("source_documents", [])}
                except Exception:
                    pass
            if not answer_text.strip():
                # As a last resort, extract from top source
                srcs = result.get("source_documents", [])
                if srcs:
                    answer_text = (srcs[0].page_content or "").strip()[:400]
            if not answer_text.strip():
                answer_text = "میں دیے گئے سیاق و سباق کی بنیاد پر واضح جواب تشکیل نہیں دے سکا۔ براہ کرم سوال مزید واضح کریں۔"
            response = {
                "answer": answer_text,
                "source_documents": result.get("source_documents", []),
                "sources_count": len(result.get("source_documents", [])),
                "confidence_score": self._calculate_confidence(result),
                "model_used": self.model_type,
                "timings_ms": {"qa_total_ms": total_ms},
            }
            return response
        except Exception as e:
            logger.error(f"Error processing conversational query: {str(e)}")
            return {
                "answer": f"خرابی: {str(e)}",
                "source_documents": [],
                "sources_count": 0,
                "confidence_score": 0.0,
                "model_used": self.model_type,
                "timings_ms": {"qa_total_ms": 0},
            }
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score based on source documents"""
        source_docs = result.get("source_documents", [])
        if not source_docs:
            return 0.0
        
        # Simple confidence calculation
        return min(len(source_docs) / 3.0, 1.0)

class SimpleLLM:
    """
    Fallback simple LLM for basic functionality
    """
    
    def __call__(self, prompt: str) -> str:
        """Simple text processing for fallback"""
        try:
            # Extract context and question from prompt
            if "Context:" in prompt and "Question:" in prompt:
                context_start = prompt.find("Context:") + len("Context:")
                question_start = prompt.find("Question:") + len("Question:")
                
                context = prompt[context_start:prompt.find("Question:")].strip()
                question_end = prompt.find("Based on") if "Based on" in prompt else len(prompt)
                question = prompt[question_start:question_end].strip()
                
                return self._generate_simple_answer(question, context)
            
            return "میں اس وقت آپ کے سوال کا جواب نہیں دے سکتا۔"
        
        except Exception as e:
            return f"خرابی: {str(e)}"
    
    def _generate_simple_answer(self, question: str, context: str) -> str:
        """Generate simple answer based on keyword matching"""
        if not context.strip():
            return "میں اس سوال کا جواب دستیاب معلومات سے نہیں دے سکتا۔"
        
        # Simple keyword matching
        question_words = question.lower().split()
        context_sentences = context.split('۔') if '۔' in context else context.split('.')
        
        # Find most relevant sentence
        best_sentence = ""
        max_matches = 0
        
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for word in question_words if word in sentence_lower)
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence.strip()
        
        if best_sentence and max_matches > 0:
            return best_sentence
        else:
            # Return first part of context
            return context[:300] + "..." if len(context) > 300 else context

# Legacy functions for backward compatibility
def create_or_load_vector_db(chunks, persist_dir="chroma_db"):
    """Legacy function for backward compatibility"""
    try:
        # Convert strings to Documents if needed
        if isinstance(chunks[0], str):
            documents = [Document(page_content=chunk, metadata={"source": "legacy"}) for chunk in chunks]
        else:
            documents = chunks
        
        # Use consistent embeddings (same as main pipeline)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Check if directory exists and clean it to avoid dimension mismatch
        import shutil
        if os.path.exists(persist_dir):
            logger.info(f"Removing existing vector store to avoid dimension mismatch: {persist_dir}")
            shutil.rmtree(persist_dir)
        
        vector_db = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)
        vector_db.persist()
        return vector_db
    except Exception as e:
        logger.error(f"Error in legacy vector DB creation: {str(e)}")
        raise

def setup_rag_chain(vector_db):
    """Legacy function for backward compatibility"""
    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        # Use simple LLM for legacy support
        llm = SimpleLLM()
        
        class LegacyRAGChain:
            def __init__(self, retriever, llm):
                self.retriever = retriever
                self.llm = llm
            
            def __call__(self, inputs):
                query = inputs.get('query', '')
                docs = self.retriever.get_relevant_documents(query)
                context = " ".join([doc.page_content for doc in docs])
                
                if context.strip():
                    result = self.llm._generate_simple_answer(query, context)
                else:
                    result = "میں اس سوال کا جواب دستیاب معلومات سے نہیں دے سکتا۔"
                
                return {
                    'result': result,
                    'source_documents': docs
                }
        
        return LegacyRAGChain(retriever, llm)
    except Exception as e:
        logger.error(f"Error in legacy RAG chain setup: {str(e)}")
        raise
    """
    Advanced RAG Pipeline with multiple LLM options and enhanced retrieval
    """
    
    def __init__(self, llm_type: str = "openai", model_name: Optional[str] = None):
        self.llm_type = llm_type
        self.model_name = model_name
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.llm = None
        
        # Initialize embeddings (using multilingual model for better Urdu support)
        self._initialize_embeddings()
        
        # Initialize LLM
        self._initialize_llm()
    
    def _initialize_embeddings(self):
        """Initialize embedding model with multilingual support"""
        try:
            # Use multilingual embedding model for better Urdu support
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            # Fallback to simpler model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def _initialize_llm(self):
        """Initialize LLM based on specified type"""
        try:
            if self.llm_type.lower() == "openai":
                self._initialize_openai_llm()
            elif self.llm_type.lower() == "ollama":
                self._initialize_ollama_llm()
            elif self.llm_type.lower() == "huggingface":
                self._initialize_huggingface_llm()
            else:
                self._initialize_fallback_llm()
                
            logger.info(f"LLM initialized successfully: {self.llm_type}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            self._initialize_fallback_llm()
    
    def _initialize_openai_llm(self):
        """Initialize OpenAI LLM"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.llm = ChatOpenAI(
            model_name=self.model_name or "gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key=api_key
        )
    
    def _initialize_ollama_llm(self):
        """Initialize Ollama LLM (local)"""
        self.llm = Ollama(
            model=self.model_name or "llama2",
            temperature=0.1
        )
    
    def _initialize_huggingface_llm(self):
        """Initialize HuggingFace LLM"""
        model_name = self.model_name or "microsoft/DialoGPT-medium"
        
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        
        # Create text generation pipeline
        text_generation = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_length=512,
            temperature=0.1,
            do_sample=True,
            device=device
        )
        
        self.llm = HuggingFacePipeline(pipeline=text_generation)
    
    def _initialize_fallback_llm(self):
        """Initialize fallback LLM for basic functionality"""
        self.llm = SimpleLLM()
    
    def create_vector_store(self, documents: List[Document], persist_dir: str = "chroma_db") -> Chroma:
        """
        Create and persist vector store from documents
        """
        try:
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            
            # Persist the vector store
            self.vector_store.persist()
            
            logger.info(f"Vector store created with {len(documents)} documents")
            return self.vector_store
        
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self, persist_dir: str = "chroma_db") -> Chroma:
        """
        Load existing vector store
        """
        try:
            self.vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            logger.info("Vector store loaded successfully")
            return self.vector_store
        
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def setup_qa_chain(self, search_kwargs: Dict[str, Any] = None) -> RetrievalQA:
        """
        Setup QA chain with custom prompt for Urdu
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        # Default search parameters
        if search_kwargs is None:
            search_kwargs = {"k": 5, "score_threshold": 0.5}
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )
        
        # Custom prompt template for Urdu Q&A
        urdu_prompt_template = """
آپ ایک مددگار اردو زبان کا AI اسسٹنٹ ہیں۔ دیے گئے سیاق و سباق کی بنیاد پر سوال کا جواب دیں۔

سیاق و سباق:
{context}

سوال: {question}

ہدایات:
- صرف دیے گئے سیاق و سباق کی بنیاد پر جواب دیں
- اگر جواب سیاق میں موجود نہیں ہے تو کہیں "میں اس سوال کا جواب دستیاب معلومات سے نہیں دے سکتا"
- اردو میں واضح اور مفصل جواب دیں
- جواب درست اور مفید ہو

جواب:"""

        PROMPT = PromptTemplate(
            template=urdu_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("QA chain setup completed")
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
        
        try:
            result = self.qa_chain({"query": question})
            
            # Enhanced response with metadata
            response = {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "sources_count": len(result.get("source_documents", [])),
                "confidence_score": self._calculate_confidence(result)
            }
            
            logger.info(f"Query processed successfully: {question[:50]}...")
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"خرابی: {str(e)}",
                "source_documents": [],
                "sources_count": 0,
                "confidence_score": 0.0
            }
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on source documents
        """
        source_docs = result.get("source_documents", [])
        if not source_docs:
            return 0.0
        
        # Simple confidence calculation based on number of sources
        return min(len(source_docs) / 5.0, 1.0)

class SimpleLLM:
    """
    Fallback simple LLM for basic functionality
    """
    
    def __call__(self, prompt: str) -> str:
        """
        Simple text processing for fallback
        """
        try:
            # Extract context and question from prompt
            if "سیاق و سباق:" in prompt and "سوال:" in prompt:
                context_start = prompt.find("سیاق و سباق:") + len("سیاق و سباق:")
                question_start = prompt.find("سوال:") + len("سوال:")
                
                context = prompt[context_start:prompt.find("سوال:")].strip()
                question = prompt[question_start:prompt.find("ہدایات:")].strip()
                
                return self._generate_simple_answer(question, context)
            
            return "میں اس وقت آپ کے سوال کا جواب نہیں دے سکتا۔"
        
        except Exception as e:
            return f"خرابی: {str(e)}"
    
    def _generate_simple_answer(self, question: str, context: str) -> str:
        """
        Generate simple answer based on keyword matching
        """
        if not context.strip():
            return "میں اس سوال کا جواب دستیاب معلومات سے نہیں دے سکتا۔"
        
        # Simple keyword matching
        question_words = question.lower().split()
        context_sentences = context.split('۔')
        
        # Find most relevant sentence
        best_sentence = ""
        max_matches = 0
        
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for word in question_words if word in sentence_lower)
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence.strip()
        
        if best_sentence and max_matches > 0:
            return best_sentence
        else:
            # Return first part of context
            return context[:300] + "..." if len(context) > 300 else context

# Legacy functions for backward compatibility
def create_or_load_vector_db(chunks, persist_dir="chroma_db"):
    """Legacy function for backward compatibility"""
    try:
        # Convert strings to Documents if needed
        if isinstance(chunks[0], str):
            documents = [Document(page_content=chunk, metadata={"source": "legacy"}) for chunk in chunks]
        else:
            documents = chunks
        
        # Use consistent embeddings (same as main pipeline)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        # Use unique subdirectory to avoid Windows file locks and dim mismatches
        os.makedirs(persist_dir, exist_ok=True)
        unique_dir = os.path.join(persist_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)
        vector_db = Chroma.from_documents(documents, embeddings, persist_directory=unique_dir)
        try:
            vector_db.persist()
        except Exception:
            pass
        return vector_db
    except Exception as e:
        logger.error(f"Error in legacy vector DB creation: {str(e)}")
        raise

def setup_rag_chain(vector_db):
    """Legacy function for backward compatibility"""
    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        # Use simple LLM for legacy support
        llm = SimpleLLM()
        
        class LegacyRAGChain:
            def __init__(self, retriever, llm):
                self.retriever = retriever
                self.llm = llm
            
            def __call__(self, inputs):
                query = inputs.get('query', '')
                docs = self.retriever.get_relevant_documents(query)
                context = " ".join([doc.page_content for doc in docs])
                
                if context.strip():
                    result = self.llm._generate_simple_answer(query, context)
                else:
                    result = "میں اس سوال کا جواب دستیاب معلومات سے نہیں دے سکتا۔"
                
                return {
                    'result': result,
                    'source_documents': docs
                }
        
        return LegacyRAGChain(retriever, llm)
    except Exception as e:
        logger.error(f"Error in legacy RAG chain setup: {str(e)}")
        raise