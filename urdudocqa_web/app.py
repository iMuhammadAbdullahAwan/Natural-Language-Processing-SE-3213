from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
from preprocess import (
    extract_text_from_pdf, 
    clean_urdu_text, 
    advanced_chunk_text,
    extract_metadata_from_pdf,
    chunk_text  # Legacy support
)
from rag_pipeline import LocalRAGPipeline, create_or_load_vector_db, setup_rag_chain
from simple_generator import generate_text as simple_generate
from metrics import MetricsManager

# Load .env early
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="اردو دستاویز سوالات و جوابات", version="2.0.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "change-me-secret"))

# Global variables
rag_pipeline: Optional[LocalRAGPipeline] = None
qa_chain = None  # Legacy support
message = ""
current_document_info: Dict[str, Any] = {}
# Metrics
metrics = MetricsManager(log_dir="logs")

# Configuration for local models
RAG_CONFIG = {
    "model_type": os.getenv("MODEL_TYPE", "flan-t5-small"),  # flan-t5-small, flan-t5-base, distilgpt2
    "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),  # More chunks improve recall
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
    "search_k": int(os.getenv("SEARCH_K", "5")),  # Retrieve more for better recall
    "score_threshold": float(os.getenv("SCORE_THRESHOLD", "0.0"))  # 0 disables thresholding
}

def initialize_rag_pipeline():
    """Initialize the local RAG pipeline"""
    global rag_pipeline
    try:
        logger.info(f"Initializing LocalRAG with model: {RAG_CONFIG['model_type']}")
        rag_pipeline = LocalRAGPipeline(model_type=RAG_CONFIG["model_type"])
        logger.info("Local RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {str(e)}")
        rag_pipeline = None

# Initialize RAG pipeline on FastAPI startup (avoids reloader double-import issues)
@app.on_event("startup")
async def _startup_init_rag():
    initialize_rag_pipeline()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with current document info"""
    doc_info = ""
    if current_document_info:
        doc_info = f"فائل لوڈ شدہ: {current_document_info.get('title', 'نامعلوم')} ({current_document_info.get('page_count', 0)} صفحات)"
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "message": message, 
        "answer": "",
    "document_info": doc_info,
    "history": request.session.get("history", [])
    })

@app.get("/simple", response_class=HTMLResponse)
async def simple_page(request: Request):
    """Minimal Urdu text generation page."""
    return templates.TemplateResponse("simple.html", {
        "request": request,
        "output": "",
        "prompt": "",
        "model": os.getenv("SIMPLE_MODEL", "google/mt5-small"),
    })

@app.post("/simple/generate", response_class=HTMLResponse)
async def simple_generate_route(
    request: Request,
    prompt: str = Form(...),
    max_new_tokens: int = Form(128),
    temperature: float = Form(0.7),
    num_beams: int = Form(1),
):
    result = simple_generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_beams=num_beams,
    )
    return templates.TemplateResponse("simple.html", {
        "request": request,
        "output": result.get("text", ""),
        "prompt": prompt,
        "model": result.get("model", "google/mt5-small"),
    })

@app.post("/")
async def home_post():
    """Gracefully handle accidental POST to root by redirecting to GET /."""
    return RedirectResponse(url="/", status_code=303)

@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """Advanced PDF upload and processing"""
    global rag_pipeline, qa_chain, message, current_document_info
    timings = {}
    t_total_start = t_stage = __import__("time").time()
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            message = "خرابی: صرف PDF فائلیں اپ لوڈ کریں۔"
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "message": message, 
                "answer": ""
            })
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        pdf_path = f"data/{file.filename}"
        
        # Save uploaded file
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        timings["save_ms"] = int((__import__("time").time() - t_stage) * 1000)
        t_stage = __import__("time").time()
        logger.info(f"File uploaded successfully: {file.filename}")
        
        # Extract metadata
        current_document_info = extract_metadata_from_pdf(pdf_path)
        timings["metadata_ms"] = int((__import__("time").time() - t_stage) * 1000)
        t_stage = __import__("time").time()
        
        # Process PDF
        raw_text = extract_text_from_pdf(pdf_path)
        timings["extract_ms"] = int((__import__("time").time() - t_stage) * 1000)
        t_stage = __import__("time").time()
        cleaned_text = clean_urdu_text(raw_text)
        timings["clean_ms"] = int((__import__("time").time() - t_stage) * 1000)
        t_stage = __import__("time").time()
        
        if len(cleaned_text.strip()) < 100:
            message = "خرابی: فائل میں کافی متن نہیں ملا۔"
            metrics.record_upload(file.filename, current_document_info.get("page_count", 0), 0, {**timings, "total_ms": int((__import__("time").time() - t_total_start) * 1000)}, status="error")
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "message": message, 
                "answer": ""
            })
        
        # Use advanced RAG pipeline if available
        if rag_pipeline:
            try:
                # Advanced chunking
                documents = advanced_chunk_text(
                    cleaned_text, 
                    chunk_size=RAG_CONFIG["chunk_size"],
                    chunk_overlap=RAG_CONFIG["chunk_overlap"]
                )
                timings["chunk_ms"] = int((__import__("time").time() - t_stage) * 1000)
                t_stage = __import__("time").time()
                
                # Create vector store
                rag_pipeline.create_vector_store(documents)
                timings["vector_ms"] = int((__import__("time").time() - t_stage) * 1000)
                t_stage = __import__("time").time()
                
                # Setup QA chain
                search_kwargs = {
                    "k": RAG_CONFIG["search_k"]
                }
                # Only add score_threshold if it's > 0
                if RAG_CONFIG["score_threshold"] > 0:
                    search_kwargs["score_threshold"] = RAG_CONFIG["score_threshold"]
                rag_pipeline.setup_qa_chain(search_kwargs)
                timings["qa_setup_ms"] = int((__import__("time").time() - t_stage) * 1000)
                
                message = f"PDF '{file.filename}' کامیابی سے پروسیس ہو گئی! ({len(documents)} حصے بنائے گئے)"
                logger.info(f"Advanced processing completed: {len(documents)} chunks created")
                timings["total_ms"] = int((__import__("time").time() - t_total_start) * 1000)
                metrics.record_upload(
                    file.filename,
                    current_document_info.get("page_count", 0),
                    len(documents),
                    timings,
                    status="success",
                )
                
            except Exception as e:
                logger.error(f"Advanced processing failed: {str(e)}")
                
                # Check if it's a dimension mismatch error
                if "dimension" in str(e).lower():
                    message = f"ڈیٹابیس کی خرابی: پرانا ڈیٹا صاف کر کے دوبارہ کوشش کریں۔ خرابی: {str(e)}"
                    # Try to clear the database automatically
                    try:
                        if os.path.exists("chroma_db"):
                            shutil.rmtree("chroma_db")
                        message += " (ڈیٹابیس صاف کر دیا گیا - دوبارہ اپ لوڈ کریں)"
                    except:
                        pass
                else:
                    # Fallback to legacy processing
                    await _legacy_processing(cleaned_text, file.filename)

                # record as error
                timings["total_ms"] = int((__import__("time").time() - t_total_start) * 1000)
                metrics.record_upload(
                    file.filename,
                    current_document_info.get("page_count", 0),
                    0,
                    timings,
                    status="error",
                )
                
        else:
            # Legacy processing
            await _legacy_processing(cleaned_text, file.filename)
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        message = f"خرابی: {str(e)}"
        timings["total_ms"] = int((__import__("time").time() - t_total_start) * 1000)
        metrics.record_upload(file.filename, current_document_info.get("page_count", 0), 0, timings, status="error")
    
    doc_info = ""
    if current_document_info:
        doc_info = f"فائل لوڈ شدہ: {current_document_info.get('title', file.filename)} ({current_document_info.get('page_count', 0)} صفحات)"
    
    # Reset chat history on new upload
    try:
        request.session["history"] = []
    except Exception:
        pass

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "message": message, 
        "answer": "",
        "document_info": doc_info,
        "history": request.session.get("history", [])
    })

async def _legacy_processing(cleaned_text: str, filename: str):
    """Legacy processing fallback"""
    global qa_chain, message
    
    try:
        chunks = chunk_text(cleaned_text)
        vector_db = create_or_load_vector_db(chunks)
        qa_chain = setup_rag_chain(vector_db)
        message = f"PDF '{filename}' کامیابی سے پروسیس ہو گئی! (Legacy mode - {len(chunks)} حصے)"
        logger.info("Legacy processing completed successfully")
    except Exception as e:
        logger.error(f"Legacy processing failed: {str(e)}")
        message = f"خرابی: {str(e)}"

@app.post("/query")
async def query(request: Request, question: str = Form(...)):
    """Enhanced query processing"""
    global rag_pipeline, qa_chain, message
    timings = {}
    t0 = __import__("time").time()
    answer = ""
    confidence_score = 0.0
    sources_count = 0
    
    # Validate input
    if not question.strip():
        message = "خرابی: سوال خالی نہیں ہو سکتا۔"
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "message": message, 
            "answer": answer,
            "user_question": question
        })
    
    # Check if system is ready
    if not rag_pipeline and not qa_chain:
        message = "پہلے PDF فائل اپ لوڈ کریں!"
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "message": message, 
            "answer": answer,
            "user_question": question
        })
    
    try:
        # Use advanced RAG pipeline if available
        if rag_pipeline and rag_pipeline.qa_chain:
            logger.info(f"Processing query with advanced pipeline: {question[:50]}...")
            history = request.session.get("history", [])
            result = rag_pipeline.query_with_history(question, history)
            answer = result["answer"]
            confidence_score = result["confidence_score"]
            sources_count = result["sources_count"]
            timings = {"qa_total_ms": result.get("timings_ms", {}).get("qa_total_ms", 0)}
            message = ""  # Clear any previous messages
            # Update history
            try:
                history.append([question, answer])
                request.session["history"] = history
            except Exception:
                pass
            
        # Fallback to legacy system
        elif qa_chain:
            logger.info(f"Processing query with legacy pipeline: {question[:50]}...")
            result = qa_chain({"query": question})
            answer = result['result']
            timings = {"qa_total_ms": int((__import__("time").time() - t0) * 1000)}
            message = ""
            
        else:
            message = "سسٹم تیار نہیں ہے۔ دوبارہ PDF اپ لوڈ کریں۔"
            
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        answer = f"خرابی: سوال پروسیس کرنے میں مسئلہ - {str(e)}"
        timings = {"qa_total_ms": int((__import__("time").time() - t0) * 1000)}
    
    # Add metadata to response if available
    response_data = {
        "request": request, 
        "message": message, 
        "answer": answer,
        "user_question": question,
        "history": request.session.get("history", [])
    }
    
    # Add confidence and source info if available
    if confidence_score > 0:
        confidence_text = "اعتماد: " + ("زیادہ" if confidence_score > 0.7 else "متوسط" if confidence_score > 0.4 else "کم")
        response_data["confidence"] = confidence_text
        response_data["sources_info"] = f"({sources_count} ذرائع سے)"
    
    # Record metrics
    metrics.record_query(
        question=question,
        success=not answer.startswith("خرابی:"),
        durations_ms={"qa_total_ms": timings.get("qa_total_ms", 0)},
        confidence=confidence_score if confidence_score else None,
        sources_count=sources_count if sources_count else None,
        model=(rag_pipeline.model_type if rag_pipeline else "legacy"),
        answer_chars=len(answer) if answer else 0,
    )

    return templates.TemplateResponse("index.html", response_data)

@app.post("/reset")
async def reset_chat(request: Request):
    """Reset chat history for the current session."""
    try:
        request.session["history"] = []
    except Exception:
        pass
    return RedirectResponse(url="/", status_code=303)

@app.post("/clear-database")
async def clear_database(request: Request):
    """Clear vector database endpoint"""
    global rag_pipeline, qa_chain, current_document_info, message
    
    try:
        # Clear vector database
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
            logger.info("Vector database cleared via web interface")
        
        # Reset global variables
        rag_pipeline = None
        qa_chain = None
        current_document_info = {}
        
        # Reinitialize RAG pipeline
        initialize_rag_pipeline()
        
        message = "ڈیٹابیس کامیابی سے صاف کر دیا گیا! اب نئی PDF اپ لوڈ کریں۔"
        
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        message = f"ڈیٹابیس صاف کرنے میں خرابی: {str(e)}"
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "message": message, 
        "answer": "",
        "document_info": ""
    })

@app.get("/health")
async def health_check():
    """Health check endpoint with model information"""
    import torch
    
    gpu_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    return {
        "status": "healthy",
        "rag_pipeline_ready": rag_pipeline is not None and rag_pipeline.qa_chain is not None,
        "legacy_ready": qa_chain is not None,
        "model_type": RAG_CONFIG["model_type"],
        "document_loaded": bool(current_document_info),
    "gpu_info": gpu_info,
        "config": RAG_CONFIG
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "rag_config": RAG_CONFIG,
        "document_info": current_document_info,
        "available_models": [
            {
                "name": "flan-t5-small",
                "description": "Fast, lightweight (77M params)",
                "recommended": True
            },
            {
                "name": "flan-t5-base", 
                "description": "Better quality (250M params)",
                "recommended": False
            },
            {
                "name": "distilgpt2",
                "description": "Creative responses (82M params)", 
                "recommended": False
            }
        ]
    }

@app.get("/metrics")
async def get_metrics():
    """JSON metrics for dashboards."""
    return JSONResponse(metrics.to_json())

@app.get("/metrics/report", response_class=HTMLResponse)
async def metrics_report(request: Request):
    """Lightweight HTML metrics report."""
    data = metrics.to_json()
    return templates.TemplateResponse(
        "metrics.html",
        {
            "request": request,
            "summary": data.get("summary", {}),
            "recent_uploads": data.get("recent_uploads", []),
            "recent_queries": data.get("recent_queries", []),
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)