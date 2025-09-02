import fitz  # PyMuPDF
import unicodedata
import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Advanced PDF text extraction with better formatting preservation
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num, page in enumerate(doc):
            # Extract text with better formatting
            page_text = page.get_text("text")
            
            # Add page separator for better chunking
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        doc.close()
        logger.info(f"Successfully extracted text from {pdf_path}")
        return text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def clean_urdu_text(text: str) -> str:
    """
    Advanced Urdu text cleaning and normalization
    """
    try:
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove diacritics but preserve main text
        text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]', '', text)
        
        # Clean up whitespace while preserving structure
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
        
        # Remove excessive whitespace but keep paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        logger.info("Text cleaning completed successfully")
        return text.strip()
    
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text

def advanced_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Advanced text chunking using LangChain's RecursiveCharacterTextSplitter
    """
    try:
        # Initialize the text splitter with optimal settings for Urdu
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                "۔",     # Urdu sentence ending
                ".",     # English sentence ending
                "!",     # Exclamation
                "?",     # Question mark
                "؟",     # Urdu question mark
                ";",     # Semicolon
                ",",     # Comma
                " ",     # Space
                ""       # Character level
            ]
        )
        
        # Split text into documents
        documents = text_splitter.create_documents([text])
        
        # Add metadata to documents
        for i, doc in enumerate(documents):
            doc.metadata = {
                "chunk_id": i,
                "chunk_size": len(doc.page_content),
                "source": "pdf_document"
            }
        
        logger.info(f"Successfully created {len(documents)} text chunks")
        return documents
    
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        # Fallback to simple chunking
        return simple_chunk_text(text, chunk_size)

def simple_chunk_text(text: str, chunk_size: int = 500) -> List[Document]:
    """
    Fallback simple text chunking method
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_text = ' '.join(words[i:i+chunk_size])
        doc = Document(
            page_content=chunk_text,
            metadata={"chunk_id": i//chunk_size, "source": "pdf_document"}
        )
        chunks.append(doc)
    
    return chunks

def extract_metadata_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF document
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
            "page_count": doc.page_count,
            "file_path": pdf_path
        }
        doc.close()
        return metadata
    
    except Exception as e:
        logger.error(f"Error extracting PDF metadata: {str(e)}")
        return {"file_path": pdf_path, "page_count": 0}

# Legacy function for backward compatibility
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Legacy chunking function - converts Document objects to strings
    """
    documents = advanced_chunk_text(text, chunk_size)
    return [doc.page_content for doc in documents]