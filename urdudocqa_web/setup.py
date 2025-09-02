#!/usr/bin/env python3
"""
Setup script for Urdu Document Q&A system
Automatically downloads and configures local models
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher required")
        return False
    return True

def install_requirements():
    """Install required packages with error handling"""
    try:
        logger.info("Installing requirements...")
        
        # First, try to install with exact requirements
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            logger.info("Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            logger.warning("Standard installation failed, trying individual packages...")
            
            # Install core packages individually with looser version constraints
            core_packages = [
                "fastapi>=0.100.0",
                "uvicorn[standard]>=0.20.0", 
                "jinja2>=3.0.0",
                "python-multipart>=0.0.5",
                "torch>=2.0.0",
                "transformers>=4.30.0",
                "sentence-transformers>=2.2.0",
                "huggingface-hub>=0.15.0",
                "chromadb>=0.4.0",
                "PyMuPDF>=1.20.0",
                "numpy>=1.20.0",
                "requests>=2.25.0",
                "python-dotenv>=0.19.0",
                "accelerate>=0.20.0"
            ]
            
            for package in core_packages:
                try:
                    logger.info(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {package}: {e}")
                    continue
            
            # Try to install LangChain packages
            langchain_packages = [
                "langchain>=0.1.0",
                "langchain-community>=0.0.10", 
                "langchain-core>=0.1.0",
                "langchain-huggingface>=0.0.1"
            ]
            
            for package in langchain_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {package}: {e}")
                    continue
            
            logger.info("Core packages installed with best effort")
            return True
            
    except Exception as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def download_models():
    """Pre-download models for faster startup"""
    try:
        logger.info("Pre-downloading models (this may take a few minutes)...")
        
        # Import after requirements are installed
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from sentence_transformers import SentenceTransformer
        
        # Download embedding model
        logger.info("Downloading embedding model...")
        SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Download FLAN-T5 Small
        logger.info("Downloading FLAN-T5 Small model...")
        AutoTokenizer.from_pretrained("google/flan-t5-small")
        AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        logger.info("Models downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        logger.info("Models will be downloaded on first use")
        return False

def create_env_file():
    """Create .env file from template"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        logger.info("Creating .env file...")
        env_file.write_text(env_example.read_text())
        logger.info(".env file created")

def create_directories():
    """Create necessary directories"""
    directories = ["data", "chroma_db", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info("GPU acceleration will be used")
        else:
            logger.info("No GPU detected. CPU mode will be used")
    except ImportError:
        logger.info("PyTorch not installed yet. GPU check will happen after installation")

def main():
    """Main setup function"""
    logger.info("Starting Urdu Document Q&A Setup...")
    
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install requirements
    if not install_requirements():
        logger.error("Setup failed during requirements installation")
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Download models (optional)
    download_models()
    
    logger.info("Setup completed successfully!")
    logger.info("To start the application, run: python app.py")
    logger.info("Or with uvicorn: uvicorn app:app --reload")

if __name__ == "__main__":
    main()
