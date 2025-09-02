@echo off
echo Installing Urdu Document QA system...
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing core packages...

REM Install packages one by one
echo Installing FastAPI...
pip install fastapi

echo Installing Uvicorn...
pip install uvicorn

echo Installing Jinja2...
pip install jinja2

echo Installing Python-multipart...
pip install python-multipart

echo Installing PyTorch...
pip install torch

echo Installing Transformers...
pip install transformers

echo Installing Sentence Transformers...
pip install sentence-transformers

echo Installing ChromaDB...
pip install chromadb

echo Installing PyMuPDF...
pip install PyMuPDF

echo Installing NumPy...
pip install numpy

echo Installing Requests...
pip install requests

echo Installing Python-dotenv...
pip install python-dotenv

echo.
echo Installing optional packages...

REM Try to install LangChain packages
echo Installing LangChain...
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-huggingface

echo.
echo Creating directories...
mkdir data 2>nul
mkdir chroma_db 2>nul
mkdir logs 2>nul

echo.
echo Creating configuration file...
if not exist .env (
    copy .env.example .env
)

echo.
echo Setup completed!
echo.
echo To run the application:
echo   python app.py
echo.
echo Or with uvicorn:
echo   uvicorn app:app --reload
echo.
pause
