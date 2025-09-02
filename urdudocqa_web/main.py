"""
ASGI entrypoint so you can run:
    uvicorn main:app --reload

This simply imports the FastAPI instance from app.py.
"""

from app import app  # noqa: F401
