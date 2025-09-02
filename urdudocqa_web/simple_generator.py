import os
import threading
from typing import Dict, Any

from transformers import pipeline
import torch

_lock = threading.Lock()
_gen = None


def _init_generator():
    global _gen
    if _gen is not None:
        return _gen
    with _lock:
        if _gen is not None:
            return _gen
        # Choose a small, multilingual seq2seq model that can handle Urdu
        model_name = os.getenv("SIMPLE_MODEL", "google/mt5-small")
        device = 0 if torch.cuda.is_available() else -1
        _gen = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )
    return _gen


def generate_text(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    num_beams: int = 1,
) -> Dict[str, Any]:
    gen = _init_generator()
    # Make sure prompt is non-empty and trimmed
    text = (prompt or "").strip()
    if not text:
        return {"text": "براہ کرم کوئی ہدایت یا سوال درج کریں۔"}

    # Run generation
    out = gen(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=(num_beams == 1),
        temperature=temperature,
        num_beams=num_beams,
        clean_up_tokenization_spaces=True,
    )
    # HF pipeline returns list of dicts with 'generated_text'
    result = out[0].get("generated_text", "").strip()
    if not result:
        result = "میں اس وقت متن تیار نہیں کر سکا۔ براہ کرم دوبارہ کوشش کریں۔"
    return {"text": result, "model": gen.model.name_or_path}
