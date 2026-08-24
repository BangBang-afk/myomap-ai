"""Multilingual report processing — XLM-R tokenizer wrapper."""
import os
from typing import Dict

KAGGLE_INPUT = "/kaggle/input/rsna-knee-abnormality-detection"

def is_kaggle() -> bool:
    return os.path.exists(KAGGLE_INPUT)

def get_tokenizer(model_name: str = "xlm-roberta-base"):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    except Exception as e:
        print(f"[text_proc] tokenizer load failed {e}, fallback to whitespace")
        return None

def clean_report(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # keep original multilingual text — XLM-R handles 12 langs natively
    return text.strip()[:4000]

def tokenize_report(text: str, tokenizer, max_len: int = 256) -> Dict:
    text = clean_report(text)
    if tokenizer is None:
        return {"input_ids": [], "attention_mask": [], "raw": text}
    enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    return {k: v.squeeze(0) for k, v in enc.items()}
